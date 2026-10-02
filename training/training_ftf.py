import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

TRAINING_SET = current + '/data/dataset_ftf.csv'
MODELS_DIR = root + '/models/ftf_detector'

os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = [
    "payload_entropy_mean", "payload_entropy_cv", "payload_size_entropy",
    "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
    "c_used", "paf", "ft_psi", "durations_entropy",
    "pkts_entropy", "bytes_entropy", "mean_duration", "cvd",
    "mean_pkts", "cvp", "mean_bytes", "cvb",
]
TARGET = "label"
SCENARIO_COL = "scenario_id"
GROUP_COL = "round"

# Colonna con il profilo di attacco (aggressive / moderate / stealth), con
# valore "benigno" per le righe di traffico non malevolo. Rinomina la
# colonna/il valore se nel CSV si chiamano diversamente.
ATTACK_PROFILE_COL = "attack_profile"
BENIGN_PROFILE_LABEL = "benigni"

RANDOM_SEED = 42

# Frazione dello scenario-set di train riservata a validation per l'early
# stopping del training finale (mai vista durante tuning/CV, mai usata come
# test).
VAL_SIZE = 0.15

N_TRIALS = 50
N_SPLITS = 5
CV_EARLY_STOPPING_ROUNDS = 50
FINAL_EARLY_STOPPING_ROUNDS = 50

BERT_N_TRIALS = 5
BERT_CV_EPOCHS = 5
BERT_CV_PATIENCE = 3
BERT_FINAL_EPOCHS = 30
BERT_FINAL_PATIENCE = 5

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------------------------------
# 3. HF-BERT su feature tokenizzate - tuning Optuna con StratifiedGroupKFold
#    per scenario_id + training finale con early stopping su validation
# --------------------------------------------------------------------------
def _tokenize_features(X_fit, *X_others, n_bins):
    """Discretizza ogni feature continua in n_bins bin (quantili, fit SOLO su
    X_fit) e assegna a ogni (indice_feature, bin) un id di vocabolario
    univoco. Antepone a ogni riga un token [CLS] dedicato (id riservato,
    l'ultimo del vocabolario). Ritorna gli id per X_fit seguiti da quelli per
    ciascun array in X_others, nello stesso ordine, piu' vocab_size (incluso
    [CLS]) e il discretizzatore."""
    discretizer = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
    )
    fit_bins = discretizer.fit_transform(X_fit).astype(np.int64)

    n_features = X_fit.shape[1]
    offsets = (np.arange(n_features) * n_bins).astype(np.int64)

    cls_id = n_features * n_bins          # primo id libero dopo i token (feature, bin)
    vocab_size = n_features * n_bins + 1  # +1 per [CLS]

    def _with_cls(bins):
        ids = bins + offsets
        cls_col = np.full((ids.shape[0], 1), cls_id, dtype=np.int64)
        return np.concatenate([cls_col, ids], axis=1)   # shape: (n, n_features + 1)

    outputs = [_with_cls(fit_bins)]
    for X_other in X_others:
        other_bins = discretizer.transform(X_other).astype(np.int64)
        outputs.append(_with_cls(other_bins))

    return (*outputs, vocab_size, discretizer)

class FlowTokenDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": self.labels[idx],
        }


def _compute_hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _class_weights(y):
    values = y.values if hasattr(y, "values") else np.asarray(y)
    counts = np.bincount(values, minlength=2)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _build_bert_config(vocab_size, n_features, params):
    hidden_size = params['num_attention_heads'] * params['head_dim']
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=params['num_hidden_layers'],
        num_attention_heads=params['num_attention_heads'],
        intermediate_size=params['intermediate_size'],
        max_position_embeddings=n_features + 2,
        type_vocab_size=1,
        num_labels=2,
    )


def tune_hf_bert(X_train, y_train, groups_train, profiles_train, n_trials=BERT_N_TRIALS, n_splits=N_SPLITS):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train, n_splits)
    n_features = X_train.shape[1]

    def objective(trial):
        params = {
            'n_bins': trial.suggest_categorical('n_bins', [8, 12, 16]),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [2, 4, 8]),
            'head_dim': trial.suggest_categorical('head_dim', [8, 16, 24]),
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 2, 6),
            'intermediate_size': trial.suggest_categorical('intermediate_size', [128, 256, 512]),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        }

        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            train_ids, val_ids, vocab_size, _ = _tokenize_features(
                X_tr.values, X_val.values, n_bins=params['n_bins']
            )

            config = _build_bert_config(vocab_size, n_features, params)
            model = BertForSequenceClassification(config)

            train_ds = FlowTokenDataset(train_ids, y_tr.values)
            val_ds = FlowTokenDataset(val_ids, y_val.values)
            class_weights = _class_weights(y_tr)

            with tempfile.TemporaryDirectory(dir=MODELS_DIR) as tmp_dir:
                training_args = TrainingArguments(
                    output_dir=tmp_dir,
                    num_train_epochs=BERT_CV_EPOCHS,
                    per_device_train_batch_size=params['batch_size'],
                    per_device_eval_batch_size=params['batch_size'],
                    learning_rate=params['learning_rate'],
                    weight_decay=params['weight_decay'],
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    save_total_limit=1,
                    load_best_model_at_end=True,
                    metric_for_best_model="f1",
                    greater_is_better=True,
                    logging_strategy="no",
                    report_to=[],
                    seed=RANDOM_SEED,
                    disable_tqdm=True,
                    dataloader_pin_memory=False
                )

                trainer = WeightedTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    compute_metrics=_compute_hf_metrics,
                    class_weights=class_weights,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=BERT_CV_PATIENCE)],
                )
                trainer.train()
                metrics = trainer.evaluate()

            fold_scores.append(metrics["eval_f1"])

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction='maximize',
        study_name='hf_bert_tuning',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[HF-BERT] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[HF-BERT] Migliori parametri: {study.best_params}")

    return study.best_params


def train_hf_bert(X_train, X_val, X_test, y_train, y_val, y_test, results, best_params, meta_test=None):
    n_features = X_train.shape[1]

    train_ids, val_ids, test_ids, vocab_size, discretizer = _tokenize_features(
        X_train.values, X_val.values, X_test.values, n_bins=best_params['n_bins']
    )

    config = _build_bert_config(vocab_size, n_features, best_params)
    model = BertForSequenceClassification(config)

    train_ds = FlowTokenDataset(train_ids, y_train.values)
    val_ds = FlowTokenDataset(val_ids, y_val.values)
    test_ds = FlowTokenDataset(test_ids, y_test.values)
    class_weights = _class_weights(y_train)

    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, "hf_bert_checkpoints"),
        num_train_epochs=BERT_FINAL_EPOCHS,
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        report_to=[],
        seed=RANDOM_SEED,
        dataloader_pin_memory=False
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,             # <-- early stopping SOLO su validation
        compute_metrics=_compute_hf_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=BERT_FINAL_PATIENCE)],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    pred_output = trainer.predict(test_ds)   # <-- test toccato una sola volta, qui
    y_pred = np.argmax(pred_output.predictions, axis=-1)
    evaluate("HF-BERT (tokenizzato)", y_test, y_pred, results, elapsed, meta=meta_test)

    out_dir = os.path.join(MODELS_DIR, "hf_bert_model")
    model.save_pretrained(out_dir)
    joblib.dump(discretizer, os.path.join(out_dir, "discretizer.joblib"))
    print(f"Model path: {out_dir}")

    return model

# --------------------------------------------------------------------------
# 4. BERT sequenziale su finestre di W round
#    Ogni round della finestra e' un token: le F feature continue del round
#    sono proiettate linearmente nello spazio hidden di BERT (inputs_embeds),
#    si antepone un token [CLS] appreso, e la classificazione avviene sul
#    [CLS] (pooler + head di BertForSequenceClassification).
#    Le finestre sono causali con padding a sinistra: il padding e' escluso
#    tramite attention_mask, e il round corrente e' sempre l'ultimo token.
# --------------------------------------------------------------------------
SEQ_N_TRIALS = 5 #20
SEQ_CV_EPOCHS = 30
SEQ_CV_PATIENCE = 5
SEQ_FINAL_EPOCHS = 100
SEQ_FINAL_PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_windows(X, scen, w):
    """X: (n, F) ordinato per (scenario, round). Ritorna seq (n, w, F) e
    valid (n, w). Finestra causale con padding a sinistra, mai a cavallo
    di due scenari."""
    X = np.asarray(X, dtype=np.float32)
    scen = np.asarray(scen)
    n, f = X.shape
    seq = np.zeros((n, w, f), dtype=np.float32)
    valid = np.zeros((n, w), dtype=bool)

    starts = np.r_[0, np.flatnonzero(scen[1:] != scen[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    assert len(set(scen[starts])) == len(starts), "scenario_id non contiguo: ordina il dataframe"

    for s, e in zip(starts, ends):
        for t in range(s, e):
            lo = max(s, t - w + 1)
            k = t - lo + 1
            seq[t, w - k:] = X[lo:t + 1]
            valid[t, w - k:] = True
    return seq, valid


def scale_windows(scaler, seq, valid):
    n, w, f = seq.shape
    out = scaler.transform(seq.reshape(-1, f)).reshape(n, w, f)
    out = np.clip(out, -10, 10).astype(np.float32)
    out[~valid] = 0.0                      # il padding resta 0 dopo la scalatura
    return out


def _build_seq_bert_config(window, params):
    """BertConfig per la sequenza di round. vocab_size=1 perche' non si usano
    input_ids (si passa inputs_embeds): la word-embedding e' presente ma mai
    interrogata. max_position_embeddings = window + 1 ([CLS] + W round)."""
    hidden_size = params['num_attention_heads'] * params['head_dim']
    return BertConfig(
        vocab_size=1,
        hidden_size=hidden_size,
        num_hidden_layers=params['num_hidden_layers'],
        num_attention_heads=params['num_attention_heads'],
        intermediate_size=params['intermediate_size'],
        hidden_dropout_prob=params['dropout'],
        attention_probs_dropout_prob=params['dropout'],
        max_position_embeddings=window + 1,
        type_vocab_size=1,
        num_labels=2,
    )


class RoundBert(nn.Module):
    def __init__(self, n_feat, window, params):
        super().__init__()
        config = _build_seq_bert_config(window, params)
        self.proj = nn.Linear(n_feat, config.hidden_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        nn.init.normal_(self.cls, std=config.initializer_range)
        self.bert = BertForSequenceClassification(config)

    def forward(self, x, valid):
        # x: (B, W, F) scalato; valid: (B, W) bool (True = round reale)
        h = self.proj(x)                                        # (B, W, d)
        cls = self.cls.expand(h.size(0), -1, -1)                # (B, 1, d)
        h = torch.cat([cls, h], dim=1)                          # (B, W+1, d)
        mask = torch.cat(
            [torch.ones_like(valid[:, :1]), valid], dim=1
        ).long()                                                # (B, W+1), 0 = padding
        return self.bert(inputs_embeds=h, attention_mask=mask).logits


def _seq_loader(seq, valid, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(seq), torch.from_numpy(valid),
                       torch.as_tensor(np.asarray(y), dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_seq(model, seq, valid, batch_size=512):
    model.eval()
    preds = []
    for xb, mb, _ in _seq_loader(seq, valid, np.zeros(len(seq)), batch_size, False):
        preds.append(model(xb.to(DEVICE), mb.to(DEVICE)).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def fit_seq(params, seq_tr, m_tr, y_tr, seq_va, m_va, y_va, epochs, patience):
    torch.manual_seed(RANDOM_SEED)
    model = RoundBert(seq_tr.shape[2], params['window'], params).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'],
                            weight_decay=params['weight_decay'])
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y_tr).to(DEVICE))
    loader = _seq_loader(seq_tr, m_tr, y_tr, params['batch_size'], shuffle=True)

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, mb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(DEVICE), mb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        f1 = f1_score(y_va, predict_seq(model, seq_va, m_va), zero_division=0)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_f1


def _prepare_seq(X_fit, g_fit, others, w):
    """Scaler fit SOLO sul train; finestre costruite per split.
    others: lista di (X, groups). Ritorna [(seq, valid)] per fit + others."""
    scaler = StandardScaler().fit(np.asarray(X_fit, dtype=np.float32))
    out = []
    for X, g in [(X_fit, g_fit)] + list(others):
        seq, valid = build_windows(X, g, w)
        out.append((scale_windows(scaler, seq, valid), valid))
    return scaler, out


def tune_seq(X_train, y_train, groups_train, profiles_train,
             n_trials=SEQ_N_TRIALS, n_splits=N_SPLITS):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train, n_splits)

    def objective(trial):
        params = {
            'window': trial.suggest_categorical('window', [3, 5, 8]),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [2, 4]),
            'head_dim': trial.suggest_categorical('head_dim', [8, 16, 32]),
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 3),
            'intermediate_size': trial.suggest_categorical('intermediate_size', [64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        }
        scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            X_tr, X_va = X_train.iloc[tr_idx].values, X_train.iloc[va_idx].values
            g_tr, g_va = groups_train.iloc[tr_idx].values, groups_train.iloc[va_idx].values
            y_tr, y_va = y_train.iloc[tr_idx].values, y_train.iloc[va_idx].values

            _, ((s_tr, m_tr), (s_va, m_va)) = _prepare_seq(
                X_tr, g_tr, [(X_va, g_va)], params['window'])
            _, f1 = fit_seq(params, s_tr, m_tr, y_tr, s_va, m_va, y_va,
                            SEQ_CV_EPOCHS, SEQ_CV_PATIENCE)
            scores.append(f1)

            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction='maximize', study_name='seq_bert_tuning',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n[SeqBERT] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[SeqBERT] Migliori parametri: {study.best_params}")
    return study.best_params


def train_seq(X_train, X_val, X_test, y_train, y_val, y_test,
              groups_train, groups_val, groups_test, results, best_params, meta_test=None):
    scaler, ((s_tr, m_tr), (s_va, m_va), (s_te, m_te)) = _prepare_seq(
        X_train.values, groups_train.values,
        [(X_val.values, groups_val.values), (X_test.values, groups_test.values)],
        best_params['window'],
    )

    t0 = time.time()
    model, _ = fit_seq(best_params, s_tr, m_tr, y_train.values, s_va, m_va, y_val.values,
                       SEQ_FINAL_EPOCHS, SEQ_FINAL_PATIENCE)   # early stopping SOLO su validation
    elapsed = time.time() - t0

    y_pred = predict_seq(model, s_te, m_te)                    # test toccato una sola volta
    evaluate(f"SeqBERT (W={best_params['window']})", y_test, y_pred, results, elapsed, meta=meta_test)

    out_dir = os.path.join(MODELS_DIR, "seq_bert")
    os.makedirs(out_dir, exist_ok=True)
    # state_dict del wrapper (proj + [CLS] + BERT); per ricaricarlo si ricostruisce
    # RoundBert(len(FEATURES), params['window'], params) e si fa load_state_dict.
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    joblib.dump({"scaler": scaler, "params": best_params, "features": FEATURES},
                os.path.join(out_dir, "meta.joblib"))
    print(f"Model path: {out_dir}")
    return model
