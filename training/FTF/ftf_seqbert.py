"""
SeqBERT tuning + training
"""
import time
import joblib
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
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
parent = os.path.dirname(current)
root = os.path.dirname(parent)
sys.path.append(root)

from ftf_utils import * 

# Torch Numpy
torch.manual_seed(RANDOM_SEED)

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tuning
SEQ_CV_EPOCHS = 30
SEQ_CV_PATIENCE = 5
# Training+Validation
SEQ_FINAL_EPOCHS = 100
SEQ_FINAL_PATIENCE = 10

def tune_seq(X_train, y_train, groups_train, profiles_train):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)

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
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"\n[SeqBERT] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[SeqBERT] Migliori parametri: {study.best_params}")
    return study.best_params

def train_seq(X_train, X_val, X_test, y_train, y_val, y_test,
              groups_train, groups_val, groups_test, results, best_params, model_dir:str):
    scaler, ((s_tr, m_tr), (s_va, m_va), (s_te, m_te)) = _prepare_seq(
        X_train.values, groups_train.values,
        [(X_val.values, groups_val.values), (X_test.values, groups_test.values)],
        best_params['window'],
    )

    t0 = time.time()
    model, _ = fit_seq(best_params, s_tr, m_tr, y_train.values, s_va, m_va, y_val.values,
                       SEQ_FINAL_EPOCHS, SEQ_FINAL_PATIENCE)
    elapsed = time.time() - t0

    y_pred = predict_seq(model, s_te, m_te) 
    evaluate(f"SeqBERT (W={best_params['window']})", y_test, y_pred, results, elapsed)

    out_dir = os.path.join(model_dir, "seq_bert")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    joblib.dump({"scaler": scaler, "params": best_params, "features": FEATURES},
                os.path.join(out_dir, "meta.joblib"))
    print(f"Model path: {out_dir}")
    return model

"""
Utilities
"""
class RoundBert(nn.Module):
    def __init__(self, n_feat, window, params):
        super().__init__()
        config = self._build_seq_bert_config(window, params)
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
    
    def _build_seq_bert_config(self, window, params):
        return BertConfig(
            vocab_size=1,
            hidden_size=params['num_attention_heads'] * params['head_dim'],
            num_hidden_layers=params['num_hidden_layers'],
            num_attention_heads=params['num_attention_heads'],
            intermediate_size=params['intermediate_size'],
            hidden_dropout_prob=params['dropout'],
            attention_probs_dropout_prob=params['dropout'],
            max_position_embeddings=window + 1,
            type_vocab_size=1,
            num_labels=2,
        )

def _class_weights(y):
    values = y.values if hasattr(y, "values") else np.asarray(y)
    counts = np.bincount(values, minlength=2)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)

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

def _prepare_seq(X_fit, g_fit, others, w):
    scaler = StandardScaler().fit(np.asarray(X_fit, dtype=np.float32))
    out = []
    for X, g in [(X_fit, g_fit)] + list(others):
        seq, valid = build_windows(X, g, w)
        out.append((scale_windows(scaler, seq, valid), valid))
    return scaler, out

def build_windows(X, scen, w):
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
    out[~valid] = 0.0
    return out
