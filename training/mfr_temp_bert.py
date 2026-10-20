"""
TempBERT tuning + training (dataset MFR)

Stessa architettura e stessi componenti condivisi della versione FTF (temp_bert.py):
un token per round (Linear(F, d), Zerveas et al., 2021, §3.1) + [CLS], finestra causale di W round
con padding mask, encoder BERT (HuggingFace). Preprocessing, inizializzazione e ciclo di training
vengono da nn_training.py.

Differenza rispetto alla versione FTF: le finestre non seguono lo scenario ma la chiave del flusso
(flow_key = scenario_id + match_hash, `keys_*`), mentre i fold del tuning sono costruiti
sugli scenari (`groups_train`) e sul profilo d'attacco.
"""
import time
import joblib
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import BertConfig, BertForSequenceClassification

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils import *
from nn_training import scale_features, init_uniform_, fit_model, predict_model

# Torch Numpy
torch.manual_seed(RANDOM_SEED)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Tuning
SEQ_CV_EPOCHS = 30
SEQ_CV_PATIENCE = 5
# Training+Validation
SEQ_FINAL_EPOCHS = 50
SEQ_FINAL_PATIENCE = 10

WINDOWS = [3, 5, 8]

def tune_tempBERT(X_train, y_train, groups_train, profiles_train, keys_train):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)

    def objective(trial):
        params = {
            'window': trial.suggest_categorical('window', WINDOWS),
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
            k_tr, k_va = keys_train.iloc[tr_idx].values, keys_train.iloc[va_idx].values
            y_tr, y_va = y_train.iloc[tr_idx].values, y_train.iloc[va_idx].values

            _, ((s_tr, m_tr), (s_va, m_va)) = _prepare_seq(
                X_tr, k_tr, [(X_va, k_va)], params['window'])
            _, f1 = fit_seq(params, s_tr, m_tr, y_tr, s_va, m_va, y_va,
                            SEQ_CV_EPOCHS, SEQ_CV_PATIENCE)
            scores.append(f1)

            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction='maximize', study_name='temp_bert_tuning',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"\n[TempBERT] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[TempBERT] Migliori parametri: {study.best_params}")
    return study.best_params

def train_tempBERT(X_train, X_val, X_test, y_train, y_val, y_test,
              keys_train, keys_val, keys_test, results, best_params, features, model_dir:str):
    scaler, ((s_tr, m_tr), (s_va, m_va), (s_te, m_te)) = _prepare_seq(
        X_train.values, keys_train.values,
        [(X_val.values, keys_val.values), (X_test.values, keys_test.values)],
        best_params['window'],
    )

    t0 = time.time()
    model, _ = fit_seq(best_params, s_tr, m_tr, y_train.values, s_va, m_va, y_val.values,
                       SEQ_FINAL_EPOCHS, SEQ_FINAL_PATIENCE)
    elapsed = time.time() - t0

    y_pred = predict_seq(model, s_te, m_te)
    evaluate(f"TempBERT (W={best_params['window']})", y_test, y_pred, results, elapsed)

    importances = permutation_importance_seq(model, s_te, m_te, y_test.values, features)
    importances_pct = to_percent(importances)
    print(importances_pct.sort_values(ascending=False))

    out_dir = os.path.join(model_dir, "temp_bert")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    joblib.dump({"scaler": scaler, "params": best_params, "features": features},
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
        d = config.hidden_size
        self.proj = nn.Linear(n_feat, d)
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        init_uniform_(self.proj.weight, self.proj.bias, fan_in=n_feat)   # = default di nn.Linear
        init_uniform_(self.cls, fan_in=d)                                # stessa regola di FeatBERT
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
            max_position_embeddings=window+1,
            type_vocab_size=1,
            num_labels=2,
        )

def fit_seq(params, seq_tr, m_tr, y_tr, seq_va, m_va, y_va, epochs, patience):
    return fit_model(lambda: RoundBert(seq_tr.shape[2], params['window'], params),
                     (seq_tr, m_tr), y_tr, (seq_va, m_va), y_va, params, epochs, patience)

def predict_seq(model, seq, valid):
    return predict_model(model, (seq, valid))

def _prepare_seq(X_fit, g_fit, others, w):
    # Il preprocessing agisce sulle righe (round) prima del windowing: è equivalente a scalare
    # le finestre dopo, perché ogni trasformazione è per-feature e non dipende dalle altre righe.
    # Qui g_fit / g sono le chiavi di flusso (keys_*), non gli scenari.
    *xs, scaler = scale_features(X_fit, *[X for X, _ in others])
    groups = [g_fit] + [g for _, g in others]
    out = [build_windows(x, g, w) for x, g in zip(xs, groups)]
    return scaler, out

def build_windows(X, scen, w):
    X = np.asarray(X, dtype=np.float32)
    scen = np.asarray(scen)
    n, f = X.shape
    seq = np.zeros((n, w, f), dtype=np.float32)
    valid = np.zeros((n, w), dtype=bool)

    starts = np.r_[0, np.flatnonzero(scen[1:] != scen[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    assert len(set(scen[starts])) == len(starts), "chiave non contigua: ordina il dataframe per scenario, flow_key, round"

    for s, e in zip(starts, ends):
        for t in range(s, e):
            lo = max(s, t - w + 1)
            k = t - lo + 1
            seq[t, w - k:] = X[lo:t + 1]
            valid[t, w - k:] = True
    return seq, valid

def permutation_importance_seq(model, seq, valid, y, features, n_repeats=5):
    rng = np.random.default_rng(RANDOM_SEED)
    y = np.asarray(y)
    base = f1_score(y, predict_seq(model, seq, valid), zero_division=0)
    seq_p = seq.copy()
    drops = np.zeros((len(features), n_repeats))
    for j in range(len(features)):
        orig = seq[:, :, j]
        for r in range(n_repeats):
            perm = rng.permutation(len(seq))
            col = orig[perm].copy()
            col[~valid] = 0.0
            seq_p[:, :, j] = col
            f1 = f1_score(y, predict_seq(model, seq_p, valid), zero_division=0)
            drops[j, r] = base - f1
        seq_p[:, :, j] = orig
    return pd.Series(drops.mean(axis=1), index=list(features))

def to_percent(importances: pd.Series) -> pd.Series:
    clipped = importances.clip(lower=0)
    total = clipped.sum()
    if total == 0:
        return clipped
    return clipped / total * 100
