"""
TempBERT tuning + training (dataset MFR)

One token per round: Linear(F, d) on the feature vector of the round (Zerveas et al., 2021, §3.1), 
plus a [CLS] in first position. Random window of W round inside the scenario, with padding mask. 
BERT (HuggingFace) encoder.

Difference from the FTF version: the windows do not follow the scenario but the flow key 
(flow_key = scenario_id + match_hash, 'keys_*'), while tuning folds are constructed on the scenarios 
('groups_train') and on the attack profile.
"""
import time
import joblib
import optuna
from sklearn.metrics import f1_score
from transformers import BertConfig, BertForSequenceClassification

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils import *

# Torch Numpy
import torch
import torch.nn as nn
torch.manual_seed(RANDOM_SEED)

optuna.logging.set_verbosity(optuna.logging.WARNING)

class TempBERT(nn.Module):
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
            hidden_size=params['num_attention_heads']*params['head_dim'],
            num_hidden_layers=params['num_hidden_layers'],
            num_attention_heads=params['num_attention_heads'],
            intermediate_size=params['intermediate_size'],
            hidden_dropout_prob=params['dropout'],
            attention_probs_dropout_prob=params['dropout'],
            max_position_embeddings=window+1,
            type_vocab_size=1,
            num_labels=2,
        )

def tune_tempBERT(X_train, y_train, groups_train, profiles_train, keys_train):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)

    def objective(trial):
        params = {
            'window': trial.suggest_categorical('window', WINDOWS),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', NUM_ATTENTION_HEADS),
            'head_dim': trial.suggest_categorical('head_dim', HEAD_DIM),
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', NUM_HIDDEN_LAYERS[0], NUM_HIDDEN_LAYERS[1]),
            'intermediate_size': trial.suggest_categorical('intermediate_size', INTERMEDIATE_SIZE),
            'dropout': trial.suggest_float('dropout', DROPOUT[0], DROPOUT[1]),
            'learning_rate': trial.suggest_float('learning_rate', LEARNING_RATE[0], LEARNING_RATE[1], log=True),
            'weight_decay': trial.suggest_float('weight_decay', WEIGHT_DECAY[0], WEIGHT_DECAY[1], log=True),
            'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZE),
        }
        scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            X_tr, X_va = X_train.iloc[tr_idx].values, X_train.iloc[va_idx].values
            k_tr, k_va = keys_train.iloc[tr_idx].values, keys_train.iloc[va_idx].values
            y_tr, y_va = y_train.iloc[tr_idx].values, y_train.iloc[va_idx].values

            _, ((s_tr, m_tr), (s_va, m_va)) = _prepare_seq(
                X_tr, k_tr, [(X_va, k_va)], params['window'])
            _, f1 = fit_seq(params, s_tr, m_tr, y_tr, s_va, m_va, y_va,
                            BERT_CV_EPOCHS, BERT_CV_PATIENCE)
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
                       BERT_FINAL_EPOCHS, BERT_FINAL_PATIENCE)
    elapsed = time.time() - t0

    y_pred = predict_seq(model, s_te, m_te)
    evaluate(f"TempBERT (W={best_params['window']})", y_test, y_pred, results, elapsed)

    ## F1-score permutation importance
    importances = permutation_importance_seq(
        lambda seq, valid: predict_seq(model, seq, valid), s_te, m_te, y_test.values, features
    )
    print(importances.sort_values(ascending=False))
    #importances_pct = to_percent(importances)
    #print(importances_pct.sort_values(ascending=False))

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
def fit_seq(params, seq_tr, m_tr, y_tr, seq_va, m_va, y_va, epochs, patience):
    return fit_model(lambda: TempBERT(seq_tr.shape[2], params['window'], params),
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
