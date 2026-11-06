"""
Ablation study TempBERT (MFR): Identical to temp_bert_ablation.py (FTF) but 
the only difference is that here the windowing takes place by flow_key 
(identity of the single rule, keys_*) instead of scenario_id.
"""
import time
import torch
import torch.nn as nn
import pandas as pd

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils import *
from mfr_temp_bert import fit_seq, predict_seq, _prepare_seq

torch.manual_seed(RANDOM_SEED)

class MLPBaseline(nn.Module):
    """
    M0: No Feature Tokenizer and there is not masking the padding.
    """
    def __init__(self, n_feat, window, params):
        super().__init__()
        d = params['num_attention_heads'] * params['head_dim']
        in_dim = window * n_feat
        self.fc1 = nn.Linear(in_dim, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(d, 2)

        init_uniform_(self.fc1.weight, self.fc1.bias, fan_in=in_dim)
        init_uniform_(self.fc2.weight, self.fc2.bias, fan_in=d)
        init_uniform_(self.classifier.weight, self.classifier.bias, fan_in=d)

    def forward(self, x, valid):            # x: (B, W, F); valid non usato, vedi sopra
        B = x.size(0)
        h = self.dropout(self.act(self.fc1(x.reshape(B, -1))))
        h = self.dropout(self.act(self.fc2(h)))
        return self.classifier(h)

class RoundTokenMeanPool(nn.Module):
    """
    M1: Round Tokenizer + shared Linear(F,d) + mean pooling is calculated with masking the padding (valid)
    """
    def __init__(self, n_feat, params):
        super().__init__()
        d = params['num_attention_heads'] * params['head_dim']
        self.proj = nn.Linear(n_feat, d)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(d, 2)
        init_uniform_(self.proj.weight, self.proj.bias, fan_in=n_feat)
        init_uniform_(self.classifier.weight, self.classifier.bias, fan_in=d)

    def forward(self, x, valid):             # x: (B, W, F); valid: (B, W) bool
        h = self.proj(x)                                      # (B, W, d)
        mask = valid.unsqueeze(-1).float()                     # (B, W, 1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.dropout(pooled))

def fit_mlp(params, window, s_tr, m_tr, y_tr, s_va, m_va, y_va, n_features, epochs, patience):
    return fit_model(lambda: MLPBaseline(n_features, window, params),
                      (s_tr, m_tr), y_tr, (s_va, m_va), y_va, params, epochs, patience)

def fit_round_meanpool(params, s_tr, m_tr, y_tr, s_va, m_va, y_va, n_features, epochs, patience):
    return fit_model(lambda: RoundTokenMeanPool(n_features, params),
                      (s_tr, m_tr), y_tr, (s_va, m_va), y_va, params, epochs, patience)

def _ablation_row(name, y_true, y_pred, elapsed, model):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} ===")
    print(f"Tempo di training: {elapsed:.1f}s | n_params={n_params}")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(classification_report(y_true, y_pred))
    return {
        "variant": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "train_time_s": elapsed,
        "n_params": n_params,
    }

def run_tempbert_ablation(X_train, X_val, X_test, y_train, y_val, y_test,
                           keys_train, keys_val, keys_test,
                           best_tempBERT_params, features, out_dir: str,
                           epochs=None, patience=None):
    """
    Shared _prepare_seq for keys_train/val/test: flow_key (scenario_id + match_hash).
    """
    epochs = epochs or BERT_FINAL_EPOCHS
    patience = patience or BERT_FINAL_PATIENCE
    window = best_tempBERT_params['window']
    n_features = X_train.shape[1]

    # Finestre costruite una sola volta, riusate da M0, M1, M2
    _, ((s_tr, m_tr), (s_va, m_va), (s_te, m_te)) = _prepare_seq(
        X_train.values, keys_train.values,
        [(X_val.values, keys_val.values), (X_test.values, keys_test.values)],
        window,
    )

    rows = []

    # --- M0: MLP su finestra appiattita ---
    t0 = time.time()
    model_m0, _ = fit_mlp(best_tempBERT_params, window, s_tr, m_tr, y_train.values,
                           s_va, m_va, y_val.values, n_features, epochs, patience)
    elapsed_m0 = time.time() - t0
    y_pred_m0 = predict_model(model_m0, (s_te, m_te))
    rows.append(_ablation_row("M0_MLP", y_test, y_pred_m0, elapsed_m0, model_m0))

    # --- M1: Round tokenizer + mean pooling mascherato (no attention) ---
    t0 = time.time()
    model_m1, _ = fit_round_meanpool(best_tempBERT_params, s_tr, m_tr, y_train.values,
                                      s_va, m_va, y_val.values, n_features, epochs, patience)
    elapsed_m1 = time.time() - t0
    y_pred_m1 = predict_model(model_m1, (s_te, m_te))
    rows.append(_ablation_row("M1_RoundToken_MeanPool", y_test, y_pred_m1, elapsed_m1, model_m1))

    # --- M2: TempBERT completo (round tokenizer + self-attention) ---
    t0 = time.time()
    model_m2, _ = fit_seq(best_tempBERT_params, s_tr, m_tr, y_train.values, s_va, m_va, y_val.values,
                           epochs, patience)
    elapsed_m2 = time.time() - t0
    y_pred_m2 = predict_seq(model_m2, s_te, m_te)
    rows.append(_ablation_row("M2_TempBERT", y_test, y_pred_m2, elapsed_m2, model_m2))

    ablation_df = pd.DataFrame(rows).set_index("variant")
    print(f"\n[TempBERT ablation] M0 vs M1 vs M2 (window={window}):")
    print(ablation_df.to_string(float_format="%.4f"))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tempbert_ablation.csv")
    ablation_df.to_csv(out_path)
    print(f"Risultati ablation salvati in: {out_path}")

    return ablation_df