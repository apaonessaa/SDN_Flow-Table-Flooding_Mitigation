"""
Ablation study FeatBERT: 
    M0 (MLP su tabulare grezzo) vs M1 (Feature Tokenizer + mean pooling, SENZA self-attention) 
        vs M2 (FeatBERT completo).

M0: No Tokenizer
M1: Feature Tokenizer Linear(1,d) + mean pooling
M2: Feature Tokenizer Linear(1,d) + self-attention + CLS pooling
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
from feat_bert import FeatureTokenizer, fit_feat, predict_feat

torch.manual_seed(RANDOM_SEED)

class MLPBaseline(nn.Module):
    """
    M0: No Feature Tokenizer. 
    Hidden dim = FeatBERT.hidden_size (`d`)
    """
    def __init__(self, n_feat, params):
        super().__init__()
        d = params['num_attention_heads'] * params['head_dim']
        self.fc1 = nn.Linear(n_feat, d)
        self.fc2 = nn.Linear(d, d)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(d, 2)

        init_uniform_(self.fc1.weight, self.fc1.bias, fan_in=n_feat)
        init_uniform_(self.fc2.weight, self.fc2.bias, fan_in=d)
        init_uniform_(self.classifier.weight, self.classifier.bias, fan_in=d)

    def forward(self, x):                                   # x: (B, F)
        h = self.dropout(self.act(self.fc1(x)))
        h = self.dropout(self.act(self.fc2(h)))
        return self.classifier(h)

class FeatTokenMeanPool(nn.Module):
    """
    M1: Feature Tokenizer + Mean Pooling (No Encoder with self-attention)
    Pooling without [CLS], it is useless without self-attention.
    """
    def __init__(self, n_feat, params):
        super().__init__()
        d = params['num_attention_heads'] * params['head_dim']
        self.tokenizer = FeatureTokenizer(n_feat, d)
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(d, 2)
        init_uniform_(self.classifier.weight, self.classifier.bias, fan_in=d)

    def forward(self, x):                                   # x: (B, F)
        h = self.tokenizer(x)                                # (B, F+1, d), [:,0,:] = CLS
        pooled = h[:, 1:, :].mean(dim=1)                      # mean pooling sui soli token di feature
        return self.classifier(self.dropout(pooled))

def fit_mlp(params, x_tr, y_tr, x_va, y_va, n_features, epochs, patience):
    return fit_model(lambda: MLPBaseline(n_features, params),
                      (x_tr,), y_tr, (x_va,), y_va, params, epochs, patience)

def fit_tokenizer_meanpool(params, x_tr, y_tr, x_va, y_va, n_features, epochs, patience):
    return fit_model(lambda: FeatTokenMeanPool(n_features, params),
                      (x_tr,), y_tr, (x_va,), y_va, params, epochs, patience)

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

def run_featbert_ablation(X_train, X_val, X_test, y_train, y_val, y_test,
                           best_featBERT_params, features, out_dir: str,
                           epochs=None, patience=None):
    """
    M0, M1, M2 train/val/test with the best_featBERT_params. 
    """
    epochs = epochs or BERT_FINAL_EPOCHS
    patience = patience or BERT_FINAL_PATIENCE

    n_features = X_train.shape[1]
    x_tr, x_va, x_te, _ = scale_features(X_train.values, X_val.values, X_test.values)

    rows = []

    # --- M0: MLP su tabulare grezzo (nessuna tokenizzazione per-feature) ---
    t0 = time.time()
    model_m0, _ = fit_mlp(best_featBERT_params, x_tr, y_train.values, x_va, y_val.values,
                           n_features, epochs, patience)
    elapsed_m0 = time.time() - t0
    y_pred_m0 = predict_model(model_m0, (x_te,))
    rows.append(_ablation_row("M0_MLP", y_test, y_pred_m0, elapsed_m0, model_m0))

    # --- M1: Feature tokenizer + mean pooling (tokenizer senza attention) ---
    t0 = time.time()
    model_m1, _ = fit_tokenizer_meanpool(best_featBERT_params, x_tr, y_train.values, x_va, y_val.values,
                                          n_features, epochs, patience)
    elapsed_m1 = time.time() - t0
    y_pred_m1 = predict_model(model_m1, (x_te,))
    rows.append(_ablation_row("M1_Tokenizer_MeanPool", y_test, y_pred_m1, elapsed_m1, model_m1))

    # --- M2: FeatBERT completo (tokenizer + self-attention) ---
    t0 = time.time()
    model_m2, _ = fit_feat(best_featBERT_params, x_tr, y_train.values, x_va, y_val.values,
                            n_features, epochs, patience)
    elapsed_m2 = time.time() - t0
    y_pred_m2 = predict_feat(model_m2, x_te)
    rows.append(_ablation_row("M2_FeatBERT", y_test, y_pred_m2, elapsed_m2, model_m2))

    ablation_df = pd.DataFrame(rows).set_index("variant")
    print("\n[FeatBERT ablation] M0 vs M1 vs M2:")
    print(ablation_df.to_string(float_format="%.4f"))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "featbert_ablation.csv")
    ablation_df.to_csv(out_path)
    print(f"Risultati ablation salvati in: {out_path}")

    return ablation_df