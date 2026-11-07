"""
Ablation semplificata di FeatBERT e TempBERT (FTF e MFR).

Tre varianti, stesso training e stessi iperparametri (i migliori del modello completo):
    Full         modello completo
    NoTokenizer  senza tokenizzazione: l'intero input diventa UN solo token
    NoAttention  senza self-attention: i token non si scambiano informazione

Una sola esecuzione per variante, nessun tuning dei baseline.
"""
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Modules loader
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(current)
sys.path.append(root)

from utils import (RANDOM_SEED, BERT_FINAL_EPOCHS, BERT_FINAL_PATIENCE,
                   scale_features, fit_model, predict_model, init_uniform_)
from feat_bert import FeatBert


# ---------------------------------------------------------------------------
# 1. Togliere la self-attention (comune a FeatBERT e TempBERT)
# ---------------------------------------------------------------------------
class NoMix(nn.Module):
    """Al posto di BertSelfAttention: ogni token passa solo per la proiezione V, nessun mixing."""
    def __init__(self, old_attention):
        super().__init__()
        self.value = old_attention.value                  # riuso il Linear V gia' inizializzato

    def forward(self, hidden_states, *args, **kwargs):
        return (self.value(hidden_states), None)          # stessa forma di output dell'attention HF


@torch.no_grad()
def drop_attention(encoder):
    for layer in encoder.layer:
        layer.attention.self = NoMix(layer.attention.self)

    # Controllo: cambio un token, gli altri devono restare identici.
    was_training = encoder.training
    encoder.eval()
    d = encoder.layer[0].attention.self.value.in_features
    h = torch.randn(2, 5, d)
    h2 = h.clone()
    h2[:, 0] += 1.0
    a, b = encoder(h).last_hidden_state, encoder(h2).last_hidden_state
    assert torch.allclose(a[:, 1:], b[:, 1:], atol=1e-6), "l'attention e' ancora attiva"
    encoder.train(was_training)


# ---------------------------------------------------------------------------
# 2. FeatBERT
# ---------------------------------------------------------------------------
class SingleToken(nn.Module):
    """Al posto di FeatureTokenizer: tutto il vettore di feature -> un solo token."""
    def __init__(self, n_feat, d):
        super().__init__()
        self.proj = nn.Linear(n_feat, d)
        init_uniform_(self.proj.weight, self.proj.bias, fan_in=n_feat)

    def forward(self, x):                                  # (B, F) -> (B, 1, d)
        return self.proj(x).unsqueeze(1)


class FeatBertNoTok(FeatBert):
    def __init__(self, n_feat, params):
        super().__init__(n_feat, params)
        self.tokenizer = SingleToken(n_feat, self.bert.config.hidden_size)
        # forward ereditato: encoder -> pooler sulla posizione 0 (qui l'unico token)


class FeatBertNoAttn(FeatBert):
    def __init__(self, n_feat, params):
        super().__init__(n_feat, params)
        drop_attention(self.bert.bert.encoder)

    def forward(self, x):                                  # x: (B, F)
        h = self.bert.bert.encoder(self.tokenizer(x)).last_hidden_state    # (B, F+1, d)
        h = h[:, 1:].mean(dim=1, keepdim=True)             # media dei token-feature (il [CLS] non riceve nulla)
        return self.bert.classifier(self.bert.dropout(self.bert.bert.pooler(h)))


# ---------------------------------------------------------------------------
# 3. TempBERT (le classi sono costruite a partire da TempBert di temp_bert o mfr_temp_bert)
# ---------------------------------------------------------------------------
def make_temp_variants(TempBert):
    class TempBertNoTok(TempBert):
        def __init__(self, n_feat, window, params):
            super().__init__(n_feat, window, params)
            d = self.bert.config.hidden_size
            in_dim = window * n_feat + window              # finestra appiattita + maschera valid
            del self.cls                                   # senza attention tra token il [CLS] non serve
            self.proj = nn.Linear(in_dim, d)
            init_uniform_(self.proj.weight, self.proj.bias, fan_in=in_dim)

        def forward(self, x, valid):                       # x: (B, W, F); valid: (B, W)
            z = torch.cat([x.flatten(1), valid.to(x.dtype)], dim=1)
            h = self.proj(z).unsqueeze(1)                  # (B, 1, d): un solo token
            mask = torch.ones(h.size(0), 1, dtype=torch.long, device=h.device)
            return self.bert(inputs_embeds=h, attention_mask=mask).logits

    class TempBertNoAttn(TempBert):
        def __init__(self, n_feat, window, params):
            super().__init__(n_feat, window, params)
            drop_attention(self.bert.bert.encoder)

        def forward(self, x, valid):
            h = self.proj(x)                               # un token per round, come in TempBert
            h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], dim=1)
            mask = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1).long()
            hs = self.bert.bert(inputs_embeds=h, attention_mask=mask).last_hidden_state
            w = valid.unsqueeze(-1).to(hs.dtype)           # media dei soli round reali (padding escluso)
            pooled = (hs[:, 1:] * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1.0)
            return self.bert.classifier(self.bert.dropout(self.bert.bert.pooler(pooled)))

    return TempBertNoTok, TempBertNoAttn


# ---------------------------------------------------------------------------
# 4. Esecuzione
# ---------------------------------------------------------------------------
def _row(name, y_true, y_pred, elapsed, model):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: acc={acc:.4f} f1={f1:.4f} ({elapsed:.1f}s, {n_params} params)")
    return {"variant": name, "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "train_time_s": elapsed, "n_params": n_params}


def _save(rows, out_dir, csv_name):
    df = pd.DataFrame(rows).set_index("variant")
    print("\n" + df.to_string(float_format="%.4f"))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, csv_name))
    return df


def run_featbert_ablation(X_train, X_val, X_test, y_train, y_val, y_test,
                          best_params, out_dir, csv_name="featbert_ablation.csv"):
    n_feat = X_train.shape[1]
    x_tr, x_va, x_te, _ = scale_features(X_train.values, X_val.values, X_test.values)

    variants = {"Full": FeatBert, "NoTokenizer": FeatBertNoTok, "NoAttention": FeatBertNoAttn}
    rows = []
    for name, cls in variants.items():
        torch.manual_seed(RANDOM_SEED)
        t0 = time.time()
        model, _ = fit_model(lambda c=cls: c(n_feat, best_params),
                             (x_tr,), y_train.values, (x_va,), y_val.values,
                             best_params, BERT_FINAL_EPOCHS, BERT_FINAL_PATIENCE)
        elapsed = time.time() - t0
        rows.append(_row(name, y_test, predict_model(model, (x_te,)), elapsed, model))
    return _save(rows, out_dir, csv_name)


def run_tempbert_ablation(TempBert, prepare_seq,
                          X_train, X_val, X_test, y_train, y_val, y_test,
                          groups_train, groups_val, groups_test,
                          best_params, out_dir, csv_name="tempbert_ablation.csv"):
    """
    FTF: TempBert e _prepare_seq da temp_bert, gruppi = scenario_id.
    MFR: TempBert e _prepare_seq da mfr_temp_bert, gruppi = flow_key.
    """
    window = best_params["window"]
    n_feat = X_train.shape[1]
    _, ((s_tr, m_tr), (s_va, m_va), (s_te, m_te)) = prepare_seq(
        X_train.values, groups_train.values,
        [(X_val.values, groups_val.values), (X_test.values, groups_test.values)],
        window,
    )

    NoTok, NoAttn = make_temp_variants(TempBert)
    variants = {"Full": TempBert, "NoTokenizer": NoTok, "NoAttention": NoAttn}
    rows = []
    for name, cls in variants.items():
        torch.manual_seed(RANDOM_SEED)
        t0 = time.time()
        model, _ = fit_model(lambda c=cls: c(n_feat, window, best_params),
                             (s_tr, m_tr), y_train.values, (s_va, m_va), y_val.values,
                             best_params, BERT_FINAL_EPOCHS, BERT_FINAL_PATIENCE)
        elapsed = time.time() - t0
        rows.append(_row(name, y_test, predict_model(model, (s_te, m_te)), elapsed, model))
    return _save(rows, out_dir, csv_name)
