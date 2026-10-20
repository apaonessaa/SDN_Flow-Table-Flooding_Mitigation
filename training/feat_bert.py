"""
FeatBERT tuning + training

Un token per feature: T_j = b_j + x_j * W_j (Linear(1, d) distinta per ogni feature,
Gorishniy et al., 2021, §3.3), più un [CLS] in prima posizione. Encoder BERT (HuggingFace).
Preprocessing, inizializzazione e ciclo di training sono condivisi con TempBERT (nn_training.py).
"""
import time
import joblib
import optuna
import torch
import torch.nn as nn
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
BERT_CV_EPOCHS = 30
BERT_CV_PATIENCE = 5
# Training+Validation
BERT_FINAL_EPOCHS = 50
BERT_FINAL_PATIENCE = 10

def _build_bert_config(n_features, params):
    return BertConfig(
        vocab_size=1,                                   # not used: continuous feature 
        hidden_size=params['num_attention_heads'] * params['head_dim'],
        num_hidden_layers=params['num_hidden_layers'],
        num_attention_heads=params['num_attention_heads'],
        intermediate_size=params['intermediate_size'],
        hidden_dropout_prob=params['dropout'],
        attention_probs_dropout_prob=params['dropout'],
        max_position_embeddings=n_features + 1,         # not used: BertEmbeddings saved
        type_vocab_size=1,
        num_labels=2,
    )

class FeatureTokenizer(nn.Module):
    """T_j = b_j + x_j * W_j: Linear(1, d) per feature, [CLS] in pos 0."""
    def __init__(self, n_features, d):
        super().__init__()
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        self.proj = nn.ModuleList([nn.Linear(1, d) for _ in range(n_features)])
        init_uniform_(self.cls, *[t for lin in self.proj for t in (lin.weight, lin.bias)], fan_in=d)

    def forward(self, x):                                                   # x: (B, F)
        h = torch.stack([lin(x[:, j:j + 1]) for j, lin in enumerate(self.proj)], dim=1)   # (B, F, d)
        return torch.cat([self.cls.expand(len(x), -1, -1), h], dim=1)       # (B, F+1, d)

class FeatBert(nn.Module):
    def __init__(self, n_feat, params):
        super().__init__()
        config = _build_bert_config(n_feat, params)
        self.tokenizer = FeatureTokenizer(n_feat, config.hidden_size)
        self.bert = BertForSequenceClassification(config)
        self.bert.bert.embeddings = nn.Identity()       # no BertEmbeddings (LayerNorm, positions)

    def forward(self, x):                                                   # x: (B, F) scalato
        h = self.bert.bert.encoder(self.tokenizer(x)).last_hidden_state     # (B, F+1, d)
        return self.bert.classifier(self.bert.dropout(self.bert.bert.pooler(h)))   # pooler sul [CLS]

def fit_feat(params, x_tr, y_tr, x_va, y_va, n_features, epochs, patience):
    return fit_model(lambda: FeatBert(n_features, params),
                     (x_tr,), y_tr, (x_va,), y_va, params, epochs, patience)

def predict_feat(model, x):
    return predict_model(model, (x,))

def tune_featBERT(X_train, y_train, groups_train, profiles_train):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)
    n_features = X_train.shape[1]

    def objective(trial):
        params = {
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [2, 4, 8]),
            'head_dim': trial.suggest_categorical('head_dim', [8, 16, 24]),
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 2, 6),
            'intermediate_size': trial.suggest_categorical('intermediate_size', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        }

        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X_train.iloc[tr_idx].values, X_train.iloc[val_idx].values
            y_tr, y_val = y_train.iloc[tr_idx].values, y_train.iloc[val_idx].values

            x_tr, x_val, _ = scale_features(X_tr, X_val)
            _, f1 = fit_feat(params, x_tr, y_tr, x_val, y_val,
                             n_features, BERT_CV_EPOCHS, BERT_CV_PATIENCE)
            fold_scores.append(f1)

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction='maximize',
        study_name='feat_bert_tuning',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n[FeatBERT] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[FeatBERT] Migliori parametri: {study.best_params}")

    return study.best_params


def train_featBERT(X_train, X_val, X_test, y_train, y_val, y_test, results, best_params, model_dir: str):
    n_features = X_train.shape[1]

    x_tr, x_va, x_te, scaler = scale_features(X_train.values, X_val.values, X_test.values)

    t0 = time.time()
    model, _ = fit_feat(best_params, x_tr, y_train.values, x_va, y_val.values,
                        n_features, BERT_FINAL_EPOCHS, BERT_FINAL_PATIENCE)
    elapsed = time.time() - t0

    y_pred = predict_feat(model, x_te)
    evaluate("FeatBERT", y_test, y_pred, results, elapsed)

    out_dir = os.path.join(model_dir, "feat_bert")
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    joblib.dump({"scaler": scaler, "params": best_params, "n_features": n_features},
                os.path.join(out_dir, "meta.joblib"))
    print(f"Model path: {out_dir}")

    return model
