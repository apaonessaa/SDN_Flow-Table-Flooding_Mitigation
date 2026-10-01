import tempfile
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
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
# Utility per la stratificazione per profilo di attacco a livello di gruppo
# (scenario). Il raggruppamento (niente scenario a cavallo di due split)
# resta quello originale; qui aggiungiamo solo il criterio di stratificazione
# per evitare che un profilo (es. stealth) finisca per caso concentrato in
# un solo split.
# --------------------------------------------------------------------------
def _group_profiles(df, group_col, profile_col):
    """Un profilo di attacco per ciascun gruppo. Uno scenario contiene tipicamente
    sia righe benigne sia (se e' uno scenario di attacco) righe con un profilo
    specifico: usare la moda su tutte le righe sarebbe fuorviante, perche' il
    traffico benigno e' di solito molto piu' numeroso e finirebbe per
    "mascherare" il vero profilo di attacco dello scenario. Si prende quindi
    il valore non-benigno presente nel gruppo (se ce n'e' piu' di uno, il piu'
    frequente); se il gruppo e' interamente benigno resta "benigno"."""
    def pick(s):
        non_benign = s[s != BENIGN_PROFILE_LABEL]
        if len(non_benign) > 0:
            return non_benign.mode().iat[0]
        return BENIGN_PROFILE_LABEL

    return df.groupby(group_col)[profile_col].agg(pick)


def stratified_group_split(df, group_col, profile_col, test_size, random_state):
    """Divide i GRUPPI (non le singole righe) in due insiemi, stratificando
    per il profilo di attacco associato a ciascun gruppo. Cosi' ogni split
    riceve scenari aggressive/moderate/stealth in proporzioni simili, invece
    di affidarsi a un semplice GroupShuffleSplit non stratificato."""
    group_profiles = _group_profiles(df, group_col, profile_col)
    groups_unique = group_profiles.index.to_numpy()
    profiles_unique = group_profiles.to_numpy()

    try:
        left, right = train_test_split(
            groups_unique,
            test_size=test_size,
            random_state=random_state,
            stratify=profiles_unique,
        )
    except ValueError as e:
        print(f"[!] Stratificazione per '{profile_col}' non applicabile ({e}); "
              f"eseguo uno split non stratificato.")
        left, right = train_test_split(
            groups_unique, test_size=test_size, random_state=random_state,
        )
    return left, right


def print_profile_distribution(name, df_subset, group_col, profile_col):
    group_profiles = _group_profiles(df_subset, group_col, profile_col)
    print(f"Distribuzione profili di attacco - {name} "
          f"({group_profiles.shape[0]} scenari):")
    print(group_profiles.value_counts())


# --------------------------------------------------------------------------
# Caricamento dati e split a TRE vie: train (tuning+fit) / val (early
# stopping) / test (valutazione finale, toccato una sola volta).
# Tutti e tre gli split sono per intero scenario_id, cosi' nessuno scenario
# finisce contemporaneamente in due split diversi, e sono stratificati per
# profilo di attacco (aggressive/moderate/stealth) cosi' che ogni split
# riceva una rappresentanza proporzionata di tutti e tre i profili.
# --------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(TRAINING_SET)
    print(f"Numero di entry del dataset: {len(df)}")

    counts = df.groupby([SCENARIO_COL, TARGET]).size().unstack(fill_value=0)
    print(counts)
    print(f"\nDeviazione standard tra scenari: {counts.sum(axis=1).std()}")
    
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[SCENARIO_COL]

    # 1) hold-out test: mai piu' toccato fino alla valutazione finale
    trainval_groups, test_groups = stratified_group_split(
        df, SCENARIO_COL, ATTACK_PROFILE_COL, test_size=0.2, random_state=RANDOM_SEED
    )
    trainval_mask = groups.isin(trainval_groups).to_numpy()
    test_mask = groups.isin(test_groups).to_numpy()

    df_trainval = df[trainval_mask].reset_index(drop=True)
    X_trainval = X[trainval_mask].reset_index(drop=True)
    y_trainval = y[trainval_mask].reset_index(drop=True)
    groups_trainval = groups[trainval_mask].reset_index(drop=True)

    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)

    # 2) dentro il train+val: separiamo una validation per l'early stopping
    #    del training finale, cosi' il test set non viene mai usato per
    #    scegliere il numero di iterazioni/epoche. Anche qui, stratificato
    #    per profilo di attacco.
    train_groups, val_groups = stratified_group_split(
        df_trainval, SCENARIO_COL, ATTACK_PROFILE_COL, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    train_mask = groups_trainval.isin(train_groups).to_numpy()
    val_mask = groups_trainval.isin(val_groups).to_numpy()

    X_train = X_trainval[train_mask].reset_index(drop=True)
    y_train = y_trainval[train_mask].reset_index(drop=True)
    groups_train = groups_trainval[train_mask].reset_index(drop=True)
    df_train = df_trainval[train_mask].reset_index(drop=True)

    X_val = X_trainval[val_mask].reset_index(drop=True)
    y_val = y_trainval[val_mask].reset_index(drop=True)

    print(f"\nScenari nel train (tuning+fit): {groups_train.nunique()} -> {sorted(groups_train.unique())}")
    print(f"Scenari in validation (early stopping): {groups_trainval[val_mask].nunique()} -> {sorted(groups_trainval[val_mask].unique())}")
    print(f"Scenari nel test finale: {groups[test_mask].nunique()} -> {sorted(groups[test_mask].unique())}")

    print_profile_distribution("train", df_train, SCENARIO_COL, ATTACK_PROFILE_COL)
    print_profile_distribution("validation", df_trainval[val_mask], SCENARIO_COL, ATTACK_PROFILE_COL)
    print_profile_distribution("test", df[test_mask], SCENARIO_COL, ATTACK_PROFILE_COL)

    # Profilo per riga, allineato a X_train/y_train: serve per stratificare
    # anche i fold di cross-validation nel tuning (vedi make_group_kfold_splits).
    profiles_train = df_train[ATTACK_PROFILE_COL].reset_index(drop=True)

    groups_val = groups_trainval[val_mask].reset_index(drop=True)
    groups_test = groups[test_mask].reset_index(drop=True)

    meta_test = df[test_mask][[SCENARIO_COL, GROUP_COL, ATTACK_PROFILE_COL]].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, profiles_train, groups_val, groups_test, meta_test

def report_by_profile(y_true, y_pred, meta):
    """Recall per profilo sulle righe positive + FPR sulle righe benigne."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    prof = meta[ATTACK_PROFILE_COL].astype(str).to_numpy()
    out = {}
    print("Recall per profilo di attacco (righe con label=1):")
    for p in sorted(set(prof[y_true == 1])):
        m = (y_true == 1) & (prof == p)
        rec = float((y_pred[m] == 1).mean())
        out[f"recall{p}"] = rec
        print(f"  {p:<12} n={m.sum():>6}  recall={rec:.4f}")
    neg = y_true == 0
    out["fpr"] = float((y_pred[neg] == 1).mean())
    print(f"  FPR (righe benigne): {out['fpr']:.4f}")
    return out


def detection_latency(y_true, y_pred, meta):
    """Per ogni scenario di attacco: round (group_id) tra l'inizio dell'attacco
    (primo label=1) e il primo allarme a partire da quel punto."""
    df = meta.assign(y=np.asarray(y_true), p=np.asarray(y_pred))
    rows = []
    for scen, g in df.groupby(SCENARIO_COL, sort=False):
        g = g.sort_values(GROUP_COL)
        pos = g[g.y == 1]
        if pos.empty:
            continue
        onset = pos[GROUP_COL].iloc[0]
        hits = g[(g[GROUP_COL] >= onset) & (g.p == 1)]
        rows.append({
            "scenario": scen,
            "profile": str(pos[ATTACK_PROFILE_COL].mode().iat[0]),
            "latency": (hits[GROUP_COL].iloc[0] - onset) if len(hits) else np.nan,
        })
    lat = pd.DataFrame(rows)
    out = {}
    print("Latenza di rilevamento (round dall'inizio dell'attacco):")
    for p, sub in lat.groupby("profile"):
        missed = int(sub.latency.isna().sum())
        det = sub.latency.dropna()
        mean_l = float(det.mean()) if len(det) else np.nan
        med_l = float(det.median()) if len(det) else np.nan
        out[f"latmean{p}"] = mean_l
        out[f"missed{p}"] = missed
        print(f"  {p:<12} scenari={len(sub)}  mancati={missed}  "
              f"media={mean_l:.2f}  mediana={med_l:.2f}")
    return out

def evaluate(name, y_true, y_pred, results, elapsed=None, meta=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    print(f"\n=== {name} ===")
    if elapsed is not None:
        print(f"Tempo di training: {elapsed:.1f}s")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_true, y_pred))

    row = {"model": name, "accuracy": acc, "precision": prec,
           "recall": rec, "f1": f1, "train_time_s": elapsed}
    if meta is not None:
        row.update(report_by_profile(y_true, y_pred, meta))
        row.update(detection_latency(y_true, y_pred, meta))
    results.append(row)


def make_group_kfold_splits(X, groups, profiles, n_splits):
    """Ritorna gli split di CV, con vincolo di gruppo per scenario_id
    (nessuno scenario a cavallo di due fold) E stratificati per profilo di
    attacco, cosi' che ogni fold riceva una proporzione simile di scenari
    aggressive/moderate/stealth. Riduce n_splits se ci sono meno gruppi
    disponibili.
    NB: `groups` qui deve essere scenario_id (non group_id di riga), altrimenti
    con gruppi quasi-singoli la StratifiedGroupKFold degenera in una KFold
    senza shuffle."""
    n_groups = groups.nunique()
    effective_splits = min(n_splits, n_groups)
    if effective_splits < 2:
        raise ValueError(
            f"Servono almeno 2 gruppi distinti per la StratifiedGroupKFold, trovati {n_groups}."
        )
    if effective_splits < n_splits:
        print(f"[!] Solo {n_groups} gruppi disponibili: uso {effective_splits}-fold invece di {n_splits}-fold.")
    sgkf = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_SEED)
    return list(sgkf.split(X, profiles, groups=groups))


# --------------------------------------------------------------------------
# 1. CatBoost - tuning Optuna con StratifiedGroupKFold per scenario_id +
#    training finale con early stopping su validation (mai sul test)
# --------------------------------------------------------------------------
def tune_catboost(X_train, y_train, groups_train, profiles_train, n_trials=N_TRIALS, n_splits=N_SPLITS):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train, n_splits)

    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
            'rsm': trial.suggest_float('rsm', 0.5, 1.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 3.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
            'iterations': 1000,
            'loss_function': 'Logloss',
            'eval_metric': 'F1',
            'random_seed': RANDOM_SEED,
            'auto_class_weights': 'Balanced',
        }

        fold_scores = []
        for tr_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            train_pool = Pool(data=X_tr, label=y_tr)
            val_pool = Pool(data=X_val, label=y_val)

            model = CatBoostClassifier(**params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=False,
                early_stopping_rounds=CV_EARLY_STOPPING_ROUNDS,
                use_best_model=True,
            )
            best_score = model.get_best_score()['validation']['F1']
            fold_scores.append(best_score)

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction='maximize',
        study_name='catboost_tuning',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[CatBoost] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[CatBoost] Migliori parametri: {study.best_params}")

    return study.best_params


def train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_params, meta_test=None):
    params = {
        'iterations': 1000,
        'loss_function': 'Logloss',
        'eval_metric': 'F1',
        'random_seed': RANDOM_SEED,
        'auto_class_weights': 'Balanced',
        **best_params,
    }

    train_pool = Pool(data=X_train, label=y_train)
    val_pool = Pool(data=X_val, label=y_val)
    test_pool = Pool(data=X_test, label=y_test)

    model = CatBoostClassifier(**params)

    t0 = time.time()
    model.fit(
        train_pool,
        eval_set=val_pool,          # <-- early stopping SOLO su validation
        verbose=50,
        early_stopping_rounds=FINAL_EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(test_pool)   # <-- test toccato una sola volta, qui
    evaluate("CatBoost", y_test, y_pred, results, elapsed, meta=meta_test)

    print(model.get_feature_importance(prettified=True))

    out_path = os.path.join(MODELS_DIR, "catboost_model.cbm")
    model.save_model(out_path)
    print(f"Model path: {out_path}")

    return model


# --------------------------------------------------------------------------
# 2. XGBoost - tuning Optuna con StratifiedGroupKFold per scenario_id +
#    training finale con early stopping su validation (mai sul test)
# --------------------------------------------------------------------------
def tune_xgboost(X_train, y_train, groups_train, profiles_train, n_trials=N_TRIALS, n_splits=N_SPLITS):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train, n_splits)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 15.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'n_estimators': 1000,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': RANDOM_SEED,
            'early_stopping_rounds': CV_EARLY_STOPPING_ROUNDS,
            'n_jobs': -1,
        }

        fold_scores = []
        for tr_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            n_pos = (y_tr == 1).sum()
            n_neg = (y_tr == 0).sum()
            scale_pos_weight = n_neg / max(n_pos, 1)

            model = XGBClassifier(scale_pos_weight=scale_pos_weight, **params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            y_pred = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred, average='binary', zero_division=0))

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_tuning',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n[XGBoost] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[XGBoost] Migliori parametri: {study.best_params}")

    return study.best_params


def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_params, meta_test=None):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        'n_estimators': 1000,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_SEED,
        'scale_pos_weight': scale_pos_weight,
        'early_stopping_rounds': FINAL_EARLY_STOPPING_ROUNDS,
        'n_jobs': -1,
        **best_params,
    }

    model = XGBClassifier(**params)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   # <-- early stopping SOLO su validation
        verbose=50,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)   # <-- test toccato una sola volta, qui
    evaluate("XGBoost", y_test, y_pred, results, elapsed, meta=meta_test)

    importances = pd.Series(model.feature_importances_, index=FEATURES)
    print(importances.sort_values(ascending=False))

    out_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    model.save_model(out_path)
    print(f"Model path: {out_path}")

    return model


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

# --------------------------------------------------------------------------
# Main: tuning + training di tutti i modelli sullo stesso split + confronto
# finale
# --------------------------------------------------------------------------
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, groups_train, profiles_train, groups_val, groups_test, meta_test = load_data()

    results = []

    # I modelli commentati restano a None cosi' il salvataggio finale non fallisce
    best_cb_params = None
    best_xgb_params = None
    best_bert_params = None

    print("\n\n=== Tuning CatBoost (Optuna + StratifiedGroupKFold per scenario_id/profilo) ===")
    best_cb_params = tune_catboost(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning XGBoost (Optuna + StratifiedGroupKFold per scenario_id/profilo) ===")
    best_xgb_params = tune_xgboost(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning HF-BERT (Optuna + StratifiedGroupKFold per scenario_id/profilo) ===")
    #best_bert_params = tune_hf_bert(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Tuning SeqBERT ===")
    best_seq_params = tune_seq(X_train, y_train, groups_train, profiles_train)

    print("\n\n=== Training finale CatBoost ===")
    train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_cb_params, meta_test)

    print("\n\n=== Training finale XGBoost ===")
    train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_xgb_params, meta_test)

    print("\n\n=== Training finale HF-BERT ===")
    #train_hf_bert(X_train, X_val, X_test, y_train, y_val, y_test, results, best_bert_params, meta_test)

    print("\n\n=== Training finale SeqBERT ===")
    train_seq(X_train, X_val, X_test, y_train, y_val, y_test,
            groups_train, groups_val, groups_test, results, best_seq_params, meta_test)

    comparison = pd.DataFrame(results).set_index("model")
    print("\n\n=== Confronto finale ===")
    print(comparison.to_string(float_format="%.4f"))

    comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison.to_csv(comparison_path)
    print(f"\nTabella di confronto salvata in: {comparison_path}")

    best_params_path = os.path.join(MODELS_DIR, "best_params.joblib")
    joblib.dump(
        {"catboost": best_cb_params, "xgboost": best_xgb_params, "hf_bert": best_bert_params, "seq": best_seq_params},
        best_params_path,
    )
    print(f"Migliori iperparametri salvati in: {best_params_path}")