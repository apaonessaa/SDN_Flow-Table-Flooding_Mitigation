"""
RANDOM FOREST tuning + training
"""
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.metrics import f1_score

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils import *

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Range di ricerca specifici per la Random Forest.
# (Se preferisci, spostali in config.py insieme agli altri range.)
RF_N_ESTIMATORS = (100, 500)
RF_DEPTH = (3, 40)
RF_MIN_SAMPLES_SPLIT = (2, 50)
RF_MIN_SAMPLES_LEAF = (1, 30)
RF_MAX_FEATURES = (0.2, 1.0)
RF_MAX_SAMPLES = (0.5, 1.0)


def tune_random_forest(X_train, y_train, groups_train, profiles_train, plots_dir=None):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)

    def objective(trial):
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *RF_N_ESTIMATORS),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'max_depth': trial.suggest_int('max_depth', *RF_DEPTH),
            'min_samples_split': trial.suggest_int('min_samples_split', *RF_MIN_SAMPLES_SPLIT),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *RF_MIN_SAMPLES_LEAF),
            'max_features': trial.suggest_float('max_features', *RF_MAX_FEATURES),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
            'bootstrap': bootstrap,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
        }
        # max_samples ha senso solo con il bootstrap attivo
        if bootstrap:
            params['max_samples'] = trial.suggest_float('max_samples', *RF_MAX_SAMPLES)
        else:
            # balanced_subsample richiede il bootstrap
            if params['class_weight'] == 'balanced_subsample':
                params['class_weight'] = 'balanced'

        fold_scores = []
        for tr_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = RandomForestClassifier(**params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred, average='binary', zero_division=0))

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction='maximize',
        study_name='random_forest_tuning',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    if plots_dir:
        save_study_plots(study, "RandomForest", plots_dir)

    print(f"\n[RandomForest] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[RandomForest] Migliori parametri: {study.best_params}")

    return study.best_params


def train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test, results, best_params, features, model_dir: str):
    # X_val / y_val non servono alla Random Forest (nessun early stopping):
    # restano nella firma per uniformità con gli altri modelli.
    params = {
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        **best_params,
    }
    # Stessa correzione usata nel tuning: balanced_subsample richiede bootstrap=True
    if not params.get('bootstrap', True) and params.get('class_weight') == 'balanced_subsample':
        params['class_weight'] = 'balanced'

    model = RandomForestClassifier(**params)

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    evaluate("RandomForest", y_test, y_pred, results, elapsed)

    ## Built-in feature importance (impurity-based)
    #importances = pd.Series(model.feature_importances_, index=features)
    #print(importances.sort_values(ascending=False))

    ## F1-score permutation importance
    importances = permutation_importance(
        lambda X: model.predict(X), X_test, y_test, features
    )
    print(importances.sort_values(ascending=False))
    #importances_pct = to_percent(importances)
    #print(importances_pct.sort_values(ascending=False))

    out_path = os.path.join(model_dir, "random_forest_model.joblib")
    joblib.dump(model, out_path)
    print(f"Model path: {out_path}")

    return model
