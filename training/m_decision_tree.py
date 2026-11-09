"""
DECISION TREE tuning + training
"""
import time
import joblib
from sklearn.tree import DecisionTreeClassifier
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

def tune_decision_tree(X_train, y_train, groups_train, profiles_train, plots_dir=None):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_categorical('max_depth', TREE_MAX_DEPTH),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *TREE_MIN_SAMPLES_LEAF, log=True),
            'max_features': trial.suggest_float('max_features', *TREE_MAX_FEATURES),
            'criterion': 'gini',
            'class_weight': 'balanced',
            'random_state': RANDOM_SEED,
        }

        fold_scores = []
        for tr_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            model = DecisionTreeClassifier(**params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_pred, average='binary', zero_division=0))

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction='maximize',
        study_name='decision_tree_tuning',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    if plots_dir:
        save_study_plots(study, "DecisionTree", plots_dir)

    print(f"\n[DecisionTree] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[DecisionTree] Migliori parametri: {study.best_params}")

    return study.best_params


def train_decision_tree(X_train, X_test, y_train, y_test, results, best_params, features, model_dir: str):
    params = {
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED,
        **best_params,
    }

    model = DecisionTreeClassifier(**params)

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    evaluate("DecisionTree", y_test, y_pred, results, elapsed)
    print(f"Profondità albero: {model.get_depth()} | Foglie: {model.get_n_leaves()}")

    ## Built-in feature importance (Gini / impurity decrease)
    #importances = pd.Series(model.feature_importances_, index=features)
    #print(importances.sort_values(ascending=False))

    ## F1-score permutation importance
    importances = permutation_importance(
        lambda X: model.predict(X), X_test, y_test, features
    )
    print(importances.sort_values(ascending=False))
    #importances_pct = to_percent(importances)
    #print(importances_pct.sort_values(ascending=False))

    out_path = os.path.join(model_dir, "decision_tree_model.joblib")
    joblib.dump(model, out_path)
    print(f"Model path: {out_path}")

    return model
