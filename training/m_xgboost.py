"""
XGBOOOST tuning + training
"""
import time
from xgboost import XGBClassifier
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

def tune_xgboost(X_train, y_train, groups_train, profiles_train, plots_dir=None):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', *DEPTH),
            'learning_rate': trial.suggest_float('learning_rate', *BOOST_LEARNING_RATE, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *LAMBDA, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *COLSAMPLE),
            'subsample': trial.suggest_float('subsample', *SUBSAMPLE),
            'min_child_weight': trial.suggest_float('min_child_weight', *MIN_CHILD_WEIGHT, log=True),
            'n_estimators': ITERATIONS,
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
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    if plots_dir:
        save_study_plots(study, "XGBoost", plots_dir)

    print(f"\n[XGBoost] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[XGBoost] Migliori parametri: {study.best_params}")

    return study.best_params

def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_params, features, model_dir:str):
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        'n_estimators': ITERATIONS,
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
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    evaluate("XGBoost", y_test, y_pred, results, elapsed)

    ## Built-in feature importance
    #importances = pd.Series(model.feature_importances_, index=features)
    #print(importances.sort_values(ascending=False))

    ## F1-score permutation importance
    importances = permutation_importance(
        lambda X: model.predict(X), X_test, y_test, features
    )
    print(importances.sort_values(ascending=False))
    #importances_pct = to_percent(importances)
    #print(importances_pct.sort_values(ascending=False))

    out_path = os.path.join(model_dir, "xgboost_model.json")
    model.save_model(out_path)
    print(f"Model path: {out_path}")

    return model
