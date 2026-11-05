"""
CATBOOST tuning + training
"""
import time
from catboost import CatBoostClassifier, Pool
import optuna

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils import * 

optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_catboost(X_train, y_train, groups_train, profiles_train, plots_dir=None):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)

    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', *DEPTH),
            'learning_rate': trial.suggest_float('learning_rate', *BOOST_LEARNING_RATE, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *L2_REG, log=True),
            'rsm': trial.suggest_float('rsm', *COLSAMPLE),
            'bagging_temperature': trial.suggest_float('bagging_temperature', *BAGGING_TEMPERATURE),
            'iterations': ITERATIONS,
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
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    if plots_dir:
        save_study_plots(study, "CatBoost", plots_dir)

    print(f"\n[CatBoost] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[CatBoost] Migliori parametri: {study.best_params}")

    return study.best_params

def train_catboost(X_train, X_val, X_test, y_train, y_val, y_test, results, best_params, model_dir:str):
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
        eval_set=val_pool,
        verbose=50,
        early_stopping_rounds=FINAL_EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    elapsed = time.time() - t0

    y_pred = model.predict(test_pool)
    evaluate("CatBoost", y_test, y_pred, results, elapsed)

    ## Built-in feature importance
    #print(model.get_feature_importance(prettified=True)) 
    
    ## F1-score permutation importance
    features = list(X_test.columns)
    importances = permutation_importance(
        lambda X: model.predict(X).ravel(), X_test, y_test, features
    )
    print(importances.sort_values(ascending=False))
    #importances_pct = to_percent(importances)
    #print(importances_pct.sort_values(ascending=False))

    out_path = os.path.join(model_dir, "catboost_model.cbm")
    model.save_model(out_path)
    print(f"Model path: {out_path}")

    return model

