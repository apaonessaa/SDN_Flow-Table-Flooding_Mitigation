"""
BERT tuning + training
"""
import time
import joblib
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import tempfile

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

from utils import * 

# Torch Numpy
torch.manual_seed(RANDOM_SEED)

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tuning
BERT_CV_EPOCHS = 10
BERT_CV_PATIENCE = 5
# Training+Validation
BERT_FINAL_EPOCHS = 20
BERT_FINAL_PATIENCE = 10

N_BINS = [2, 4, 6, 8]

def _tokenize_features(X_fit, X_others, n_bins):
    discretizer = KBinsDiscretizer(
        n_bins=n_bins, 
        encode="ordinal", 
        strategy="quantile", 
        subsample=None
    )
    fit_bins = discretizer.fit_transform(X_fit).astype(np.int64)

    n_features = X_fit.shape[1]
    offsets = (np.arange(n_features) * n_bins).astype(np.int64)

    cls_id = n_features * n_bins
    vocab_size = cls_id + 1

    def _with_cls(bins):
        ids = bins + offsets                                        # (N, F)
        cls_col = np.full((ids.shape[0], 1), cls_id, dtype=np.int64)
        return np.concatenate([cls_col, ids], axis=1)               # (N, F+1)

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

def _compute_metrics(eval_pred):
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
        max_position_embeddings=n_features+1,
        type_vocab_size=1,
        num_labels=2,
    )

def tune_bert(X_train, y_train, groups_train, profiles_train, model_dir:str):
    splits = make_group_kfold_splits(X_train, groups_train, profiles_train)
    n_features = X_train.shape[1]

    def objective(trial):
        params = {
            'n_bins': trial.suggest_categorical('n_bins', N_BINS),
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

            with tempfile.TemporaryDirectory(dir=model_dir) as tmp_dir:
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
                    compute_metrics=_compute_metrics,
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
        study_name='bert_tuning',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n[HF-BERT] Miglior F1 medio in CV: {study.best_value:.4f}")
    print(f"[HF-BERT] Migliori parametri: {study.best_params}")

    return study.best_params

def train_bert(X_train, X_val, X_test, y_train, y_val, y_test, results, best_params, model_dir:str):
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
        output_dir=os.path.join(model_dir, "bert_checkpoints"),
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
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=BERT_FINAL_PATIENCE)],
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    pred_output = trainer.predict(test_ds)
    y_pred = np.argmax(pred_output.predictions, axis=-1)
    evaluate("BERT", y_test, y_pred, results, elapsed)


    out_dir = os.path.join(model_dir, "bert_model")
    model.save_pretrained(out_dir)
    joblib.dump(discretizer, os.path.join(out_dir, "discretizer.joblib"))
    print(f"Model path: {out_dir}")

    return model
