"""
Componenti condivisi da FeatBERT e TempBERT

  - scale_features : preprocessing (identico per i due modelli)
  - init_uniform_  : regola unica di inizializzazione dei parametri di input
  - fit_model      : ciclo di training (AdamW, lr costante, early stopping su F1)
  - predict_model  : inferenza a batch

"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from utils import RANDOM_SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parametri esclusi dal weight decay (oltre a bias e LayerNorm, cioè i tensori con ndim < 2).
# Come in FT-Transformer (Gorishniy et al., 2021, App. E.3): tokenizer/proiezione di input e [CLS] non decadono.
NO_DECAY_PREFIXES = ("tokenizer.", "proj.", "cls")

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def scale_features(X_fit, *X_others):
    """
    Data pre-processing: Gorishniy et al. 2021
    """
    pipeline = Pipeline([
        ("quantile", QuantileTransformer(output_distribution="normal", random_state=RANDOM_SEED))
    ])
    pipeline.fit(np.asarray(X_fit, dtype=np.float64))

    def _tf(X):
        return pipeline.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)

    return (_tf(X_fit), *[_tf(X) for X in X_others], pipeline)

# --------------------------------------------------------------------------- #
# Inizializzazione
# --------------------------------------------------------------------------- #
def init_uniform_(*tensors, fan_in):
    """
    Regola unica per tutti i parametri di input dei due modelli (token di feature, proiezione del
    round, [CLS]):  U(-1/sqrt(fan_in), +1/sqrt(fan_in)).

    È kaiming_uniform_(a=sqrt(5)): il default di nn.Linear e lo schema del codice di FT-Transformer.
    fan_in = d per i vettori-token (pesi, bias e [CLS], con il layout (n, d) del codice originale);
    fan_in = F per la Linear(F, d) di TempBERT (coincide con il default di PyTorch).
    L'encoder BERT mantiene l'inizializzazione di HuggingFace.
    """
    bound = float(fan_in) ** -0.5
    for t in tensors:
        nn.init.uniform_(t, -bound, bound)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def class_weights(y):
    counts = np.bincount(np.asarray(y), minlength=2)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def _loader(inputs, y, batch_size, shuffle):
    ds = TensorDataset(*[torch.from_numpy(a) for a in inputs],
                       torch.as_tensor(np.asarray(y), dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_model(model, inputs, batch_size=512):
    """inputs: tuple di array numpy; il modello riceve model(*inputs)."""
    model.eval()
    preds = []
    for *xs, _ in _loader(inputs, np.zeros(len(inputs[0])), batch_size, False):
        preds.append(model(*[x.to(DEVICE) for x in xs]).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def fit_model(model_fn, train_inputs, y_tr, val_inputs, y_va, params, epochs, patience):
    """
    Ciclo di training unico: AdamW con learning rate costante (nessuno scheduler, come in
    Gorishniy et al. 2021, §4.3), loss pesata per classe, clip dei gradienti a 1.0,
    early stopping su F1 di validazione e ripristino del modello migliore.
    params deve contenere: learning_rate, weight_decay, batch_size.
    """
    torch.manual_seed(RANDOM_SEED)
    model = model_fn().to(DEVICE)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or name.startswith(NO_DECAY_PREFIXES) else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": params["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=params["learning_rate"],
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr).to(DEVICE))
    loader = _loader(train_inputs, y_tr, params["batch_size"], shuffle=True)

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for *xs, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(*[x.to(DEVICE) for x in xs]), yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        f1 = f1_score(y_va, predict_model(model, val_inputs), zero_division=0)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_f1
