from transformers import BertConfig, BertForSequenceClassification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

# Torch Numpy
import torch
import torch.nn as nn

"""
TempBERT
"""
class TempBert(nn.Module):
    def __init__(self, n_feat, window, params):
        super().__init__()
        config = self._build_bert_config(window, params)
        d = config.hidden_size
        self.proj = nn.Linear(n_feat, d)
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        init_uniform_(self.proj.weight, self.proj.bias, fan_in=n_feat)   # = default di nn.Linear
        init_uniform_(self.cls, fan_in=d)                                # stessa regola di FeatBERT
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

    def _build_bert_config(self, window, params):
        return BertConfig(
            vocab_size=1,
            hidden_size=params['num_attention_heads']*params['head_dim'],
            num_hidden_layers=params['num_hidden_layers'],
            num_attention_heads=params['num_attention_heads'],
            intermediate_size=params['intermediate_size'],
            hidden_dropout_prob=params['dropout'],
            attention_probs_dropout_prob=params['dropout'],
            max_position_embeddings=window+1,
            type_vocab_size=1,
            num_labels=2,
        )

"""
Utilities
"""
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

def init_uniform_(*tensors, fan_in):
    """
    Uniform rule for all input parameters in FeatBERT and TempBERT (feature tokens, 
    round projection, [CLS]):
        
        U(-1/sqrt(fan_in), +1/sqrt(fan_in)).

    This corresponds to `kaiming_uniform_(a=sqrt(5))` - the default for `nn.Linear` and 
    the scheme used in the FT-Transformer codebase.    
    
        `fan_in = d` for token vectors (weights, biases, and the `[CLS]` token, following 
            the `(n, d)` layout from the original code). 
        `fan_in = F` for the `Linear(F, d)` layer in TempBERT (which aligns with the 
        PyTorch default). 
    
    The BERT encoder retains the HuggingFace initialization.
    """
    bound = float(fan_in) ** -0.5
    for t in tensors:
        nn.init.uniform_(t, -bound, bound)

def fit_seq(params, seq_tr, m_tr, y_tr, seq_va, m_va, y_va, epochs, patience):
    return fit_model(lambda: TempBert(seq_tr.shape[2], params['window'], params),
                     (seq_tr, m_tr), y_tr, (seq_va, m_va), y_va, params, epochs, patience)

def predict_seq(model, seq, valid):
    return predict_model(model, (seq, valid))

def prepare_seq(X_fit, g_fit, others, w):
    *xs, scaler = scale_features(X_fit, *[X for X, _ in others])
    groups = [g_fit] + [g for _, g in others]
    out = [build_windows(x, g, w) for x, g in zip(xs, groups)]
    return scaler, out

def build_windows(X, scen, w):
    X = np.asarray(X, dtype=np.float32)
    scen = np.asarray(scen)
    n, f = X.shape
    seq = np.zeros((n, w, f), dtype=np.float32)
    valid = np.zeros((n, w), dtype=bool)

    starts = np.r_[0, np.flatnonzero(scen[1:] != scen[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    assert len(set(scen[starts])) == len(starts), "Order the dataframe by scenario_id (FTF) or scenario_id+flow_key,round (MFR)"

    for s, e in zip(starts, ends):
        for t in range(s, e):
            lo = max(s, t - w + 1)
            k = t - lo + 1
            seq[t, w - k:] = X[lo:t + 1]
            valid[t, w - k:] = True
    return seq, valid