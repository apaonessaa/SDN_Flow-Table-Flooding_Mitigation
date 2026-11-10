from transformers import BertConfig, BertForSequenceClassification

# Torch Numpy
import torch
import torch.nn as nn

from training.config import RANDOM_SEED
torch.manual_seed(RANDOM_SEED)

"""
FTF TempBERT
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
MFR TempBERT
"""
class MFRTempBert(nn.Module):
    def __init__(self, n_feat, window, params):
        super().__init__()
        config = self._build_seq_bert_config(window, params)
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

    def _build_seq_bert_config(self, window, params):
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