import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention
    """

    def __init__(self, d_model, h, qkv_fc, out_fc, dropout_rate=0):
        """
        Initialize Multi-Head Attention Layer
        Args:
            d_model: d_k * h (d_k = vector dimension)
            h: num of parallel
            qkv_fc: fully connected layer for query, key, value
            out_fc: fully connected for output
            dropout_rate: dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, *args, query, key, value, mask=None):
        batch_size = query.size(0)

        def transform(x, fc):
            out = fc(x)
            out = out.view(batch_size, -1, self.h, self.d_k)
            out.transpose(1, 2)
            return out

        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        output = self.calculate_attention(query, key, value, mask, )

        output = output.transpose(1, 2)
        output = output.contiguous().view(batch_size, -1, self.d_model)
        output = self.out_fc(output)
        return output

    def calculate_attention(self, query, key, value, mask=None):
        """Scaled dot product
        query, key, value: (n_batch, seq_len, d_k)
        mask: (n_batch, seq_len, seq_len)
        Args:
            query: embedding vector
            key: embedding vector
            value: embedding vector
            mask: Option
            dropout: dropout rate

        Returns:
            out: torch.Tensor
        """
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1))
        attention_score = attention_score / torch.sqrt(d_k)

        if mask is not None:
            attention_score = torch.masked_fill(attention_score, (mask == 0), -1e9)

        attention_prob = F.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, value)
        return out
