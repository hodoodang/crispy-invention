import copy

import torch.nn as nn
from transformer.layer.residual_connection import ResidualConnectionLayer


class EncoderBlock(nn.Module):
    """Encoder Block
    """

    def __init__(self, self_attention, position_ff, norm, dropout_rate=0):
        """
        Initialize Encoder Block
        Args:
            self_attention: multi-head attention layer
            position_ff: position-wise feed-forward layer
            norm: normalization
            dropout_rate: dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), dropout_rate)
        self.position_ff = position_ff
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), dropout_rate)

    def forward(self, origin, origin_mask):
        """
        Args:
            origin: query, key, value
            origin_mask: mask

        Returns:
            torch.Tensor:
        """
        out = self.residual1(origin, lambda x: self.self_attention(query=origin, key=origin,
                                                                   value=origin, mask=origin_mask))
        out = self.residual2(out, self.position_ff)
        return out
