import copy

import torch.nn as nn


class Encoder(nn.Module):
    """Encoder
    Encoder of Transformer
    """

    def __init__(self, encoder_block, n_layer, norm):
        """
        Initialize Encoder
        Args:
            encoder_block: encoder block
            n_layer: num of encoder block
            norm: normalization
        """
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        self.norm = norm

    def forward(self, source, source_mask):
        """

        Args:
            source: embedding vector of sentence
            source_mask: masking vector

        Returns:
            context vector
        """
        out = source
        for layer in self.layers:
            out = layer(out, source_mask)
        out = self.norm(out)
        return out
