import torch.nn as nn


class Transformer(nn.Module):
    """Transformer
    Consists of Encoder and Decoder
    """

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, mask):
        """
        Encoding process
        Args:
            x: sentence
            mask

        Returns:
            context vector
        """
        return self.encoder(x, mask)

    def decode(self, z, c):
        """
        Decoding process
        Args:
            z: sentence
            c: context vector

        Returns:
            sentence
        """
        return self.decoder(z, c)

    def forward(self, x, z, mask):
        c = self.encode(x, mask)
        y = self.decode(z, c)
        return y

    def make_source_mask(self, x):
        return make_pad_mask(x, x)


def make_pad_mask(query, key, pad_idx=1):
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len)

    mask = key_mask & query_mask
    mask.requires_grad = False
    return mask
