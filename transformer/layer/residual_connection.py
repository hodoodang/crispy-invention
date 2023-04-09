import torch.nn as nn


class ResidualConnectionLayer(nn.Module):
    """Residual Connection Layer
    """

    def __init__(self, norm, dr_rate=0):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, x, sub_layer):
        out = x
        out1 = self.norm(out)
        out2 = sub_layer(out1)
        out3 = self.dropout(out2)
        output = out3 + x
        return output
