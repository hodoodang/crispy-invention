import torch.nn as nn


class FNN(nn.Module):
    """Position-wise Feed-Forward Network
    """
    def __init__ (self, fc1, fc2, dr_rate=0.1):
        super().__init__()

        self.fc1 = fc1
        self.fc2 = fc2
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, x):
        output = x
        output = self.fc1(output)
        output2 = self.relu(output)
        output2 = self.dropout(output2)
        output3 = self.fc2(output2)

        return output3
