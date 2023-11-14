from torch import nn as nn


class FeedForwardBlock(nn.Module):
    """A single block for the FFN containing Dense, LayerNorm, ReLU, and Dropout layers."""

    def __init__(self, in_dim, out_dim, dropout_rate, use_relu):
        super(FeedForwardBlock, self).__init__()

        self.dense = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU() if use_relu else None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.dense(x)
        out = self.norm(out)
        out = out if self.activation is None else self.activation(out)
        return self.dropout(out)


class FFN(nn.Module):
    """Feedforward Neural Network with configurable blocks."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks, dropout_rate, relu_on_last_layer=False):
        super(FFN, self).__init__()

        layers = []
        in_features = input_dim

        # Add n-1 blocks with given hidden dimensions
        for _ in range(n_blocks - 1):
            layers.append(FeedForwardBlock(in_features, hidden_dim, dropout_rate, True))
            in_features = hidden_dim

        # Add the last block with output dimensions
        layers.append(FeedForwardBlock(in_features, output_dim, dropout_rate, relu_on_last_layer))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
