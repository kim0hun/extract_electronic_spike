import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()

        self.seq_len = seq_len

        self.encoder = nn.LSTM(
            n_features,
            embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.decoder = nn.LSTM(
            embedding_dim,
            embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden = hidden[-1]

        hidden = hidden.unsqueeze(1).repeat(1, self.seq_len, 1)

        decoded, _ = self.decoder(hidden)
        out = torch.sigmoid(self.output_layer(decoded))

        return out
