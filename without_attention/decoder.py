from turtle import forward
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, units) -> None:
        super(Decoder, self).__init__()

        # bs, 1, embedding dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # bs, 1, units
        self.gru = nn.GRU(embedding_dim, units, batch_first=True)

        self.fc = nn.Linear(units, units)
        self.out = nn.Linear(units, vocab_size)

    def forward(self, x, prev_hidden, enc_output=None):
        emb = self.embedding(x)
        context, hidden = self.gru(emb, prev_hidden)
        context = context.squeeze(1)
        x = self.fc(context)
        x = self.out(x)
        return x, hidden
