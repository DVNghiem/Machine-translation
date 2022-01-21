from turtle import forward
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, units) -> None:
        super(Encoder, self).__init__()
        # shape = bs, maxlen, embedding dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # shape = bs, maxlen, units
        self.gru = nn.GRU(embedding_dim, units, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        output, hidden = self.gru(emb)
        return output, hidden
