import torch
import torch.nn as nn
from .attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)

        self.att = Attention(hidden_size)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(0.25)

    def forward(self, ipt, hidden, enc_output):
        x = self.embedding(ipt)
        outputs, state = self.gru(x, hidden)

        context = self.att(outputs, enc_output)
        outputs = outputs.squeeze(1)

        outputs = torch.cat([outputs, context], dim=1)
        outputs = torch.tanh(self.fc1(outputs))
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        return outputs, state
