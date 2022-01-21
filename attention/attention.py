import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query, value):
        score = self.V(torch.tanh(self.W1(query)+self.W2(value)))
        att_weight = torch.softmax(score, axis=1)
        context_vector = att_weight+value
        return torch.sum(context_vector, axis=1)  # batch_size, hidden_size
