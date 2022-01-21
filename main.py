import torch
from model import Seq2Seq
from utils import device, Loss

maxlen = 5
embedding_dim = 8
vocab_size = 10
units = 8
inputs = torch.tensor([[1, 4, 5, 6, 2],
                       [1, 4, 3, 2, 0]], device=device)

targets = torch.tensor([[1, 3, 5, 8, 2],
                       [1, 3, 7, 2, 0]], device=device)

model = Seq2Seq(vocab_size, embedding_dim, units, maxlen, att=False)
pred = model(inputs)
loss_obj = Loss(vocab_size)
print(loss_obj(pred, targets))
