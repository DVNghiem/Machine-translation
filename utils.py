import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Loss(object):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.loss_obj = nn.CrossEntropyLoss(reduction='none')
        self.vocab_size = vocab_size

    def loss_fn(self, y_pred, y_true):
        y_onehot = F.one_hot(y_true, num_classes=self.vocab_size).type(
            torch.float32).to(device)
        loss = self.loss_obj(y_pred, y_onehot)
        equal = torch.eq(y_true, torch.zeros_like(y_true, device=device))
        equal = torch.logical_not(equal)
        mask = equal.type(torch.float32)
        return torch.mean(loss*mask)

    def __call__(self, y_pred, y_true):
        loss = 0
        _, _, maxlen = y_pred.size()
        for i in range(maxlen):
            loss += self.loss_fn(y_pred[:, :, i], y_true[:, i])
        return loss/maxlen
