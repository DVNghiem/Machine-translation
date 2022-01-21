from attention import Decoder as DecoderAtt
from without_attention import Encoder, Decoder
import torch
import torch.nn as nn
from utils import device


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, units, maxlen_target, att=True) -> None:
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, units)
        if att:
            self.decoder = DecoderAtt(vocab_size, embedding_dim, units)
        else:
            self.decoder = Decoder(vocab_size, embedding_dim, units)
        self.maxlen_target = maxlen_target
        self.vocab_size = vocab_size

    def forward(self, inputs):
        en_out, hidden = self.encoder(inputs)
        bs, _, _ = en_out.size()
        out = torch.ones(size=(bs, self.vocab_size, self.maxlen_target))
        inp = torch.tensor([[1]]*bs, device=device)
        for i in range(1, self.maxlen_target):
            predict, hidden = self.decoder(inp, hidden, en_out)
            out[:, :, i] = predict
            inp = torch.argmax(torch.softmax(
                predict, dim=1), dim=1).unsqueeze(-1)
        return out
