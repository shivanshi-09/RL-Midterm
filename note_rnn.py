"""One-hot Note RNN with a recurrent state API for later Sequence Tutor work."""

import torch
from torch import nn
from torch.nn import functional as F

NUM_CLASSES = 38


class NoteRNN(nn.Module):
    def __init__(self, hidden_size=100):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(NUM_CLASSES, hidden_size, num_layers=1, batch_first=True)
        self.projection = nn.Linear(hidden_size, NUM_CLASSES)
        # TF's original BasicLSTMCell uses a forget bias of one.
        with torch.no_grad():
            self.lstm.bias_ih_l0.zero_()
            self.lstm.bias_hh_l0.zero_()
            self.lstm.bias_ih_l0[hidden_size:2*hidden_size].fill_(1)

    def forward(self, tokens, state=None):
        """[batch,time] integer events -> logits [batch,time,38], (h,c).

        State starts at zero when omitted. Call with one event and the previous
        state to obtain the next-event prior during sampling or future RL.
        """
        inputs = F.one_hot(tokens.long(), NUM_CLASSES).to(self.projection.weight.dtype)
        output, state = self.lstm(inputs, state)
        return self.projection(output), state


def collate(records):
    """Right-pad inputs; targets use -100 so padding never enters CE."""
    lengths = torch.tensor([len(r["tokens"]) - 1 for r in records])
    x = torch.zeros((len(records), int(lengths.max())), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, r in enumerate(records):
        n = int(lengths[i])
        x[i, :n] = r["tokens"][:-1].long()
        y[i, :n] = r["tokens"][1:].long()
    return x, y


def masked_targets(targets, skip_first=8):
    result = targets.clone()
    result[:, :skip_first] = -100
    return result


def next_event_loss(logits, targets, skip_first=8):
    scored = masked_targets(targets, skip_first)
    if not (scored != -100).any():
        raise ValueError("Batch has no prediction targets after warmup")
    return F.cross_entropy(logits.reshape(-1, NUM_CLASSES), scored.reshape(-1))
