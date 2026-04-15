import torch
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class InvariantGRU(nn.Module):

    def __init__(self, input_dim, hidden=64, n_hospitals=10):
        super().__init__()

        self.encoder = nn.GRU(input_dim, hidden, batch_first=True)

        self.task_head = nn.Linear(hidden, 1)

        self.hospital_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_hospitals)
        )

    def forward(self, x, alpha=1.0):

        _, h = self.encoder(x)
        h = h.squeeze(0)

        y_pred = self.task_head(h).squeeze(-1)

        h_rev = grad_reverse(h, alpha)
        h_pred = self.hospital_head(h_rev)

        return y_pred, h_pred
