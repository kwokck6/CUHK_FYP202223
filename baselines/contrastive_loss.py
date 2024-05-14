import torch
from torch.nn import Module, PairwiseDistance


class ContrastiveLoss(Module):
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
        self.dist_fn = PairwiseDistance()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = self.dist_fn(input1, input2)
        loss = (target) * torch.pow(dist, 2) + \
            (1 - target) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(loss)
        return loss
