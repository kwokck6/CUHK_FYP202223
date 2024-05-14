import torch
from torch.nn import Module, PairwiseDistance


class TripletLoss(Module):
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin
        self.dist_fn = PairwiseDistance()

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        p_dist = self.dist_fn(anchor, pos)
        n_dist = self.dist_fn(anchor, neg)
        loss = torch.clamp(torch.pow(p_dist, 2) - torch.pow(n_dist, 2) + self.margin, min=0.0)
        loss = torch.mean(loss)
        return loss
