import torch
import torch.nn as nn


class DistributionCoherenceLoss(nn.Module):
    """
    Compute the Unsupervised KL-based Distribution Coherence Loss

    PARAMETERS
    ----------
    global_weight: float
        Global weight to apply on the entire loss
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(DistributionCoherenceLoss, self).__init__()
        self.global_weight = global_weight
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, rd_input: torch.Tensor,
                ra_input: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the loss between the two predicted view masks"""
        rd_softmax = nn.Softmax(dim=1)(rd_input)
        ra_softmax = nn.LogSoftmax(dim=1)(ra_input)
        rd_range_probs = torch.max(rd_softmax, dim=3, keepdim=True)[0]
        # Rotate RD Range vect to match zero range
        rd_range_probs = torch.rot90(rd_range_probs, 2, [2, 3])
        ra_range_log_probs = torch.max(ra_softmax, dim=3, keepdim=True)[0]
        coherence_loss = self.kl_div(ra_range_log_probs, rd_range_probs)
        weighted_coherence_loss = self.global_weight*coherence_loss
        return weighted_coherence_loss
