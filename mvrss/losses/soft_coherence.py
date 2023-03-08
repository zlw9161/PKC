#### zlw@20220304 ####
#### SofCoL using idea of Contrastive Learning ####
import torch
import torch.nn as nn

class SoftCoherenceLoss(nn.Module):
    """
    Compute the Unsupervised Soft Coherence Loss

    PARAMETERS
    ----------
    global_weight: float
        Global weight to apply on the entire loss
    relax_factor: float
        Coef to control the relaxation of SofCoL
    margin: a threshold for acceptance of the coherence
    """

    def __init__(self, global_weight: float = 1.,
                 relax_factor: float = 0.2,
                 margin: float = 0.01) -> None:
        super(SoftCoherenceLoss, self).__init__()
        self.global_weight = global_weight
        self.relax_factor = relax_factor
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, rd_input: torch.Tensor,
                ra_input: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the loss between the two predicted view masks"""
        rd_softmax = nn.Softmax(dim=1)(rd_input)
        ra_softmax = nn.Softmax(dim=1)(ra_input)
        rd_range_probs = torch.max(rd_softmax, dim=3, keepdim=True)[0]
        # Rotate RD Range vect to match zero range
        rd_range_probs = torch.rot90(rd_range_probs, 2, [2, 3])
        ra_range_probs = torch.max(ra_softmax, dim=3, keepdim=True)[0]
        metric = self.mse(rd_range_probs, ra_range_probs)
        relax_term = torch.max(torch.tensor([0, torch.tensor(self.margin)-metric]), 0)[0]
        loss = metric*self.relax_factor + (1-self.relax_factor)*relax_term
        weighted_sofcol = self.global_weight*loss
        return weighted_sofcol
