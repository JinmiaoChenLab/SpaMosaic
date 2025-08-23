"""Loss functions for contrastive alignment in SpaMosaic.

Provides a multi-view contrastive loss and a small utility to set a matrix diagonal.
"""

import torch
import torch.nn.functional as F


def set_diag(matrix, v):
    """Set the diagonal of a square tensor to a specified value.

    Parameters
    ----------
    matrix : torch.Tensor
        2D square tensor of shape ``[N, N]`` whose diagonal will be modified.
    v : float
        Value to set on the diagonal.

    Returns
    -------
    torch.Tensor
        The same tensor with its diagonal set to ``v``.
    """
    mask = torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
    matrix[mask] = v
    return matrix


class CL_loss(torch.nn.Module):
    """Contrastive loss for multi-view (multi-modality) alignment.

    Encourages embeddings of the same sample across modalities to be similar
    while pushing apart embeddings from different samples.

    Parameters
    ----------
    batch_size : int
        Number of samples per mini-batch (per modality).
    rep : int, optional
        Number of modalities or views. Default is ``3``.
    bias : float, optional
        Small constant added inside the log term for numerical stability.
        Default is ``0``.
    """
    def __init__(self, batch_size, rep=3, bias=0):
        super().__init__()
        self.batch_size = batch_size
        self.n_mods = rep
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * rep, batch_size * rep, dtype=bool)).float()
        )
        iids = torch.arange(batch_size).repeat(rep)
        pos_mask = set_diag(iids.view(-1, 1) == iids.view(1, -1), 0)
        self.register_buffer('pos_mask', pos_mask.float())
        self.bias = bias

    def forward(self, simi):
        """Compute the contrastive loss from a similarity matrix.

        Parameters
        ----------
        simi : torch.Tensor
            Pairwise similarity matrix of shape ``[B * rep, B * rep]``,
            where ``B`` is ``batch_size``.

        Returns
        -------
        torch.Tensor
            A scalar tensor containing the loss value.
        """
        # stabilize logits by subtracting per-row max
        simi_max, _ = torch.max(simi, dim=1, keepdim=True)
        simi = simi - simi_max.detach()

        # positives: average over same-sample, cross-modality positions
        positives = (simi * self.pos_mask).sum(dim=1) / self.pos_mask.sum(dim=1)

        # negatives: log-sum-exp over the rest
        negatives = (torch.exp(simi) * self.negatives_mask).sum(dim=1)

        # bias helps prevent log(0) and extreme gradients
        loss = -(positives - torch.log(negatives + self.bias)).mean()
        return loss
