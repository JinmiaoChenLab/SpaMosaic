import torch
import torch.nn.functional as F

def set_diag(matrix, v):
    """
    Set the diagonal of a square matrix to a specified value.

    Parameters
    ----------
    matrix : torch.Tensor
        A 2D square tensor whose diagonal will be modified.
    v : float
        Value to set on the diagonal.

    Returns
    -------
    torch.Tensor
        The modified tensor with updated diagonal values.
    """
    mask = torch.eye(matrix.size(0), dtype=torch.bool)
    matrix[mask] = v
    return matrix

class CL_loss(torch.nn.Module):
    """
    Contrastive Loss for multi-view representation alignment.

    This loss function is designed to encourage representations from 
    the same sample across multiple modalities to be similar, while 
    pushing apart representations from different samples.

    Parameters
    ----------
    batch_size : int
        Number of samples in a mini-batch (per modality).
    rep : int, optional
        Number of modalities or views (default: 3).
    bias : float, optional
        Small constant added to negative sample logits to avoid instability 
        in log computations (default: 0).
    """
    def __init__(self, batch_size, rep=3, bias=0):
        super().__init__()
        self.batch_size = batch_size
        self.n_mods = rep
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * rep, batch_size * rep, 
                                                           dtype=bool)).float())
        iids = torch.arange(batch_size).repeat(rep)
        pos_mask = set_diag(iids.view(-1, 1) == iids.view(1, -1), 0)
        self.register_buffer('pos_mask', pos_mask.float())
        self.bias = bias
            
    def forward(self, simi):
        """
        Compute contrastive loss from a similarity matrix.

        Parameters
        ----------
        simi : torch.Tensor
            Pairwise similarity matrix of shape [B * rep, B * rep].

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        simi_max, _ = torch.max(simi, dim=1, keepdim=True)
        simi = simi - simi_max.detach()

        positives = (simi * self.pos_mask).sum(dim=1) / self.pos_mask.sum(dim=1)
        negatives = (torch.exp(simi) * self.negatives_mask).sum(dim=1)
        loss = -(positives - torch.log(negatives+self.bias)).mean()   # adding a non-zero constant in case grad explosion

        return loss