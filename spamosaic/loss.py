import torch
import torch.nn.functional as F

def set_diag(matrix, v):
    mask = torch.eye(matrix.size(0), dtype=torch.bool)
    matrix[mask] = v
    return matrix

class CL_loss(torch.nn.Module):
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
        simi_max, _ = torch.max(simi, dim=1, keepdim=True)
        simi = simi - simi_max.detach()

        positives = (simi * self.pos_mask).sum(dim=1) / self.pos_mask.sum(dim=1)
        negatives = (torch.exp(simi) * self.negatives_mask).sum(dim=1)
        loss = -(positives - torch.log(negatives+self.bias)).mean()   # adding a non-zero constant in case grad explosion

        # print('pos', positives)
        # print('neg', negatives)
        # print('loss', loss)

        return loss