import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import reduce_loss
from .utils import weight_reduce_loss

@LOSSES.register_module()
class DualFocalLoss(nn.Module):
    def __init__(self, 
                 gamma=0, 
                 size_average=False, 
                 reduction='mean', 
                 loss_weight=1.0):
        super(DualFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, 
                pred, 
                target, 
                weight=None, 
                avg_factor=None):
        # pred as (num, logits)
        # target as (num, )
        num_classes = pred.size(1)
        target = target.view(-1, 1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        logp_k = F.log_softmax(pred, dim=1)
        softmax_logits = logp_k.exp()
        logp_k = logp_k.gather(1, target)
        logp_k = logp_k.view(-1)
        p_k = logp_k.exp()
        p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1
        p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

        loss = -1 * (1 - p_k + p_j) ** self.gamma * logp_k

        if weight is not None:
            loss = loss * weight

        if avg_factor is None:
            loss = reduce_loss(loss, self.reduction)
        else:
            if self.reduction == 'mean':
                eps = torch.finfo(torch.float32).eps
                loss = loss.sum() / (avg_factor + eps)
            elif self.reduction != 'none':
                raise ValueError('avg_factor can not be used with reduction="sum"')

        return loss * self.loss_weight
