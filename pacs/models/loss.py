import torch
from torch import nn

class PACSSingleLabelClassificationLoss(nn.Module):
    """Cross Entropy Loss for PACS multi-class classification"""
    def __init__(self, weights=None):
        super().__init__()
        # クラス不均衡の重みがある場合は適用する
        if weights is not None:
            weights = torch.tensor(weights).float()
            self.criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits, labels):
        # labelsがOne-hotではなくクラスインデックス（整数値長）であることを前提とします
        return self.criterion(logits, labels.long())