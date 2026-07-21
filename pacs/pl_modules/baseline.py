import torch
from torch import nn
import lightning as pl
from torchmetrics.functional.classification import accuracy

from ..data.data_module import DOMAIN_ENUM
from ..models.loss import PACSSingleLabelClassificationLoss

class PACSLitModule(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 task_weights: list = None,
                 lr: float = 0.0001,
                 w_decay: float = 0,
                 min_lr: float = 1e-6
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 3, 224, 224) # 3チャンネルRGB
        self.criterion = PACSSingleLabelClassificationLoss(weights=task_weights)
        self.num_classes = 7

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.w_decay)
        if self.min_lr is not None:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=self.min_lr
            )
            return [optimizer], [lr_scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        labels = batch['lab']

        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        
        acc = accuracy(logits, labels, task="multiclass", num_classes=self.num_classes)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch['img']
        labels = batch['lab']

        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        
        acc = accuracy(logits, labels, task="multiclass", num_classes=self.num_classes)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss