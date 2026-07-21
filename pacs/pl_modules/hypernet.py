import torch
import wandb
from torch import nn
from lightning.pytorch import LightningModule
from typing import Optional
from torchmetrics.functional.classification import accuracy
from lightning.pytorch.loggers import WandbLogger
from pytorch_metric_learning import losses, miners
from wandb.plot import confusion_matrix

from hyda.utils import load_submodule_from_checkpoint
from ..data.data_module import DOMAIN_ENUM
from ..models.pacs_hypernet import HyResNet18
from ..models.loss import PACSSingleLabelClassificationLoss


class DomainConditionedPACSLitModule(HyResNet18, LightningModule):
    def __init__(self,
                 domain_encoder: nn.Module,
                 domain_classifier: nn.Module,
                 target_domain: Optional[str] = None,
                 dom_clf_ckpt: Optional[str] = None,

                 imagenet_pretrained=True,
                 num_classes=7,
                 task_weights: list = None,

                 hyper_in_size=64,
                 hyper_emb=0,
                 init_method=None,
                 input_var=None,

                 lr: float=0.001,
                 min_lr: float=1e-6,
                 w_decay: float=0,
                 msim_loss_weight: int=0,
                 use_aux_msim_loss: bool=False,
                 sanity_check = False):
                
        super().__init__(domain_encoder=domain_encoder, 
                         domain_classifier=domain_classifier,
                         imagenet_pretrained=imagenet_pretrained,
                         num_classes=num_classes,
                         hyper_in_size=hyper_in_size, 
                         hyper_emb=hyper_emb, 
                         init_method=init_method, 
                         input_var=input_var)
        
        self.save_hyperparameters(ignore=['domain_encoder', 'domain_classifier'])
        self.sanity_check = sanity_check
        self.automatic_optimization = False # 手動最適化

        self.lr = lr
        self.min_lr = min_lr
        self.w_decay = w_decay
        self.example_input_array = torch.Tensor(1, 3, 224, 224)

        self.criterion = PACSSingleLabelClassificationLoss(weights=task_weights)
        self.domain_loss = torch.nn.CrossEntropyLoss()

        self.msim_loss_weight = msim_loss_weight
        self.use_aux_msim_loss = use_aux_msim_loss
        if self.msim_loss_weight > 0:
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
            self.msim_loss = losses.MultiSimilarityLoss(alpha=2, beta=50)
            if self.use_aux_msim_loss:
                self.aux_msim_loss = losses.MultiSimilarityLoss(alpha=2, beta=50)

        self.target_domain = target_domain
        gt2train_ids = {}
        self.cls_names = []
        curr_idx = 0
        for k, v in DOMAIN_ENUM.items():
            if k != self.target_domain:
                gt2train_ids[v] = curr_idx
                curr_idx += 1
                self.cls_names.append(k)
            else:
                gt2train_ids[v] = -1
        gt2train_ids_tensor = torch.tensor([gt2train_ids[k] for k in sorted(gt2train_ids.keys())])
        self.register_buffer("gt2train_ids", gt2train_ids_tensor)

        self.val_domain_labels = []
        self.val_domain_preds = []

        # 【追加】テスト用データ保持リスト
        self.test_domain_labels = []
        self.test_domain_preds = []
    
    def setup(self, stage: Optional[str] = None):
        dom_clf_ckpt = self.hparams.get("dom_clf_ckpt")
        if dom_clf_ckpt is not None:
            print("loading from", dom_clf_ckpt)
            ckpt = torch.load(dom_clf_ckpt, map_location="cpu")

            self.domain_encoder = load_submodule_from_checkpoint(
                ckpt, "encoder", existing_module=self.domain_encoder
            )
            self.domain_classifier = load_submodule_from_checkpoint(
                ckpt, "classifier", existing_module=self.domain_classifier
            )

    def configure_optimizers(self):
        resnet_features = [p for n, p in self.model.named_parameters() if not n.startswith('fc')]
        
        pacs_optimizer = torch.optim.AdamW(resnet_features +
                                           list(self.model.fc.parameters()) +
                                           list(self.hyper_encode.parameters()),
                                           lr=self.lr, weight_decay=self.w_decay)
        pacs_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pacs_optimizer, T_max=self.trainer.max_epochs, eta_min=self.min_lr)
        
        domain_optimizer = torch.optim.AdamW(list(self.domain_encoder.parameters()) +
                                             list(self.domain_classifier.parameters()) +
                                             list(self.hyper_encode.parameters()),
                                             lr=self.lr, weight_decay=self.w_decay)
        domain_sched = torch.optim.lr_scheduler.CosineAnnealingLR(domain_optimizer, T_max=self.trainer.max_epochs, eta_min=self.min_lr)
        return [pacs_optimizer, domain_optimizer], [pacs_sched, domain_sched]
        #return [pacs_optimizer, domain_optimizer]

    def _step(self, batch):
        img = batch['img']
        domains = self.gt2train_ids[batch['domain']]
        labels = batch['lab']
        logits, dom_logits, h_emb, h_out = self(img)
        return logits, dom_logits, labels, domains, h_emb, h_out

    def training_step(self, batch, batch_idx):
        pacs_opt, dom_opt = self.optimizers()
        logits, dom_logits, labels, domains, h_emb, h_out = self._step(batch)

        pacs_loss = self.criterion(logits, labels)
        self.log('train_loss', pacs_loss, prog_bar=True)

        pacs_opt.zero_grad()
        self.manual_backward(pacs_loss, retain_graph=True)
        pacs_opt.step()

        domain_loss = self.domain_loss(dom_logits, domains)
        if self.msim_loss_weight > 0:
            hard_pairs = self.miner(dom_logits, domains)
            msim_loss = self.msim_loss(dom_logits, domains, hard_pairs)
            if self.use_aux_msim_loss:
                aux_hard_pairs = self.miner(h_emb, domains)
                aux_msim_loss = self.aux_msim_loss(h_emb, domains, aux_hard_pairs)
                msim_loss = msim_loss + aux_msim_loss
            domain_loss = domain_loss + msim_loss * self.msim_loss_weight
            self.log('train_msim_loss', msim_loss)
        self.log('train_domain_loss', domain_loss, prog_bar=True)

        dom_opt.zero_grad()
        self.manual_backward(domain_loss)
        dom_opt.step()

        acc = accuracy(logits, labels, task="multiclass", num_classes=7)
        self.log('train_acc', acc, prog_bar=True)

    def on_train_epoch_end(self):
        pacs_sch, dom_sch = self.lr_schedulers()
        pacs_sch.step()
        dom_sch.step()

    def validation_step(self, batch, batch_idx):
        logits, dom_logits, labels, domains, h_emb, h_out = self._step(batch)

        pacs_loss = self.criterion(logits, labels)
        self.log('val_loss', pacs_loss, prog_bar=True)

        domain_loss = self.domain_loss(dom_logits, domains)
        self.log('val_domain_loss', domain_loss, prog_bar=True)

        acc = accuracy(logits, labels, task="multiclass", num_classes=7)
        self.log('val_acc', acc, prog_bar=True)

        if self.current_epoch % 10 == 0:
            self.val_domain_preds.append(dom_logits.argmax(1).detach())
            self.val_domain_labels.append(domains.detach())

    def on_validation_epoch_end(self):
        if self.current_epoch % 10 == 0 and len(self.val_domain_labels) > 0:
            val_domain_labels = torch.cat(self.val_domain_labels).cpu().numpy()
            val_domain_preds = torch.cat(self.val_domain_preds).cpu().numpy()
            if isinstance(self.logger, WandbLogger):
                wandb.log({"conf_mat": confusion_matrix(preds=val_domain_preds, y_true=val_domain_labels, class_names=self.cls_names)})
        self.val_domain_preds.clear()
        self.val_domain_labels.clear()

    # テストステップ
    def test_step(self, batch, batch_idx=None):
        img = batch['img']
        labels = batch['lab'] # PACSの7クラス分類ラベル (0~6)

        # ドメイン情報を使わず、画像からタスク予測の logits のみを取得
        logits, dom_logits, h_emb, h_out = self(img)

        # 1. 画像分類タスクの Loss と Accuracy を計算 (0~6 のラベル)
        pacs_loss = self.criterion(logits, labels)
        self.log('test_loss', pacs_loss, prog_bar=True)

        acc = accuracy(logits, labels, task="multiclass", num_classes=7)
        self.log('test_acc', acc, prog_bar=True)

        # テスト時は target_domain のラベルが -1 になるため、
        # dom_logits 側の CrossEntropyLoss や混同行列処理は行わない（クラッシュ防止）

    # テストエポック終了時処理
    def on_test_epoch_end(self):
        # メモリ解放・クリーンアップのみ
        self.test_domain_preds.clear()
        self.test_domain_labels.clear()