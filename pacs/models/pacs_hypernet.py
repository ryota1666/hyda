import torch
from torch import nn
import torch.nn.functional as F

from hyda.layers import HyperLinearBMM
from .pacs_resenet import PACSResNet18

class HyResNet18(PACSResNet18):
    def __init__(self,
                 #domain branch params
                 domain_encoder: nn.Module,
                 domain_classifier: nn.Module,

                 imagenet_pretrained=True,
                 num_classes=7,

                 # Hypernet params
                 hyper_in_size=64,
                 hyper_emb=0,
                 init_method=None,
                 input_var=None,
                 ):
        super().__init__(imagenet_pretrained=imagenet_pretrained, num_classes=num_classes)
        self.num_classes = num_classes
        self.domain_encoder = domain_encoder
        self.domain_classifier = domain_classifier
        self.hyper_emb = hyper_emb
        self.hyper_in_size = hyper_in_size
        
        if self.hyper_emb > 0:
            self.hyper_encode = nn.Sequential(
                nn.Linear(self.hyper_in_size, self.hyper_emb),
                nn.ReLU(),
            )
        else:
            self.hyper_encode = nn.Identity()
            self.hyper_emb = hyper_in_size

        # 変更: DenseNetの 1024次元 から ResNet18 の 512次元 に変更します
        # また、ResNet18 の分類層は `self.model.fc` なので、そこを HyperLinearBMM で上書きします
        self.model.fc = HyperLinearBMM(in_features=512, out_features=num_classes, hyper_size=32,
                                       init_method=init_method, input_var=input_var)

    # pacs_hypernet.py 内の forward メソッド

    def forward(self, x):
        # 1. ドメイン特徴量の抽出と加工
        dom_feats = self.domain_encoder(x)

        if dom_feats.dim() == 4:
            dom_feats_pooled = torch.nn.functional.adaptive_avg_pool2d(
                dom_feats, (1, 1)
            )
            dom_feats_flat = torch.flatten(dom_feats_pooled, 1)
        else:
            dom_feats_flat = dom_feats

        dom_logits = self.domain_classifier(dom_feats)
        h_in = self.hyper_encode(dom_feats_flat)

        # 2. ResNet18 のバックボーンから主タスクの特徴量を抽出
        # (self.model.features(x) の代わりに ResNet のレイヤーを順番に通します)
        path_feats = self.model.conv1(x)
        path_feats = self.model.bn1(path_feats)
        path_feats = self.model.relu(path_feats)
        path_feats = self.model.maxpool(path_feats)

        path_feats = self.model.layer1(path_feats)
        path_feats = self.model.layer2(path_feats)
        path_feats = self.model.layer3(path_feats)
        path_feats = self.model.layer4(path_feats)

        path_feats = self.model.avgpool(path_feats)
        path_feats = torch.flatten(path_feats, 1)

        # 3. ハイパーネットワークが制御する全結合層（fc）に特徴量を渡す
        logits, h_out = self.model.fc(path_feats, h_in)

        return logits, dom_logits, h_in, h_out