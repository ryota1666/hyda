import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class PACSResNet18(nn.Module):
    # 引数に imagenet_pretrained=True を追加します
    def __init__(self, num_classes=7, imagenet_pretrained=True): 
        super().__init__()
        self.num_classes = num_classes
        self.imagenet_pretrained = imagenet_pretrained
        
        # フラグによって事前学習済み重みを使うか、ランダム初期化にするかを制御
        if imagenet_pretrained:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.model = torchvision.models.resnet18(weights=weights)
        
        # ResNet18の最終全結合層（fc）の入力次元は512です。
        # ここをPACSのクラス数（7）に変更します。
        self.model.fc = nn.Linear(512, num_classes)

    def get_features(self, x):
        # 最終全結合層の手前までの特徴量（512次元）を抽出して返す共通メソッド
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        out = torch.flatten(x, 1)
        return out

    def forward(self, x):
        # get_featuresで特徴量を抽出し、分類層（fc）に通します
        features = self.get_features(x)
        out = self.model.fc(features)
        return out

    def classifier(self, x):
        # 抽出した特徴量を入力してクラススコアを出力します
        return self.model.fc(x)