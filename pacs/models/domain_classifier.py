from torch import nn

class NaiveEncoder(nn.Module):
    def __init__(self, activation='ReLU', norm_layer='BatchNorm2d'):
        super().__init__()
        act_layer = getattr(nn, activation)
        norm_layer = getattr(nn, norm_layer)
        self.model = nn.Sequential(
            # block 1
            # 変更: 入力チャンネルを 1 から 3 に修正
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2),
            norm_layer(8),

            # block 2 (変更なし)
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2),
            norm_layer(16),

            # block 3 (変更なし)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2),
            norm_layer(32),

            # block 4 (変更なし)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2),
            norm_layer(64),
        )

    def forward(self, x):
        return self.model(x)

class NaiveClassifier(nn.Module):
    def __init__(self, num_classes=3): # 変更: YAMLから渡されるドメイン分類数を受け取る
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            nn.Linear(64, num_classes)
        )
    
    def  forward(self, x):
        return self.model(x)