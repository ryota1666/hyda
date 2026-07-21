import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

# PACSデータセットのドメイン定義
DOMAIN_ENUM = {
    'art_painting': 0,
    'cartoon': 1,
    'photo': 2,
    'sketch': 3
}
DOMAINS = list(DOMAIN_ENUM.keys())

class PACSDataset(Dataset):
    """PACSの特定のドメイン（Art, Cartoonなど）のデータを読み込むデータセットクラス"""
    def __init__(self, root_dir, domain_name, transform=None):
        super().__init__()
        self.domain_name = domain_name
        self.domain_id = DOMAIN_ENUM[domain_name]
        self.transform = transform
        
        # ドメインごとのフォルダ（例: data_dir/art_painting）
        self.domain_dir = os.path.join(root_dir, domain_name)
        
        if not os.path.exists(self.domain_dir):
            raise FileNotFoundError(f"Directory not found: {self.domain_dir}")
            
        # ImageFolderを使って画像とラベル（クラスインデックス）のリストを自動取得
        self.base_dataset = ImageFolder(root=self.domain_dir)
        
    @property
    def labels(self):
        # 元のコードの `task_weights`（重み計算）が参照できるようにラベルのリストを返す
        return [label for _, label in self.base_dataset.samples]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 画像パスと正解ラベルの取得
        img_path, label = self.base_dataset.samples[idx]
        
        # 画像の読み込み（RGB 3チャンネルとして強制読み込み）
        img = Image.open(img_path).convert('RGB')
        
        # 前処理・データオーグメンテーションの適用
        if self.transform is not None:
            img = self.transform(img)
            
        # 元のコードのデータ形式（辞書型）に一致させる
        sample = {
            'idx': idx,
            'img': img,
            'lab': label,
            'domain': self.domain_id
        }
        return sample