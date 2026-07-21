import os
import numpy as np
import torch
import lightning as pl
import torchvision.transforms as tforms
from typing import Optional
from torch.utils.data import DataLoader, ConcatDataset, Subset

from .dataset import PACSDataset, DOMAIN_ENUM

class PACSDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 target_domain: Optional[str] = None,
                 batch_size: int = 128,
                 num_workers: int = 4,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.target_domain = target_domain
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def task_weights(self):
        # クラスごとの不均衡を補正するための重み計算ロジック
        self.setup()
        train_dl = self.train_dataloader()
        
        # Subset の場合は .dataset.labels からインデックスを考慮して抽出
        if isinstance(train_dl.dataset, Subset):
            full_labels = np.array(train_dl.dataset.dataset.labels)
            train_labels = full_labels[train_dl.dataset.indices]
        else:
            train_labels = np.array(train_dl.dataset.labels)
        
        # PACSはマルチクラス（単一ラベル）なので、One-hot形式に変換してカウント
        num_classes = 7
        labels_onehot = np.eye(num_classes)[train_labels]
        
        weights = np.sum(labels_onehot, axis=0)
        weights = weights.max() - weights + weights.mean()
        weights = weights / weights.max()
        return weights

    def setup(self, stage=None):
        # 基本的な前処理（224x224にリサイズしてTensor化、一般的なImageNetの標準化）
        transform_base = [
            tforms.Resize((224, 224)),
            tforms.ToTensor(),
            tforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # 訓練用のデータオーグメンテーション
        transform_train = tforms.Compose([
            tforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            tforms.RandomHorizontalFlip(p=0.5),
            tforms.RandomRotation(degrees=15),
            tforms.ToTensor(),
            tforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_val = tforms.Compose(transform_base)

        # 全4ドメインのデータセットを定義
        datasets_train = {
            'art_painting': PACSDataset(self.data_dir, 'art_painting', transform=transform_train),
            'cartoon': PACSDataset(self.data_dir, 'cartoon', transform=transform_train),
            'photo': PACSDataset(self.data_dir, 'photo', transform=transform_train),
            'sketch': PACSDataset(self.data_dir, 'sketch', transform=transform_train)
        }
        
        datasets_val = {
            'art_painting': PACSDataset(self.data_dir, 'art_painting', transform=transform_val),
            'cartoon': PACSDataset(self.data_dir, 'cartoon', transform=transform_val),
            'photo': PACSDataset(self.data_dir, 'photo', transform=transform_val),
            'sketch': PACSDataset(self.data_dir, 'sketch', transform=transform_val)
        }

        # ターゲットドメイン（テスト対象）の選定とマージ
        if self.target_domain is None:
            full_train_dataset = ConcatDataset(list(datasets_train.values()))
            full_val_dataset = ConcatDataset(list(datasets_val.values()))
            self.test_dataset = None
        else:
            if self.target_domain not in DOMAIN_ENUM:
                raise ValueError(f"Target domain {self.target_domain} not supported. Choose from {list(DOMAIN_ENUM.keys())}")
            
            # ターゲットドメイン「以外」をマージ
            train_list = [ds for name, ds in datasets_train.items() if name != self.target_domain]
            val_list = [ds for name, ds in datasets_val.items() if name != self.target_domain]
            
            full_train_dataset = ConcatDataset(train_list)
            full_val_dataset = ConcatDataset(val_list)
            
            # ターゲットドメインそのものをテストに使用
            self.test_dataset = datasets_val[self.target_domain]

        # --- 【修正の核心部分】 90% : 10% のランダム分割処理（再現性シード固定） ---
        total_count = len(full_train_dataset)
        val_count = int(total_count * 0.1)       # 全体の 10%
        train_count = total_count - val_count    # 全体の 90%

        # 毎回異なるランダムな順序を生成する
        indices = torch.randperm(total_count).tolist()

        # 同じインデックス配分を適用することで、前処理（transform）の整合性を保ちつつリークを防ぐ
        self.train_dataset = Subset(full_train_dataset, indices[:train_count])
        self.val_dataset = Subset(full_val_dataset, indices[train_count:])
        # ------------------------------------------------------------------------

        # task_weightsプロパティのために、マージされたデータセット全体のlabels属性を擬似的に持たせる
        all_labels = []
        source_datasets = list(datasets_train.values()) if self.target_domain is None else train_list
        for ds in source_datasets:
            all_labels.extend(ds.labels)
        full_train_dataset.labels = all_labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)