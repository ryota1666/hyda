import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torchvision.transforms as tforms

# プロジェクト内のモジュール
from pacs.data.data_module import PACSDataModule, DOMAIN_ENUM
from pacs.pl_modules.hypernet import DomainConditionedPACSLitModule
from pacs.models.domain_classifier import NaiveEncoder, NaiveClassifier

CLASS_NAMES = ['dog', 'elephant', 'giraffe', 'guitar', 'house', 'horse', 'person']
DOMAIN_NAMES = list(DOMAIN_ENUM.keys()) # ['art_painting', 'cartoon', 'photo', 'sketch']

def extract_features(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)

    dom_features_list = []
    prim_features_list = []
    labels_list = []
    domains_list = []
    preds_list = []

    domain_to_idx = {name: i for i, name in enumerate(DOMAIN_NAMES)}

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img'].to(device)
            lab = batch['lab'].to(device)
            
            # ドメインラベルの数値化
            if isinstance(batch['domain'], (list, tuple)):
                dom_ids = [domain_to_idx[name] for name in batch['domain']]
                dom = torch.tensor(dom_ids, dtype=torch.long, device=device)
            else:
                dom = batch['domain'].to(device)

            # ----------------------------------------------------
            # 1. Domain Encoder 特徴量の抽出 (NaiveEncoder)
            # ----------------------------------------------------
            # model.domain_encoder は NaiveEncoder のインスタンス
            dom_feats = model.domain_encoder(img)
            # NaiveEncoder の出力は [B, 64, 14, 14] などのテンソルなので AdaptiveAvgPool でベクトル化
            if dom_feats.dim() == 4:
                dom_feats = torch.nn.functional.adaptive_avg_pool2d(dom_feats, (1, 1))
                dom_feats_flat = torch.flatten(dom_feats, 1)
            else:
                dom_feats_flat = dom_feats

            # ----------------------------------------------------
            # 2. Primary Network 特徴量の抽出 (ResNet18 512次元)
            # ----------------------------------------------------
            # HyResNet18 構造を正確に辿り、ResNet18 本体の layer1~4 を通過させる
            resnet = model.model # PACSResNet18
            
            # もし PACSResNet18 が内部でさらに resnet や model を保持している場合のフォールバック処理
            if hasattr(resnet, 'model'):
                backbone = resnet.model
            else:
                backbone = resnet

            feat = backbone.conv1(img)
            feat = backbone.bn1(feat)
            feat = backbone.relu(feat)
            feat = backbone.maxpool(feat)

            feat = backbone.layer1(feat)
            feat = backbone.layer2(feat)
            feat = backbone.layer3(feat)
            feat = backbone.layer4(feat)

            feat = backbone.avgpool(feat)
            path_feats = torch.flatten(feat, 1) # 🌟 512次元のピュアな物体特徴量ベクトル

            # ----------------------------------------------------
            # 3. 推定クラス (preds) の取得
            # ----------------------------------------------------
            logits, _, _, _ = model(img)
            preds = logits.argmax(dim=1)

            # リストに追加
            dom_features_list.append(dom_feats_flat.cpu().numpy())
            prim_features_list.append(path_feats.cpu().numpy())
            labels_list.append(lab.cpu().numpy())
            domains_list.append(dom.cpu().numpy())
            preds_list.append(preds.cpu().numpy())

    return (
        np.concatenate(dom_features_list, axis=0),
        np.concatenate(prim_features_list, axis=0),
        np.concatenate(labels_list, axis=0),
        np.concatenate(domains_list, axis=0),
        np.concatenate(preds_list, axis=0)
    )

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = "./data/PACS" 
    ckpt_path = os.path.expanduser("~/ckpts/pacs_hyda/pacs_hyda_epoch=058.ckpt") 

    print("データモジュールとモデルを準備中...")
    datamodule = PACSDataModule(data_dir=data_dir, target_domain=None, batch_size=64, num_workers=4)
    datamodule.setup()
    
    # 9991枚（または9988枚）の全データをそのまま読み出す DataLoader
    viz_source_dataset = datamodule.val_dataset.dataset 
    viz_source_loader = DataLoader(
        viz_source_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        drop_last=False # 端数も絶対に切り捨てない
    )

    # ロードに必要なダミー構造を作成
    encoder = NaiveEncoder()
    classifier = NaiveClassifier()

    # チェックポイントからモデルを完全復元
    model = DomainConditionedPACSLitModule.load_from_checkpoint(
        ckpt_path,
        domain_encoder=encoder,
        domain_classifier=classifier,
        map_location=device
    )
    model.eval()
    model.to(device)

    print("全サンプルの特徴量を抽出中...")
    dom_feats, prim_feats, labels, domains, preds = extract_features(model, viz_source_loader, device=device)
    num_samples = len(labels)
    print(f"抽出完了: 全 {num_samples} サンプル")

    # t-SNE 計算
    print("t-SNE 計算中 (Domain Encoder)...")
    tsne_dom = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    dom_2d = tsne_dom.fit_transform(dom_feats)

    print("t-SNE 計算中 (Primary Network)...")
    tsne_prim = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    prim_2d = tsne_prim.fit_transform(prim_feats)

    # 描画と保存
    os.makedirs("./tsne_results", exist_ok=True)
    cmap = plt.get_cmap('tab10')

    # 【1】Domain Encoder Feature Space
    plt.figure(figsize=(9, 8))
    plt.grid(True, linestyle='--', alpha=0.3)
    for i, domain_name in enumerate(DOMAIN_NAMES):
        idx = (domains == i)
        if np.any(idx):
            plt.scatter(dom_2d[idx, 0], dom_2d[idx, 1], label=domain_name, alpha=0.4, s=15, color=cmap(i))
    plt.title(f"1. Domain Encoder Feature Space ($x_D$) [final_frozen_setup]\n[All {num_samples} samples]", fontsize=13)
    plt.legend(loc='upper right', framealpha=0.8)
    plt.tight_layout()
    plt.savefig("./tsne_results/1_domain_encoder_space.png", dpi=300)
    plt.show()
    plt.close()

    # 【2】Primary Network Feature Space (Colored by Domain)
    plt.figure(figsize=(9, 8))
    plt.grid(True, linestyle='--', alpha=0.3)
    for i, domain_name in enumerate(DOMAIN_NAMES):
        idx = (domains == i)
        if np.any(idx):
            plt.scatter(prim_2d[idx, 0], prim_2d[idx, 1], label=domain_name, alpha=0.4, s=15, color=cmap(i))
    plt.title(f"2. Primary Network Feature Space ($\mathcal{{P}}_{{enc}}$) [final_frozen_setup]\n[All {num_samples} samples]", fontsize=13)
    plt.legend(loc='upper right', framealpha=0.8)
    plt.tight_layout()
    plt.savefig("./tsne_results/2_primary_network_domain_space.png", dpi=300)
    plt.show()
    plt.close()

    # 【3】Primary Network Space (左: 正解クラスGT / 右: 予測クラスPredictions)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左: True Classes (GT)
    for i in range(7):
        idx = (labels == i)
        if np.any(idx):
            axes[0].scatter(prim_2d[idx, 0], prim_2d[idx, 1], label=CLASS_NAMES[i], alpha=0.4, s=12, color=cmap(i))
    axes[0].set_title(f"1. Primary Network Space ($\mathcal{{P}}_{{enc}}$) - Colored by GT\n[All {num_samples} samples / True Classes]", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8, framealpha=0.8)

    # 右: Model Outputs (Predictions)
    for i in range(7):
        idx = (preds == i)
        if np.any(idx):
            axes[1].scatter(prim_2d[idx, 0], prim_2d[idx, 1], label=CLASS_NAMES[i], alpha=0.4, s=12, color=cmap(i))
    axes[1].set_title(f"2. Primary Network Space ($\mathcal{{P}}_{{enc}}$) - Colored by Predictions\n[All {num_samples} samples / Model Outputs]", fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=8, framealpha=0.8)

    plt.suptitle("HyDA Primary Network Task-Class Space Verification (PACS)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig("./tsne_results/3_primary_network_class_verification.png", dpi=300)
    plt.show()
    plt.close()

    print("すべてのプロットが出力されました！ (保存先: ./tsne_results/)")

if __name__ == '__main__':
    main()