"""
使い方:
pip install -U scikit-learn umap-learn pandas matplotlib pillow torch torchvision

python "C:\\pythonのコード\\Deep_learning\\visualize_tsne_umap_from_best.py" ^
  --tiles_root "C:\\fungi_dataset\\CLAHE2.0_4split" ^
  --split_root "D:\\fungi\\k-fold\\fold5x3\\seed_000" ^
  --save_dir "D:\\Study\\crossvalidation_result4\\train_rate_change\\early_stop_scheguler" ^
  --model_name vit_l_32 ^
  --seed 42 ^
  --fold 0 ^
  --split all ^
  --max_samples 5000

概要:
- 学習コードの保存形式: save_dir/{model_name}/seed_XXX/foldY/best_model.pth を読み込む
- split_root の fold{Y}_train/val/test.txt は「元画像相対パス」(例: Alternaria/10.png)
- tiles_root は「タイル画像」(例: Alternaria/10_r0_c0.png)
- raw split を tiles に展開して DataLoader を構築し、特徴量を抽出
- t-SNE / UMAP を 2D に落として PNG + CSV を出力
"""

import os
import re
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# UMAP は import 名が "umap"（umap-learn）
import umap
import matplotlib.pyplot as plt


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------- Utility ----------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def class_name_from_rel(rel_path: str) -> str:
    # "Alternaria/10.png" -> "Alternaria"
    return os.path.normpath(rel_path).split(os.sep)[0]


def build_class_to_idx_from_split_lists(lists: List[List[str]]) -> Dict[str, int]:
    classes = sorted({class_name_from_rel(p) for lst in lists for p in lst})
    return {c: i for i, c in enumerate(classes)}


# ---------------- Tiles expansion (raw split -> tiles list) ----------------
_BASE_ID_RE = re.compile(r"^(\d+)")  # "10.png" -> "10", "10_r0_c0.png" -> "10"


def base_id_from_rel_raw(rel_path: str) -> Optional[str]:
    # "Alternaria/10.png" -> "10"
    name = Path(rel_path).name
    stem = Path(name).stem
    m = _BASE_ID_RE.match(stem)
    if m:
        return m.group(1)
    return None


def list_tiles_for_base_id(tiles_root: str, class_name: str, base_id: str) -> List[str]:
    """
    returns rel paths under tiles_root, e.g.
    ["Alternaria/10_r0_c0.png", "Alternaria/10_r0_c1.png", ...]
    """
    cls_dir = Path(tiles_root) / class_name
    if not cls_dir.exists():
        return []

    patterns = [
        f"{base_id}_r*_c*.png",
        f"{base_id}_r*_c*.jpg",
        f"{base_id}_r*_c*.jpeg",
    ]
    found = []
    for pat in patterns:
        found.extend(sorted(cls_dir.glob(pat)))

    rels = [str(Path(class_name) / p.name).replace("\\", "/") for p in found]
    return rels


def expand_raw_split_to_tiles(
    raw_rel_list: List[str],
    tiles_root: str,
) -> Tuple[List[str], Dict[str, int]]:
    stats = {
        "raw_total": 0,
        "raw_skipped_no_id": 0,
        "raw_skipped_no_tiles": 0,
        "tiles_total": 0
    }

    tiles_rel: List[str] = []
    stats["raw_total"] = len(raw_rel_list)

    for raw_rel in raw_rel_list:
        cls = class_name_from_rel(raw_rel)
        base_id = base_id_from_rel_raw(raw_rel)
        if base_id is None:
            stats["raw_skipped_no_id"] += 1
            continue

        tile_rels = list_tiles_for_base_id(tiles_root, cls, base_id)
        if len(tile_rels) == 0:
            stats["raw_skipped_no_tiles"] += 1
            continue

        tiles_rel.extend(tile_rels)

    stats["tiles_total"] = len(tiles_rel)
    return tiles_rel, stats


# ---------------- Transforms ----------------
def build_transforms():
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf_eval


# ---------------- Dataset ----------------
class ListDataset(Dataset):
    """
    split txt に書かれた相対パスを base_dir と結合して読み込む。
    base_dir 内に存在しない画像が混ざっていても、無視して続行する。
    """
    def __init__(self, base_dir: str, rel_paths: List[str], class_to_idx: Dict[str, int], transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

        items = []
        missing = 0

        for rp in rel_paths:
            p = os.path.join(base_dir, rp)
            if not os.path.exists(p):
                missing += 1
                continue
            cls = class_name_from_rel(rp)
            if cls not in class_to_idx:
                # 念のため
                continue
            items.append((rp, class_to_idx[cls]))

        self.items = items
        self.missing = missing

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rp, y = self.items[idx]
        path = os.path.join(self.base_dir, rp)

        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, y, rp


# ---------------- Model (学習コードと同一の CNNClassifier) ----------------
class CNNClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool, fine_tune: bool):
        super().__init__()
        self.model_name = model_name

        # torchvision の weights 指定（古い環境でも動くように try/except）
        def _maybe_weights(name: str):
            if not pretrained:
                return None
            try:
                # ResNet
                if name == "resnet50":
                    return models.ResNet50_Weights.DEFAULT
                if name == "resnet101":
                    return models.ResNet101_Weights.DEFAULT

                # VGG
                if name == "vgg16":
                    return models.VGG16_Weights.DEFAULT
                if name == "vgg16_bn":
                    return models.VGG16_BN_Weights.DEFAULT
                if name == "vgg19_bn":
                    return models.VGG19_BN_Weights.DEFAULT

                # DenseNet
                if name == "densenet121":
                    return models.DenseNet121_Weights.DEFAULT
                if name == "densenet169":
                    return models.DenseNet169_Weights.DEFAULT

                # MobileNet / EfficientNet
                if name == "mobilenet_v2":
                    return models.MobileNet_V2_Weights.DEFAULT
                if name == "mobilenet_v3_large":
                    return models.MobileNet_V3_Large_Weights.DEFAULT
                if name == "efficientnet_b0":
                    return models.EfficientNet_B0_Weights.DEFAULT
                if name == "efficientnet_v2_m":
                    return models.EfficientNet_V2_M_Weights.DEFAULT

                # ConvNeXt
                if name == "convnext_base":
                    return models.ConvNeXt_Base_Weights.DEFAULT

                # ViT
                if name == "vit_b_32":
                    return models.ViT_B_32_Weights.DEFAULT
                if name == "vit_l_32":
                    return models.ViT_L_32_Weights.DEFAULT

                # Swin V2
                if name == "swin_v2_b":
                    return models.Swin_V2_B_Weights.DEFAULT

                # MaxViT
                if name == "maxvit_t":
                    return models.MaxVit_T_Weights.DEFAULT

                return None
            except Exception:
                return None

        weights = _maybe_weights(model_name)

        # ---- ResNet 系 ----
        if model_name == "resnet50":
            base = models.resnet50(weights=weights) if weights is not None else models.resnet50(pretrained=pretrained)
            in_features = base.fc.in_features
            base.fc = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "resnet"

        elif model_name == "resnet101":
            base = models.resnet101(weights=weights) if weights is not None else models.resnet101(pretrained=pretrained)
            in_features = base.fc.in_features
            base.fc = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "resnet"

        # ---- VGG 系 ----
        elif model_name == "vgg16":
            base = models.vgg16(weights=weights) if weights is not None else models.vgg16(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "vgg"

        elif model_name == "vgg16_bn":
            base = models.vgg16_bn(weights=weights) if weights is not None else models.vgg16_bn(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "vgg"

        elif model_name == "vgg19_bn":
            base = models.vgg19_bn(weights=weights) if weights is not None else models.vgg19_bn(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "vgg"

        # ---- DenseNet 系 ----
        elif model_name == "densenet121":
            base = models.densenet121(weights=weights) if weights is not None else models.densenet121(pretrained=pretrained)
            in_features = base.classifier.in_features
            base.classifier = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "densenet"

        elif model_name == "densenet169":
            base = models.densenet169(weights=weights) if weights is not None else models.densenet169(pretrained=pretrained)
            in_features = base.classifier.in_features
            base.classifier = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "densenet"

        # ---- MobileNet ----
        elif model_name == "mobilenet_v2":
            base = models.mobilenet_v2(weights=weights) if weights is not None else models.mobilenet_v2(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "mobilenet"

        elif model_name == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(weights=weights) if weights is not None else models.mobilenet_v3_large(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "mobilenet"

        # ---- EfficientNet ----
        elif model_name == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights) if weights is not None else models.efficientnet_b0(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "efficientnet"

        elif model_name == "efficientnet_v2_m":
            base = models.efficientnet_v2_m(weights=weights) if weights is not None else models.efficientnet_v2_m(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "efficientnet"

        # ---- ConvNeXt ----
        elif model_name == "convnext_base":
            base = models.convnext_base(weights=weights) if weights is not None else models.convnext_base(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "convnext"

        # ---- ViT ----
        elif model_name == "vit_b_32":
            base = models.vit_b_32(weights=weights) if weights is not None else models.vit_b_32(pretrained=pretrained)
            in_features = base.heads.head.in_features
            base.heads.head = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "vit"

        elif model_name == "vit_l_32":
            base = models.vit_l_32(weights=weights) if weights is not None else models.vit_l_32(pretrained=pretrained)
            in_features = base.heads.head.in_features
            base.heads.head = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "vit"

        # ---- Swin V2 ----
        elif model_name == "swin_v2_b":
            base = models.swin_v2_b(weights=weights) if weights is not None else models.swin_v2_b(pretrained=pretrained)
            in_features = base.head.in_features
            base.head = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "swin"

        # ---- MaxViT ----
        elif model_name == "maxvit_t":
            base = models.maxvit_t(weights=weights) if weights is not None else models.maxvit_t(pretrained=pretrained)
            if isinstance(base.classifier, nn.Sequential):
                last = base.classifier[-1]
                if isinstance(last, nn.Linear):
                    in_features = last.in_features
                    base.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    raise RuntimeError("Unexpected MaxViT classifier structure")
            else:
                raise RuntimeError("Unexpected MaxViT classifier type")
            self.base = base
            self.feat_dim = in_features
            self._type = "maxvit"

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # ---- fine_tune=False なら分類ヘッド以外を freeze ----
        if not fine_tune:
            for p in self.base.parameters():
                p.requires_grad = False

            if self._type == "resnet":
                for p in self.base.fc.parameters():
                    p.requires_grad = True
            elif self._type == "vgg":
                for p in self.base.classifier[-1].parameters():
                    p.requires_grad = True
            elif self._type == "densenet":
                for p in self.base.classifier.parameters():
                    p.requires_grad = True
            elif self._type in ("mobilenet", "efficientnet", "convnext"):
                for p in self.base.classifier[-1].parameters():
                    p.requires_grad = True
            elif self._type == "vit":
                for p in self.base.heads.parameters():
                    p.requires_grad = True
            elif self._type == "swin":
                for p in self.base.head.parameters():
                    p.requires_grad = True
            elif self._type == "maxvit":
                for p in self.base.classifier.parameters():
                    p.requires_grad = True
    def _forward_vit_pre_head(self, x):
        """
        ViT の可視化特徴(feat)を「MLP head 内の LayerNorm 出力」にする。
        - head 内に LayerNorm が存在する場合：その LayerNorm 直後のテンソルを feat として返す
        - 存在しない場合：従来どおり CLS token（heads 直前）を feat として返す
        戻り値:
        logits: 最終出力
        feat  : 可視化に使う特徴
        """
        b = self.base

        try:
            # 1) 入力をパッチ列に変換 (N, num_patches, hidden_dim)
            x = b._process_input(x)
            n = x.shape[0]

            # 2) CLS token を先頭に付与
            cls_tok = b.class_token.expand(n, -1, -1)  # (N, 1, hidden_dim)
            x = torch.cat([cls_tok, x], dim=1)         # (N, 1+num_patches, hidden_dim)

            # 3) Encoder を通す（pos_embed + transformer blocks + ln）
            enc = b.encoder
            x = x + enc.pos_embedding
            x = enc.dropout(x)
            x = enc.layers(x)
            x = enc.ln(x)

            # 4) CLS token（heads入力のベース）
            cls_feat = x[:, 0]  # (N, hidden_dim)

            # 5) MLP head 内の LayerNorm を探し、あれば「その出力」を feat にする
            feat_for_vis = cls_feat
            h = cls_feat

            found_ln = False
            # heads が Sequential の想定（torchvision ViT）
            for layer in b.heads.children():
                h = layer(h)
                if isinstance(layer, nn.LayerNorm):
                    feat_for_vis = h
                    found_ln = True

            # logits は head の最終出力
            logits = h

            # head に LayerNorm が無い環境では CLS token を可視化特徴にする
            if not found_ln:
                feat_for_vis = cls_feat

            return logits, feat_for_vis

        except Exception:
            # どうしても内部構造が違う場合は落とさず従来挙動
            logits = b(x)
            feat = logits
            return logits, feat


        except Exception:
            # 万一 torchvision の内部構造が違う環境でも、落ちないようにフォールバック
            logits = b(x)
            feat = logits
            return logits, feat

    def forward(self, x):
        if self._type == "resnet":
            b = self.base
            x = b.conv1(x)
            x = b.bn1(x)
            x = b.relu(x)
            x = b.maxpool(x)
            x = b.layer1(x)
            x = b.layer2(x)
            x = b.layer3(x)
            x = b.layer4(x)
            feat = torch.flatten(b.avgpool(x), 1)
            logits = b.fc(feat)
            return logits, feat

        if self._type == "vgg":
            b = self.base
            x = b.features(x)
            x = b.avgpool(x)
            x = torch.flatten(x, 1)
            feat = b.classifier[:-1](x)
            logits = b.classifier[-1](feat)
            return logits, feat

        if self._type == "densenet":
            b = self.base
            feat_map = b.features(x)
            feat_map = nn.functional.relu(feat_map, inplace=True)
            feat = nn.functional.adaptive_avg_pool2d(feat_map, (1, 1)).view(feat_map.size(0), -1)
            logits = b.classifier(feat)
            return logits, feat

        if self._type == "mobilenet":
            b = self.base
            feat = b.features(x)
            feat = b.avgpool(feat)
            feat = torch.flatten(feat, 1)
            mid = b.classifier[:-1](feat)
            logits = b.classifier[-1](mid)
            return logits, mid

        if self._type == "convnext":
            b = self.base
            feat = b.features(x)
            feat = b.avgpool(feat)
            mid = b.classifier[:-1](feat)
            logits = b.classifier[-1](mid)
            return logits, mid

        if self._type == "efficientnet":
            b = self.base
            feat = b.features(x)
            feat = b.avgpool(feat)
            feat = torch.flatten(feat, 1)
            mid = b.classifier[:-1](feat)
            logits = b.classifier[-1](mid)
            return logits, mid

    # ViT は「分類ヘッド直前（CLS token）」を特徴として利用（ResNet-50と同じ立ち位置）
        if self._type == "vit":
            return self._forward_vit_pre_head(x)

        # Swin / MaxViT は従来どおり logits を特徴として利用
        if self._type in ("swin", "maxvit"):
            b = self.base
            logits = b(x)
            feat = logits
            return logits, feat


        raise RuntimeError("Unknown model type")


# ---------------- Feature extraction & sampling ----------------
@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    feats_all = []
    y_all = []
    pred_all = []
    paths_all: List[str] = []

    for x, y, rp in loader:
        x = x.to(device)
        y = y.to(device)

        logits, feat = model(x)
        pred = torch.argmax(logits, dim=1)

        feats_all.append(feat.detach().cpu().numpy())
        y_all.append(y.detach().cpu().numpy())
        pred_all.append(pred.detach().cpu().numpy())
        paths_all.extend(list(rp))

    feats = np.concatenate(feats_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    y_pred = np.concatenate(pred_all, axis=0)
    return feats, y_true, y_pred, paths_all


def subsample_indices(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    n = len(y)
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n)

    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=max_samples, replace=False)
    return np.sort(idx)


# ---------------- Plotting ----------------
def plot_2d_scatter(
    xy: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    title: str,
    out_png: str,
    max_legend_items: int = 30,
):
    plt.figure(figsize=(10, 8))

    labels = labels.astype(int)
    uniq = np.unique(labels)
    num_classes = len(label_names)

    # scatter と legend で「同じ colormap / 同じ正規化」を共有する
    cmap = plt.get_cmap("tab20", num_classes)  # クラス数に合わせて離散化
    norm = plt.Normalize(vmin=0, vmax=max(num_classes - 1, 1))

    plt.scatter(
        xy[:, 0], xy[:, 1],
        c=labels,
        s=6,
        alpha=0.8,
        cmap=cmap,
        norm=norm
    )

    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.grid(True, linewidth=0.3, alpha=0.4)

    if len(uniq) <= max_legend_items:
        handles = []
        for cid in sorted(uniq.tolist()):
            color = cmap(cid)  # scatter と同じ色
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    label=label_names[cid],
                )
            )
        plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tiles_root", type=str, required=True,
                        help="224x224タイル画像のroot。例: tiles_root/Alternaria/10_r0_c0.png")
    parser.add_argument("--split_root", type=str, required=True,
                        help="fold{K}_train/val/test.txt があるディレクトリ（中身は元画像相対パス）")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="学習結果保存root（save_dir/{model}/seed_xxx/foldY/best_model.pth）")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2, 3, 4])

    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "all"],
                        help="可視化に使う split（all は train+val+test を結合）")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="best_model.pth を明示指定したい場合に使用。未指定なら保存規約から自動推定。")

    parser.add_argument("--out_subdir", type=str, default="tsne_umap",
                        help="foldフォルダ配下に作る出力サブフォルダ名")

    parser.add_argument("--max_samples", type=int, default=14893,
                        help="点数が多すぎると重いので、上限を超えたらランダムに間引く（0以下で無効）")

    # t-SNE params
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iters", type=int, default=1000)

    # UMAP params
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)

    args = parser.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # fold dir (学習コード準拠)
    fold_dir = os.path.join(args.save_dir, args.model_name, f"seed_{args.seed:03d}", f"fold{args.fold}")
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"[ERROR] fold_dir not found: {fold_dir}")

    # checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(fold_dir, "best_model.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[ERROR] checkpoint not found: {ckpt_path}")

    out_dir = os.path.join(fold_dir, args.out_subdir)
    ensure_dir(out_dir)

    # --- load splits (raw) ---
    raw_tr = read_lines(os.path.join(args.split_root, f"fold{args.fold}_train.txt"))
    raw_va = read_lines(os.path.join(args.split_root, f"fold{args.fold}_val.txt"))
    raw_te = read_lines(os.path.join(args.split_root, f"fold{args.fold}_test.txt"))

    class_to_idx = build_class_to_idx_from_split_lists([raw_tr, raw_va, raw_te])
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # --- expand raw -> tiles ---
    tiles_tr, st_tr = expand_raw_split_to_tiles(raw_tr, args.tiles_root)
    tiles_va, st_va = expand_raw_split_to_tiles(raw_va, args.tiles_root)
    tiles_te, st_te = expand_raw_split_to_tiles(raw_te, args.tiles_root)

    print(f"[INFO] expand stats train={st_tr} val={st_va} test={st_te}")

    if args.split == "train":
        tiles = tiles_tr
    elif args.split == "val":
        tiles = tiles_va
    elif args.split == "test":
        tiles = tiles_te
    else:
        tiles = tiles_tr + tiles_va + tiles_te

    tf = build_transforms()
    ds = ListDataset(args.tiles_root, tiles, class_to_idx, transform=tf)
    if len(ds) == 0:
        raise RuntimeError("[ERROR] dataset is empty after expansion/filtering.")

    g = torch.Generator()
    g.manual_seed(args.seed)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    # --- model load ---
    num_classes = len(class_to_idx)
    model = CNNClassifier(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=False,   # 重みは ckpt から読むので不要
        fine_tune=True
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"[INFO] loaded checkpoint: {ckpt_path}")

    # --- extract features ---
    feats, y_true, y_pred, paths = extract_features(model, dl, device=device)
    print(f"[INFO] extracted feats: {feats.shape}")

    # --- subsample if needed ---
    idx_keep = subsample_indices(y_true, args.max_samples, seed=args.seed)
    feats = feats[idx_keep]
    y_true = y_true[idx_keep]
    y_pred = y_pred[idx_keep]
    paths = [paths[i] for i in idx_keep.tolist()]
    print(f"[INFO] after subsample: {feats.shape}")

    # --- standardize ---
    scaler = StandardScaler()
    feats_z = scaler.fit_transform(feats)

    # ===================== t-SNE =====================
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_iters,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    )

    xy_tsne = tsne.fit_transform(feats_z)
    tsne_png = os.path.join(out_dir, "tsne.png")
    plot_2d_scatter(
        xy_tsne, y_true, class_names,
        title=f"t-SNE ({args.model_name}) fold{args.fold} split={args.split} n={len(y_true)}",
        out_png=tsne_png
    )
    df_tsne = pd.DataFrame({
        "x": xy_tsne[:, 0],
        "y": xy_tsne[:, 1],
        "true_id": y_true,
        "true_name": [class_names[i] for i in y_true],
        "pred_id": y_pred,
        "pred_name": [class_names[i] for i in y_pred],
        "rel_path": paths,
        "split": args.split,
        "model": args.model_name,
        "fold": args.fold,
        "seed": args.seed,
        "method": "tsne"
    })
    df_tsne.to_csv(os.path.join(out_dir, "embeddings_tsne.csv"), index=False)
    print(f"[INFO] saved: {tsne_png}")
    print(f"[INFO] saved: {os.path.join(out_dir, 'embeddings_tsne.csv')}")

    # ===================== UMAP =====================
    um = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    )
    xy_umap = um.fit_transform(feats_z)
    umap_png = os.path.join(out_dir, "umap.png")
    plot_2d_scatter(
        xy_umap, y_true, class_names,
        title=f"UMAP ({args.model_name}) fold{args.fold} split={args.split} n={len(y_true)}",
        out_png=umap_png
    )
    df_umap = pd.DataFrame({
        "x": xy_umap[:, 0],
        "y": xy_umap[:, 1],
        "true_id": y_true,
        "true_name": [class_names[i] for i in y_true],
        "pred_id": y_pred,
        "pred_name": [class_names[i] for i in y_pred],
        "rel_path": paths,
        "split": args.split,
        "model": args.model_name,
        "fold": args.fold,
        "seed": args.seed,
        "method": "umap"
    })
    df_umap.to_csv(os.path.join(out_dir, "embeddings_umap.csv"), index=False)
    print(f"[INFO] saved: {umap_png}")
    print(f"[INFO] saved: {os.path.join(out_dir, 'embeddings_umap.csv')}")

    print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
