"""
使い方（例）

# 5-fold全部を ResNet50 と VGG16 で回す（splitは元画像txt、学習はtilesを使用）
python "C:\\pythonのコード\\Deep_learning\\cross_validation_CNN_multi_csv.py" ^
  --tiles_root "C:\\fungi_dataset\\CLAHE2.0_4split"
  --split_root "D:\\fungi\\k-fold\\fold5x3\\seed_000" ^
  --models vit_l_32,swin_v2_b,maxvit_t
  --all_folds ^
  --epochs 20 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 ^
  --pretrained --fine_tune ^
  --sil_every 20 ^
  --save_dir "D:\\Study\\crossvalidation_result4\\train_rate_change" 
  --num_workers 0

# fold 4 だけ回す
python "C:\\pythonのコード\\Deep_learning\\cross_validation_CNN_multi_csv_sche.py"^
  --tiles_root "C:\\fungi_dataset\\CLAHE2.0_4split" ^
  --split_root "D:\\fungi\\k-fold\\fold5x3\\seed_000" ^
  --models mobilenet_v3_large,vgg16_bn,vgg19_bn ^mobilenet_v2,
  --fold 4 ^
  --epochs 20 --batch_size 32 --lr 1e-3 --weight_decay 1e-4 ^0.05
  --pretrained --fine_tune ^
  --sil_every 20 ^
  --save_dir "D:\\Study\\crossvalidation_result4\\train_rate_change\\rennet50_she" ^
  --num_workers 0

#eraly stopping
python "C:\\pythonのコード\\Deep_learning\\cross_validation_CNN_multi_csv_sche_earst.py" \
  --tiles_root "C:\\fungi_dataset\\CLAHE2.0_4split" \
  --split_root "D:\\fungi\\k-fold\\fold5x3\\seed_000" \
  --models maxvit_t \
  --epochs 50 \
  --early_stopping_patience 5 \
  --early_stopping_min_delta 0.002 \
  --all_folds \
  --pretrained \
  --save_dir "D:\\Study\\crossvalidation_result4\\train_rate_change"
  
  --fine_tune \
"resnet50",
"resnet101",
"vgg16",
"vgg16_bn",
"vgg19_bn",
"densenet121",
"densenet169",
"mobilenet_v2",        # ★ 追加
"mobilenet_v3_large",
"efficientnet_b0",
"efficientnet_v2_m",   # ★ 追加
"convnext_base",
"vit_b_32",
"vit_l_32",
"swin_v2_b",
"maxvit_t"

注意
- split_root の txt は「分割前（元画像）の相対パス」例: Alternaria/10.png
- tiles_root は「224x224のタイル画像」例: tiles_root/Alternaria/10_r0_c0.png
- splitの各行に対応するタイルが存在しない場合はスキップします（件数をログ表示）
- それでも train/val/test のどれかが空になったら停止します（学習不能）
"""

import os
import re
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score, silhouette_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from torch.optim.lr_scheduler import CosineAnnealingLR 
import time  # ★ 追加



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

#学習率の変更
def get_model_lr(base_lr: float, model_name: str) -> float:
    """
    モデル系列ごとに学習率をスケーリングするためのヘルパー。

    base_lr : --lr で指定した値（例: 1e-3）
    model_name : "resnet50", "vgg16", "densenet121" など

    例:
      base_lr=1e-3 のとき
        resnet50    -> 1e-3
        vgg16       -> 1e-4
        densenet121 -> 5e-4
    """
    name = model_name.lower()

    # ResNet 系: 基準そのまま
    if name.startswith("resnet"):
        return base_lr

    # VGG 系: 勾配が暴れやすいので 1/10
    if name.startswith("vgg"):
        return base_lr * 0.1

    # DenseNet 系: すこしだけ控えめ (1/2)
    if name.startswith("densenet"):
        return base_lr * 0.5

    # ConvNeXt 系: ResNet の 1/10 くらいが無難
    if name.startswith("convnext"):
        return base_lr * 0.1

    # MobileNet 系
    if name.startswith("mobilenet_v2"):
        return base_lr * 0.5      # 基準の半分 (例: 1e-3 -> 5e-4)

    if name.startswith("mobilenet_v3_large"):
        return base_lr * 0.3      # 1e-3 -> 3e-4

    # EfficientNet 系（旧）
    if name.startswith("efficientnet_b0"):
        return base_lr * 0.2      # 1e-3 -> 2e-4

    # EfficientNet V2 系
    if name.startswith("efficientnet_v2_m"):
        return base_lr * 0.1      # 1e-3 -> 1e-4

    # ★ ViT 系: 少し控えめに
    if name.startswith("vit_"):
        return base_lr * 0.3

    # ★ Swin 系
    if name.startswith("swin_v2"):
        return base_lr * 0.1

    # ★ MaxViT 系
    if name.startswith("maxvit"):
        return base_lr * 0.1
    # その他（mobilenet, efficientnet など）はひとまず基準そのまま
    return base_lr

# ---------------- Utility ----------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 再現性優先（速度は少し落ちる）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    # DataLoader worker の乱数も固定
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_json_dump(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def class_name_from_rel(rel_path: str) -> str:
    # "Alternaria/10.png" -> "Alternaria"
    return os.path.normpath(rel_path).split(os.sep)[0]


def build_class_to_idx_from_split_lists(lists: List[List[str]]) -> Dict[str, int]:
    # trainだけで作ると事故る可能性があるので、train/val/test全てからクラス集合を作る
    classes = sorted({class_name_from_rel(p) for lst in lists for p in lst})
    return {c: i for i, c in enumerate(classes)}


def parse_models_arg(s: str) -> List[str]:
    # "resnet50,vgg16" -> ["resnet50","vgg16"]
    items = [x.strip() for x in s.split(",") if x.strip()]
    if not items:
        raise ValueError("--models is empty.")
    return items


# ---------------- Tiles expansion (raw split -> tiles list) ----------------
_BASE_ID_RE = re.compile(r"^(\d+)")  # "10.png" -> "10", "10_r0_c0.png" -> "10"

def base_id_from_rel_raw(rel_path: str) -> Optional[str]:
    # "Alternaria/10.png" -> "10"
    name = Path(rel_path).name
    stem = Path(name).stem  # "10"
    m = _BASE_ID_RE.match(stem)
    if m:
        return m.group(1)
    # fallback: if something like "img10.png" then no match -> None
    return None


def list_tiles_for_base_id(tiles_root: str, class_name: str, base_id: str) -> List[str]:
    """
    returns rel paths under tiles_root, e.g.
    ["Alternaria/10_r0_c0.png", "Alternaria/10_r0_c1.png", ...]
    """
    cls_dir = Path(tiles_root) / class_name
    if not cls_dir.exists():
        return []

    # まず png/jpg/jpeg を全部拾えるようにする
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
    """
    raw_rel_list: ["Alternaria/10.png", ...]
    return: tiles_rel_list: ["Alternaria/10_r0_c0.png", ...]
    also returns stats: {"raw_total":..., "raw_skipped_no_id":..., "raw_skipped_no_tiles":..., "tiles_total":...}
    """
    stats = {"raw_total": 0, "raw_skipped_no_id": 0, "raw_skipped_no_tiles": 0, "tiles_total": 0}

    tiles_rel = []
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
# ★ 224×224 前提なので Crop / Resize は完全に除去
def build_transforms(aug: str):
    """
    CV_Resnet50_2 と揃える：
    - 入力はすべて 224×224 に揃っている前提
    - デフォルトでは train/val/test すべて同じ Transform（Aug なし）
    - ToTensor + Normalize のみ
    """

    # 評価用（val/test）
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if aug == "strong":
        # Augmentation を使いたい場合のみオンにする
        tf_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02
                )
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        # デフォルトは Resnet50_2 と同じ：Aug 無し
        tf_train = tf_eval

    return tf_train, tf_eval



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
            abs_path = os.path.join(base_dir, rp)
            if not os.path.exists(abs_path):
                missing += 1
                continue

            cls = class_name_from_rel(rp)
            if cls not in class_to_idx:
                # class_to_idx にいないものは除外（通常は起きないはず）
                missing += 1
                continue

            items.append((abs_path, class_to_idx[cls], rp))

        self.items = items
        self.missing = missing

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y, rel = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y, rel


# ---------------- Models ----------------
# ---------------- Models ----------------
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

                # ViT 系
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

        # ---- MobileNet 系 ----
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

        # ---- EfficientNet 系 ----
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

        # ---- ConvNeXt 系 ----
        elif model_name == "convnext_base":
            base = models.convnext_base(weights=weights) if weights is not None else models.convnext_base(pretrained=pretrained)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "convnext"

        # ---- ViT 系 ----
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

        # ---- Swin V2 系 ----
        elif model_name == "swin_v2_b":
            base = models.swin_v2_b(weights=weights) if weights is not None else models.swin_v2_b(pretrained=pretrained)
            in_features = base.head.in_features
            base.head = nn.Linear(in_features, num_classes)
            self.base = base
            self.feat_dim = in_features
            self._type = "swin"

        # ---- MaxViT 系 ----
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

            # head だけ学習可能に戻す
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

    def forward(self, x):
        # 既存 CNN 群
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

        # ViT / Swin / MaxViT はひとまず logits をそのまま特徴として使う
        if self._type in ("vit", "swin", "maxvit"):
            b = self.base
            logits = b(x)      # (B, num_classes)
            feat = logits      # Silhouette 等ではロジット空間をそのまま使う
            return logits, feat

        raise RuntimeError("Unknown model type")



# ---------------- Train / Eval ----------------
@torch.no_grad()
def evaluate(model, loader, criterion, device, compute_silhouette=False):
    model.eval()
    preds, labels, feats_all = [], [], []

    total_loss = 0.0
    total_samples = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits, feat = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        preds.append(torch.argmax(logits, 1).cpu().numpy())
        labels.append(y.cpu().numpy())
        if compute_silhouette:
            feats_all.append(feat.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    sil = np.nan
    if compute_silhouette and len(np.unique(y_true)) >= 2:
        sil = silhouette_score(np.concatenate(feats_all), y_true)

    avg_loss = total_loss / total_samples
    return acc, f1, sil, y_true, y_pred, avg_loss



def train_one_epoch(model, loader, criterion, opt, device, use_amp, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()

    avg_loss = total_loss / total_samples
    train_acc = total_correct / total_samples
    return avg_loss, train_acc



# ---------------- Data loader builder ----------------
def build_loaders_tiles_from_raw_split(
    tiles_root: str,
    split_root: str,
    fold: int,
    batch_size: int,
    num_workers: int,
    aug: str,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int, int, int, Dict]:
    """
    split_root 内: fold{fold}_train/val/test.txt は「元画像相対パス」
    tiles_root 内: 「タイル画像」
    """
    raw_tr = read_lines(os.path.join(split_root, f"fold{fold}_train.txt"))
    raw_va = read_lines(os.path.join(split_root, f"fold{fold}_val.txt"))
    raw_te = read_lines(os.path.join(split_root, f"fold{fold}_test.txt"))

    # class_to_idx は raw split のクラス集合から作る
    class_to_idx = build_class_to_idx_from_split_lists([raw_tr, raw_va, raw_te])
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # ★ 追加
    # raw -> tiles 展開
    tr, st_tr = expand_raw_split_to_tiles(raw_tr, tiles_root)
    va, st_va = expand_raw_split_to_tiles(raw_va, tiles_root)
    te, st_te = expand_raw_split_to_tiles(raw_te, tiles_root)

    tf_train, tf_eval = build_transforms(aug=aug)

    ds_train = ListDataset(tiles_root, tr, class_to_idx, tf_train)
    ds_val   = ListDataset(tiles_root, va, class_to_idx, tf_eval)
    ds_test  = ListDataset(tiles_root, te, class_to_idx, tf_eval)

    # ここで dataset が空になると学習不能なのでガード
    if len(ds_train) == 0 or len(ds_val) == 0 or len(ds_test) == 0:
        raise RuntimeError(
            "Empty dataset after filtering/expansion. "
            f"train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)} "
            f"(raw: tr={len(raw_tr)}, va={len(raw_va)}, te={len(raw_te)})"
        )

    g = torch.Generator()
    g.manual_seed(seed)

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )

    stats = {
        "raw_counts": {"train": len(raw_tr), "val": len(raw_va), "test": len(raw_te)},
        "tile_counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "expand_stats": {"train": st_tr, "val": st_va, "test": st_te},
        "missing_files_filtered": {
            "train": ds_train.missing,
            "val": ds_val.missing,
            "test": ds_test.missing,
        },
        "class_to_idx": class_to_idx,   # ★ 追加
        "idx_to_class": idx_to_class,   # ★ 追加
    }


    return dl_train, dl_val, dl_test, len(class_to_idx), len(ds_train), len(ds_val), len(ds_test), stats,class_to_idx,

#学習履歴から曲線PNGを吐くヘルパー関数
def plot_training_curves(hist_df: pd.DataFrame, out_dir: str):
    epochs = hist_df["epoch"].values

    # Loss 曲線
    plt.figure()
    plt.plot(epochs, hist_df["train_loss"].values, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "curve_loss.png"), bbox_inches="tight")
    plt.close()

    # Accuracy 曲線
    if "val_acc" in hist_df.columns:
        plt.figure()
        plt.plot(epochs, hist_df["val_acc"].values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "curve_accuracy.png"), bbox_inches="tight")
        plt.close()

    # Macro-F1 曲線
    if "val_macro_f1" in hist_df.columns:
        plt.figure()
        plt.plot(epochs, hist_df["val_macro_f1"].values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Macro-F1")
        plt.title("Validation Macro-F1")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "curve_macro_f1.png"), bbox_inches="tight")
        plt.close()

# ---------------- Fold runner ----------------
def run_one_fold_for_model(args, model_name: str, fold: int, device) -> Dict:
    # 出力ディレクトリ：save_dir/{model_name}/seed_xxx/foldY/
    out_dir = os.path.join(args.save_dir, model_name, f"seed_{args.seed:03d}", f"fold{fold}")
    ensure_dir(out_dir)

    # loaders（raw split -> tiles）
    dl_tr, dl_va, dl_te, num_classes, ntr, nva, nte, stats, class_to_idx = build_loaders_tiles_from_raw_split(
        tiles_root=args.tiles_root,
        split_root=args.split_root,
        fold=fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        aug=args.aug,
        seed=args.seed,
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # ★ 追加
    class_ids = list(range(num_classes))                    # ★ よく使うのでここで作成
    class_names = [idx_to_class[i] for i in class_ids]      # ★ 実クラス名のリスト

    print(f"[INFO] model={model_name} fold={fold} train={ntr} val={nva} test={nte} classes={num_classes}")
    safe_json_dump(vars(args), os.path.join(out_dir, "args.json"))
    safe_json_dump(stats, os.path.join(out_dir, "split_expand_stats.json"))

    # model
    model = CNNClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        fine_tune=args.fine_tune
    ).to(device)

    # このモデル用の学習率を決定
    lr_this = get_model_lr(args.lr, model_name)

    # optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr_this,
        weight_decay=args.weight_decay
    )

    # ★ CosineAnnealingLR を追加
    scheduler = CosineAnnealingLR(
        opt,
        T_max=args.epochs,           # 総エポック数
      eta_min=lr_this * 0.01       # 最小学習率（例: 1e-3 → 1e-5）
    )

    criterion = nn.CrossEntropyLoss()

    use_amp = (not args.no_amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = []
    best_val_f1 = -1.0
    best_epoch = -1

    # ★ Early Stopping 用の設定
    patience = getattr(args, "early_stopping_patience", 0)
    min_delta = getattr(args, "early_stopping_min_delta", 0.0)
    no_improve_epochs = 0  # 改善しなかった epoch 数
    
    # 追加：バリデーションの epoch ごとの混同行列 / クラス別スコアを保存
    confmat_val_epochs = []       # List[np.ndarray (num_classes x num_classes)]
    per_class_val_epochs = []     # List[dict(precision/recall/f1/support per class)]

    # ★ resnet50_revise3 互換の train_log.csv 用設定
    log_headers = [
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "elapsed_sec",
    ]
    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(",".join(log_headers) + "\n")
    for ep in range(1, args.epochs + 1):
        ep_start = time.time()

        # train
        train_loss, train_acc = train_one_epoch(
            model, dl_tr, criterion, opt, device, use_amp, scaler
        )

        # val
        do_sil = (args.sil_every > 0) and (ep % args.sil_every == 0)
        val_acc, val_f1, val_sil, y_true_val, y_pred_val, val_loss = evaluate(
            model, dl_va, criterion, device, compute_silhouette=do_sil
        )

        # ★ epoch の経過時間
        elapsed = time.time() - ep_start

        # ここで Val の混同行列 & クラス別指標を保存
        cm_val = confusion_matrix(y_true_val, y_pred_val)
        confmat_val_epochs.append(cm_val)

        prec_val, rec_val, f1_val_arr, sup_val = precision_recall_fscore_support(
            y_true_val,
            y_pred_val,
            labels=list(range(num_classes)),
            zero_division=0,
        )
        per_class_val_epochs.append(
            {
                "precision": prec_val.tolist(),
                "recall": rec_val.tolist(),
                "f1": f1_val_arr.tolist(),
                "support": sup_val.tolist(),
            }
        )

        # 既存の history 用（後で training_history.csv / curve_*.png に使用）
        row = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_acc": float(val_acc),
            "val_macro_f1": float(val_f1),
            "val_silhouette": float(val_sil)
            if do_sil and not np.isnan(val_sil)
            else None,
        }
        history.append(row)

        # ★ このエポックで実際に使われた学習率を取得
        current_lr = opt.param_groups[0]["lr"]

        log_row = {
        "epoch": ep,
        "lr": float(current_lr),          # ← lr_this ではなく current_lr を記録
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_f1),
        "elapsed_sec": float(elapsed),
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(",".join(str(log_row[h]) for h in log_headers) + "\n")

        # ★ epoch ごとに scheduler を1ステップ進める
        scheduler.step()

        # ★epochごとにログ出力
        if do_sil and row["val_silhouette"] is not None:
            print(
            f"[{model_name}][fold{fold}][Epoch {ep:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macroF1={val_f1:.4f} val_sil={val_sil:.4f}"
            )
        else:
            print(
            f"[{model_name}][fold{fold}][Epoch {ep:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_macroF1={val_f1:.4f}"
            )

        # ★ best model は val_macro_f1 で選ぶ（Early Stopping 用カウンタもここで更新）
        improved = val_f1 > (best_val_f1 + min_delta)

        if improved:
            best_val_f1 = float(val_f1)
            best_epoch = ep
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            no_improve_epochs = 0  # 改善があったのでリセット
        else:
            no_improve_epochs += 1

        # ★ Early Stopping 判定（patience <= 0 なら無効）
        if patience > 0 and no_improve_epochs >= patience:
            print(
            f"[{model_name}][fold{fold}] Early stopping at epoch {ep} "
            f"(best_epoch={best_epoch}, best_val_macro_f1={best_val_f1:.4f})"
            )
            break

    # epochループが終わった後に保存（testは最後に一回だけ）
    safe_json_dump(history, os.path.join(out_dir, "training_history.json"))

    # 追加：学習履歴を CSV でも保存
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    # 追加：Loss / Acc / Macro-F1 の曲線を PNG で保存
    plot_training_curves(hist_df, out_dir)

    # bestモデルでtest評価（研究として一番筋がいい）
    best_path = os.path.join(out_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_path, map_location=device))

    test_acc, test_f1, test_sil, y_true, y_pred, test_loss = evaluate(model, dl_te, criterion, device, compute_silhouette=True)

    cm = confusion_matrix(y_true, y_pred)
    safe_json_dump(cm.tolist(), os.path.join(out_dir, "confmat_test.json"))

    # class_ids / class_names は run_one_fold_for_model の冒頭で定義済み
    df_cm_test = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm_test.index.name = "true_class"
    df_cm_test.columns.name = "pred_class"
    df_cm_test.to_csv(os.path.join(out_dir, "confmat_test.csv"))


    # 追加：per_class_test.csv（per-class precision/recall/F1/support）
    prec_t, rec_t, f1_t, sup_t = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0,
    )
    df_per_class_test = pd.DataFrame({
        "class_id": class_ids,
        "class_name": class_names,  # ★ 実クラス名
        "precision": prec_t,
        "recall": rec_t,
        "f1": f1_t,
        "support": sup_t,
    })
    df_per_class_test.to_csv(os.path.join(out_dir, "per_class_test.csv"), index=False)


    # 追加：confmat_val_epoch.csv
    rows_cm_val = []
    for ep_idx, cm_val in enumerate(confmat_val_epochs, start=1):
        cm_val = np.asarray(cm_val)
        for i in range(num_classes):
            for j in range(num_classes):
                rows_cm_val.append({
                    "epoch": ep_idx,
                    "true_class_id": i,
                    "true_class_name": idx_to_class[i],   # ★ 実クラス名
                    "pred_class_id": j,
                    "pred_class_name": idx_to_class[j],   # ★ 実クラス名
                    "count": int(cm_val[i, j]),
                })
    df_cm_val = pd.DataFrame(rows_cm_val)
    df_cm_val.to_csv(os.path.join(out_dir, "confmat_val_epoch.csv"), index=False)


    # 追加：per_class_val_epoch.csv
    rows_pc_val = []
    for ep_idx, d in enumerate(per_class_val_epochs, start=1):
        prec_e = d["precision"]
        rec_e = d["recall"]
        f1_e = d["f1"]
        sup_e = d["support"]
        for cls_id in class_ids:
            rows_pc_val.append({
                "epoch": ep_idx,
                "class_id": cls_id,
                "class_name": idx_to_class[cls_id],   # ★ 実クラス名
                "precision": prec_e[cls_id],
                "recall": rec_e[cls_id],
                "f1": f1_e[cls_id],
                "support": sup_e[cls_id],
            })

    df_pc_val = pd.DataFrame(rows_pc_val)
    df_pc_val.to_csv(os.path.join(out_dir, "per_class_val_epoch.csv"), index=False)


    result = {
        "model": model_name,
        "seed": args.seed,
        "fold": fold,
        "val_best_epoch": int(best_epoch),
        "val_best_macro_f1": float(best_val_f1),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_silhouette": float(test_sil) if not np.isnan(test_sil) else None,
        "sizes": {"train": int(ntr), "val": int(nva), "test": int(nte)},
        "best_model_path": best_path,
    }
    safe_json_dump(result, os.path.join(out_dir, "result.json"))
    return result


def main():
    parser = argparse.ArgumentParser()

    # data / split
    parser.add_argument("--tiles_root", type=str, required=True,
                        help="224x224タイル画像のroot。例: tiles_root/Alternaria/10_r0_c0.png")
    parser.add_argument("--split_root", type=str, required=True,
                        help="fold{K}_train/val/test.txt があるディレクトリ（中身は元画像相対パス）")

    # fold control
    parser.add_argument("--fold", type=int, default=None, help="単一foldだけ回す (0-4)")
    parser.add_argument("--all_folds", action="store_true", help="fold0-4全部回す")

    # models
    parser.add_argument("--models", type=str, default="resnet50",
                        help="CNNモデル名をカンマ区切りで指定。例: resnet50,vgg16,densenet121")

    # train params
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--fine_tune", action="store_true", help="True: 全層学習 / False: headのみ学習")
    parser.add_argument("--aug", type=str, default="none", choices=["none", "strong"],
                    help="data augmentation setting")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--sil_every", type=int, default=50,
                        help="how often (in epochs) to compute silhouette score")
    
    # ★ Early Stopping 用
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Val Macro-F1 が改善しないまま何epoch続いたら打ち切るか (0 なら無効)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.002,
                        help="Val Macro-F1 の「改善」とみなす最小差分")
    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True, help="結果保存root（モデル別にサブフォルダ作成）")

    args = parser.parse_args()

    if (args.fold is None) and (not args.all_folds):
        raise ValueError("Either --fold or --all_folds must be specified.")

    # 対応モデルの最低限チェック
    models_list = parse_models_arg(args.models)
    supported = {
    "resnet50",
    "resnet101",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "densenet121",
    "densenet169",
    "mobilenet_v2",        # ★ 追加
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_v2_m",   # ★ 追加
    "convnext_base",
    "vit_b_32",
    "vit_l_32",
    "swin_v2_b",
    "maxvit_t"
    }
    for m in models_list:
        if m not in supported:
            raise ValueError(f"Unsupported model: {m}. Supported: {sorted(supported)}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    folds = list(range(5)) if args.all_folds else [int(args.fold)]

    # まとめ出力（save_dir/summary.json）
    all_results = []

    for model_name in models_list:
        for fold in folds:
            res = run_one_fold_for_model(args, model_name=model_name, fold=fold, device=device)
            all_results.append(res)

    ensure_dir(args.save_dir)
    safe_json_dump(all_results, os.path.join(args.save_dir, "summary_all_results.json"))
    print(f"[INFO] Saved summary: {os.path.join(args.save_dir, 'summary_all_results.json')}")


if __name__ == "__main__":
    main()
