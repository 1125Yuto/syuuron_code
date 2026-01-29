#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ファイル名: smart_crop_by_gradient_split.py

目的:
  train/val/test/class のフォルダ構造を維持したまま、
  Resize(短辺=resize_short) → 勾配ベースSmartCrop(crop_size) を適用して出力する。

想定入力:
  in_root/
    train/
      classA/*.png ...
      classB/*.png ...
    val/
      classA/*.png ...
      classB/*.png ...
    test/
      classA/*.png ...
      classB/*.png ...

出力:
  out_root/
    train/classA/*.png ...
    val/classB/*.png ...
    test/classA/*.png ...

依存:
  pip install pillow opencv-python numpy

使い方:
  python "C:\\pythonのコード\\画像編集\\smart_crop_by_gradient4.py" ^
    --in_root  "D:\\fungi\\rawdataset_main_folded" ^
    --out_root "D:\\fungi\\smart_crop\\rawdataset_main_folded" ^
    --splits "train,val,test" ^
    --crop_size 224 ^
    --resize_short 256 ^
    --exts "png,jpg,jpeg,tif,tiff,bmp" ^
    --suffix "" ^
    --recursive
"""

import argparse
from pathlib import Path
from typing import Set

import numpy as np
import cv2
from PIL import Image


# ---------------------------
# ユーティリティ
# ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_exts(exts_csv: str) -> Set[str]:
    exts = set()
    for e in exts_csv.split(","):
        e = e.strip().lower().lstrip(".")
        if not e:
            continue
        exts.add("." + e)
    return exts


# ---------------------------
# コア処理：Resize → SmartCrop
# ---------------------------

def resize_keep_aspect(img: Image.Image, short_side: int) -> Image.Image:
    """短辺を short_side に合わせてアスペクト維持リサイズ"""
    w, h = img.size
    short = min(w, h)
    if short <= 0 or short == short_side:
        return img
    scale = short_side / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC)


def smart_crop_by_gradient(img_rgb: Image.Image, crop_size: int, resize_short: int) -> Image.Image:
    """
    1) 短辺をresize_shortにリサイズ（アスペクト維持）
    2) Sobel勾配の局所平均が最大の位置を中心に crop_size×crop_size を切り出す
    """
    img_r = resize_keep_aspect(img_rgb, resize_short)

    gray = np.array(img_r.convert("L"), dtype=np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)

    # crop_size 窓で局所平均（シンプル版）
    local_mean = cv2.boxFilter(grad, ddepth=-1, ksize=(crop_size, crop_size), normalize=True)

    _, _, _, max_loc = cv2.minMaxLoc(local_mean)
    cx, cy = max_loc  # (x, y)

    H, W = gray.shape
    half = crop_size // 2

    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + crop_size
    bottom = top + crop_size

    # 端のはみ出しをクリップ
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > W:
        shift = right - W
        left -= shift
        right = W
        if left < 0:
            left = 0
    if bottom > H:
        shift = bottom - H
        top -= shift
        bottom = H
        if top < 0:
            top = 0

    crop = img_r.crop((left, top, right, bottom))

    # 万一サイズが揃わなければ最後に合わせる
    if crop.size != (crop_size, crop_size):
        crop = crop.resize((crop_size, crop_size), Image.BICUBIC)

    return crop


# ---------------------------
# データセット処理（train/val/test対応）
# ---------------------------

def iter_images(dir_path: Path, exts: Set[str], recursive: bool):
    if recursive:
        for p in dir_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p
    else:
        for p in dir_path.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                yield p


def process_dataset_split(
    in_root: Path,
    out_root: Path,
    splits: list[str],
    exts: Set[str],
    crop_size: int,
    resize_short: int,
    suffix: str,
    recursive: bool,
) -> None:
    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")

    ensure_dir(out_root)

    total = 0
    done = 0
    skipped = 0

    for split in splits:
        split_dir = in_root / split
        if not split_dir.exists():
            print(f"[WARN] split not found, skip: {split_dir}")
            continue

        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            print(f"[WARN] no class dirs under: {split_dir}")
            continue

        for class_dir in sorted(class_dirs):
            class_name = class_dir.name

            out_class_dir = out_root / split / class_name
            ensure_dir(out_class_dir)

            for img_path in sorted(iter_images(class_dir, exts, recursive)):
                total += 1
                try:
                    img = Image.open(img_path).convert("RGB")
                    out_img = smart_crop_by_gradient(img, crop_size=crop_size, resize_short=resize_short)

                    # 出力名：元stem + suffix + .png（保存形式固定）
                    out_name = f"{img_path.stem}{suffix}.png"
                    out_path = out_class_dir / out_name
                    out_img.save(out_path, format="PNG")
                    done += 1

                except Exception as e:
                    skipped += 1
                    print(f"[WARN] skip: {img_path}  reason: {e}")

    print(f"[DONE] total_found={total} processed={done} skipped={skipped} out_root='{out_root}'")


def main():
    ap = argparse.ArgumentParser(description="train/val/test対応：Resize→スマートクロップを一括適用")
    ap.add_argument("--in_root", type=str, required=True, help="入力root（train/val/test配下にclassフォルダ）")
    ap.add_argument("--out_root", type=str, required=True, help="出力root（train/val/test構造を自動生成）")
    ap.add_argument("--splits", type=str, default="train,val,test", help="対象split（カンマ区切り）")
    ap.add_argument("--crop_size", type=int, default=224, help="最終クロップの一辺")
    ap.add_argument("--resize_short", type=int, default=256, help="リサイズ後の短辺")
    ap.add_argument("--exts", type=str, default="png,jpg,jpeg,tif,tiff,bmp", help="対象拡張子（カンマ区切り）")
    ap.add_argument("--suffix", type=str, default="", help="出力ファイル名の末尾につける文字列")
    ap.add_argument("--recursive", action="store_true", help="class配下を再帰探索（サブフォルダがある場合）")

    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    exts = parse_exts(args.exts)

    process_dataset_split(
        in_root=in_root,
        out_root=out_root,
        splits=splits,
        exts=exts,
        crop_size=args.crop_size,
        resize_short=args.resize_short,
        suffix=args.suffix,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
