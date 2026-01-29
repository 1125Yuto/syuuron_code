"""
Attention Rollout for torchvision ViT / Swin / MaxViT
対応モデル: vit_b_32, vit_l_32, swin_v2_b, maxvit_t
visualize attention maps.

使い方:
python "C:\\pythonのコード\\Deep_learning\\attention_rollout_transformer.py" ^
  --save_dir "D:\\Study\\crossvalidation_result4\\train_rate_change\\early_stop_scheguler" --model_name vit_b_32 --seed 42 --fold 0 ^
  --input "D:\\fungi\\smart_crop\\visuarize\\vit_l_32_vis" --output_dir "D:\\Study\\crossvalidation_result4\\train_rate_change\\early_stop_scheguler\\vit_l_32\\seed_042\\fold0" --recursive

"""
"""
Attention Rollout for torchvision ViT (vit_b_32 / vit_l_32)

使い方:
python "C:\\pythonのコード\\Deep_learning\\attention_rollout_transformer.py" ^
  --save_dir "D:\\Study\\crossvalidation_result4\\train_rate_change\\early_stop_scheguler" ^
  --model_name vit_l_32 --seed 42 --fold 0 ^
  --input "D:\\fungi\\smart_crop\\visuarize\\vit_l_32_vis" ^
  --output_dir "D:\\Study\\crossvalidation_result4\\train_rate_change\\early_stop_scheguler\\vit_l_32\\seed_042\\fold0" --recursive
"""
import os
import argparse
from pathlib import Path
import types

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.vision_transformer import EncoderBlock
from PIL import Image
import matplotlib.pyplot as plt


# =======================
# ViT Attention Rollout
# =======================

def enable_vit_attention_rollout(model):
    """
    torchvision ViT 用
    nn.MultiheadAttention.forward をラップして attention を保存しつつ、
    呼び出し元互換のため (out, attn) を必ず返す。
    """
    import types
    import torch
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            if hasattr(m, "_rollout_enabled"):
                continue

            original_forward = m.forward

            def forward_with_rollout(self, *args, **kwargs):
                # 可能なら attention を取得する方向に強制
                kwargs["need_weights"] = True
                # PyTorch の版によっては無いので try 的に扱う
                kwargs.setdefault("average_attn_weights", False)

                res = original_forward(*args, **kwargs)

                # res が (out, attn) の場合
                if isinstance(res, tuple) and len(res) == 2:
                    out, attn = res
                    if isinstance(attn, torch.Tensor):
                        self.attention_weights = attn.detach()
                    return out, attn

                # res が out だけ返す版が混ざっても、2戻りに揃える
                out = res
                attn = None
                return out, attn

            m.forward = types.MethodType(forward_with_rollout, m)
            m._rollout_enabled = True



def collect_vit_attentions(model):
    """
    monkey-patch 済み ViT から attention を集める
    """
    attns = []
    for m in model.modules():
        if hasattr(m, "attention_weights"):
            attns.append(m.attention_weights.cpu())
    return attns


def rollout_attention(attns):
    """
    attns: list of (B, heads, N, N)
    return: (N, N)
    """
    mats = []
    for a in attns:
        a = a.mean(dim=1)   # head average → (B, N, N)
        mats.append(a[0])

    n = mats[0].shape[-1]
    eye = torch.eye(n)

    result = mats[0] + eye
    result = result / result.sum(dim=-1, keepdim=True)

    for m in mats[1:]:
        m = m + eye
        m = m / m.sum(dim=-1, keepdim=True)
        result = m @ result

    return result.numpy()


# =======================
# Utility
# =======================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_images(path, recursive):
    p = Path(path)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if p.is_file():
        return [p]
    if recursive:
        return [x for x in p.rglob("*") if x.suffix.lower() in exts]
    else:
        return [x for x in p.glob("*") if x.suffix.lower() in exts]


def save_rollout(img, rollout, outpath):
    cls_attn = rollout[0, 1:]  # CLS → patches
    side = int(np.sqrt(len(cls_attn)))
    attn_map = cls_attn.reshape(side, side)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() + 1e-8)

    heat = Image.fromarray(np.uint8(attn_map * 255)).resize(img.size)
    plt.imshow(img)
    plt.imshow(heat, cmap="jet", alpha=0.45)
    plt.axis("off")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


# =======================
# Main
# =======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--model_name", choices=["vit_b_32", "vit_l_32"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    NUM_CLASSES = 8  # ← あなたのタスクに合わせる

    if args.model_name == "vit_b_32":
        model = models.vit_b_32(weights=None, num_classes=NUM_CLASSES)
    else:
        model = models.vit_l_32(weights=None, num_classes=NUM_CLASSES)


    # ===== load checkpoint =====
    ckpt = os.path.join(
        args.save_dir,
        args.model_name,
        f"seed_{args.seed:03d}",
        f"fold{args.fold}",
        "best_model.pth"
    )
    state = torch.load(ckpt, map_location="cpu")

    new_state = {k.replace("base.", "", 1): v for k, v in state.items()}
    model.load_state_dict(new_state, strict=True)
    model.eval()

    enable_vit_attention_rollout(model)

    tf = build_transform(args.img_size)
    files = load_images(args.input, args.recursive)

    for fp in files:
        img = Image.open(fp).convert("RGB")
        x = tf(img).unsqueeze(0)

        _ = model(x)

        attns = collect_vit_attentions(model)
        if len(attns) == 0:
            print("[WARN] no attention:", fp)
            continue

        rollout = rollout_attention(attns)
        out = os.path.join(args.output_dir, fp.stem + "_rollout.png")
        save_rollout(img, rollout, out)
        print("Saved:", out)


if __name__ == "__main__":
    main()
