"""
使い方:
  # 例: epoch=20 の混同行列を作って保存（CSV+PNG）
  python "C:\\pythonのコード\\Deep_learning\\confusion_matrix_from_epoch_csv.py" ^
    --csv "D:\\Study\\crossvalidation_result4\\train_rate_change\\early_stop_scheguler\\vit_l_32\\seed_042\\fold0\\confmat_val_epoch.csv" ^
    --epoch 15 ^
    --out_dir "D:\\Study\\crossvalidation_result4\\train_rate_change\\early_stop_scheguler\\vit_l_32\\seed_042\\fold0" ^
    --normalize none

  # 正規化（true行で割る）して割合表示にしたい場合
  python make_confusion_matrix_from_epoch_csv.py --csv confmat_val_epoch.csv --epoch 20 --normalize true

入力CSVの想定フォーマット（あなたの confmat_val_epoch.csv に対応）:
  epoch, true_class_id, true_class_name, pred_class_id, pred_class_name, count

出力:
  out_dir/confmat_epoch{EPOCH}_raw.csv  （混同行列の数値）
  out_dir/confmat_epoch{EPOCH}_plot.png （混同行列のヒートマップ）

依存:
  pip install pandas numpy matplotlib
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REQUIRED_COLS = [
    "epoch",
    "true_class_id",
    "true_class_name",
    "pred_class_id",
    "pred_class_name",
    "count",
]


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要列がありません: {missing}\n実際の列: {list(df.columns)}")


def _build_class_maps(df_epoch: pd.DataFrame) -> Tuple[List[int], List[str], List[int], List[str]]:
    """
    true/pred 側それぞれの class_id と class_name を安定に作る。
    """
    true_map = (
        df_epoch[["true_class_id", "true_class_name"]]
        .drop_duplicates()
        .sort_values("true_class_id")
    )
    pred_map = (
        df_epoch[["pred_class_id", "pred_class_name"]]
        .drop_duplicates()
        .sort_values("pred_class_id")
    )

    true_ids = true_map["true_class_id"].astype(int).tolist()
    true_names = true_map["true_class_name"].astype(str).tolist()
    pred_ids = pred_map["pred_class_id"].astype(int).tolist()
    pred_names = pred_map["pred_class_name"].astype(str).tolist()
    return true_ids, true_names, pred_ids, pred_names


def _make_confusion_matrix(
    df_epoch: pd.DataFrame,
    true_ids: List[int],
    pred_ids: List[int],
) -> np.ndarray:
    """
    count から混同行列（行=true, 列=pred）を構築。
    """
    t_index = {cid: i for i, cid in enumerate(true_ids)}
    p_index = {cid: j for j, cid in enumerate(pred_ids)}

    cm = np.zeros((len(true_ids), len(pred_ids)), dtype=np.int64)

    for row in df_epoch.itertuples(index=False):
        ti = t_index[int(row.true_class_id)]
        pj = p_index[int(row.pred_class_id)]
        cm[ti, pj] = int(row.count)

    return cm


def _normalize_cm(cm: np.ndarray, mode: str) -> np.ndarray:
    """
    mode:
      - none: そのまま
      - true: 行（true）方向で正規化
      - pred: 列（pred）方向で正規化
      - all : 全体で正規化
    """
    mode = mode.lower()
    if mode == "none":
        return cm.astype(float)

    cm_f = cm.astype(float)
    eps = 1e-12

    if mode == "true":
        denom = cm_f.sum(axis=1, keepdims=True) + eps
        return cm_f / denom
    if mode == "pred":
        denom = cm_f.sum(axis=0, keepdims=True) + eps
        return cm_f / denom
    if mode == "all":
        denom = cm_f.sum() + eps
        return cm_f / denom

    raise ValueError("--normalize は none / true / pred / all のいずれかです")


from matplotlib.colors import LogNorm

def _plot_cm(
    cm_show: np.ndarray,
    true_names: List[str],
    pred_names: List[str],
    title: str,
    out_png: str,
    is_normalized: bool,
    scale: str = "linear",   # 追加
) -> None:
    fig_w = max(8, 0.6 * len(pred_names))
    fig_h = max(8, 0.6 * len(true_names))

    plt.figure(figsize=(fig_w, fig_h))

    # ★追加：ログスケール（正規化のときは不要）
    norm = None
    if (not is_normalized) and (scale == "log"):
        # LogNorm は 0 を扱えないので、正の最小値を使う
        positive = cm_show[cm_show > 0]
        if positive.size > 0:
            vmin = float(np.min(positive))
            vmax = float(np.max(cm_show))
            norm = LogNorm(vmin=vmin, vmax=vmax)

    plt.imshow(cm_show, aspect="auto", norm=norm)
    plt.colorbar()

    plt.xticks(np.arange(len(pred_names)), pred_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(true_names)), true_names)

    plt.title(title)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    n_cells = cm_show.size
    if n_cells <= 900:
        for i in range(cm_show.shape[0]):
            for j in range(cm_show.shape[1]):
                v = cm_show[i, j]
                s = f"{v:.2f}" if is_normalized else f"{int(v)}"
                plt.text(j, i, s, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="confmat_val_epoch.csv のパス")
    ap.add_argument("--epoch", type=int, required=True, help="抽出したい epoch 番号")
    ap.add_argument("--out_dir", default="out_confmat", help="出力先フォルダ")
    ap.add_argument(
        "--normalize",
        default="none",
        choices=["none", "true", "pred", "all"],
        help="混同行列の正規化方式",
    )
    ap.add_argument("--scale", default="linear", choices=["linear", "log"], help="raw表示のスケール")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    _validate_columns(df)

    epochs = sorted(df["epoch"].dropna().astype(int).unique().tolist())
    if args.epoch not in epochs:
        raise ValueError(f"epoch={args.epoch} がCSVに存在しません。存在するepoch例: {epochs[:20]} ... (合計 {len(epochs)} 個)")

    df_epoch = df[df["epoch"].astype(int) == args.epoch].copy()

    true_ids, true_names, pred_ids, pred_names = _build_class_maps(df_epoch)

    # 念のため true/pred のクラス集合が同じなら、順序も揃える（id順で）
    # ただし片側にしか存在しないクラスがあるケースもあるので、ここは無理に一致させない。
    cm = _make_confusion_matrix(df_epoch, true_ids, pred_ids)

    cm_show = _normalize_cm(cm, args.normalize)
    is_norm = args.normalize != "none"

    os.makedirs(args.out_dir, exist_ok=True)

    out_csv = os.path.join(args.out_dir, f"confmat_epoch{args.epoch}_{args.normalize}.csv")
    out_png = os.path.join(args.out_dir, f"confmat_epoch{args.epoch}_{args.normalize}.png")

    # 行列を DataFrame 化して保存
    cm_df = pd.DataFrame(cm_show, index=true_names, columns=pred_names)
    cm_df.to_csv(out_csv, encoding="utf-8-sig")

    title = f"Confusion Matrix (epoch={args.epoch}, normalize={args.normalize})"
    _plot_cm(cm_show, true_names, pred_names, title, out_png, is_norm, scale=args.scale)

    print(f"[OK] saved: {out_csv}")
    print(f"[OK] saved: {out_png}")


if __name__ == "__main__":
    main()
