import os
import glob
import json
import random
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold, train_test_split

# ========= ユーザ設定 =========
BASE_DIR   = "D:\\fungi\\rawdataset_png_main"
IMG_DIR    = BASE_DIR
META_PATH  = os.path.join(BASE_DIR, "meta.csv")
SPLIT_DIR  = os.path.join(BASE_DIR, "splits", "fold5x3")

N_SPLITS   = 5        # k-fold の k
N_SEEDS    = 3        # シード数
OOD_RATIO  = 0.0      # 外部テスト比率（0.0 なら作らない）
SHUFFLE    = True     # fold 生成時にシャッフル
# ============================


def make_meta(meta_path=META_PATH, img_dir=IMG_DIR, base_dir=BASE_DIR):
    """images/ 以下を走査して meta.csv を作成（相対パス＋ラベル）"""
    rows = []
    for class_name in sorted(os.listdir(img_dir)):
        class_dir = os.path.join(img_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for p in glob.glob(os.path.join(class_dir, "*")):
            if os.path.isdir(p):
                continue
            rel = os.path.relpath(p, BASE_DIR)
            rows.append({"path": rel, "label": class_name})
    if not rows:
        raise RuntimeError(f"No images found under: {img_dir}")

    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    df.to_csv(meta_path, index=False)
    print(f"[meta] saved {meta_path} with {len(df)} rows, {df['label'].nunique()} classes")
    return df


def sanity_check_class_counts(y, k):
    """各クラスの枚数が k 以上あるか事前チェック(StratifiedKFold の必須条件）"""
    c = Counter(y)
    too_small = {cls: n for cls, n in c.items() if n < k}
    if too_small:
        msg = "\n".join([f"  - {cls}: {n} (< {k})" for cls, n in too_small.items()])
        raise ValueError(
            "[error] Some classes have fewer samples than k-folds.\n"
            f"Each class must have at least k={k} images.\n{msg}"
        )


def write_list(paths, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(list(paths)))


def make_splits(meta_path=META_PATH, split_dir=SPLIT_DIR,
                n_splits=N_SPLITS, n_seeds=N_SEEDS,
                ood_ratio=OOD_RATIO, shuffle=SHUFFLE):
    """val と test を完全分離する Stratified K-fold 分割を書き出す"""

    df = pd.read_csv(meta_path)
    X_all = df["path"].values
    y_all = df["label"].values

    sanity_check_class_counts(y_all, max(n_splits, 2))  # 念のため

    for seed in range(n_seeds):
        seed_dir = os.path.join(split_dir, f"seed_{seed:03d}")
        os.makedirs(seed_dir, exist_ok=True)

        # ---- OOD（外部テスト）を先に stratify で切り出し ----
        if ood_ratio and ood_ratio > 0.0:
            X_rem, X_ood, y_rem, y_ood = train_test_split(
                X_all, y_all, test_size=ood_ratio, stratify=y_all, random_state=seed
            )
            write_list(X_ood, os.path.join(seed_dir, "ood_test.txt"))
        else:
            X_rem, y_rem = X_all, y_all
            # 空でも仕様としてファイルは作っておく（パイプライン簡略化）
            write_list([], os.path.join(seed_dir, "ood_test.txt"))

        # ---- K-fold を作成（val/test を別 fold に固定）----
        sanity_check_class_counts(y_rem, n_splits)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

        # 各 fold のインデックスを事前にリスト化
        folds = [(tr_idx, va_idx) for tr_idx, va_idx in skf.split(X_rem, y_rem)]
        # scikit-learn の仕様上、(tr_idx, va_idx) の「va_idx」が fold の本体
        # ここでは「val_fold = f」「test_fold = (f+1)%k」に固定し、
        # train = それ以外の全 fold とする

        # fold 単位アクセスを容易にするため、fold → indices に変換
        fold_indices = [va_idx for _, va_idx in folds]  # 各 fold のバリデーション部分

        N = len(X_rem)
        all_idx = set(range(N))

        for f in range(n_splits):
            val_idx  = set(fold_indices[f])
            test_idx = set(fold_indices[(f + 1) % n_splits])
            train_idx = list(all_idx - val_idx - test_idx)

            # 整合性チェック
            assert len(val_idx & test_idx) == 0, "val と test が重複しています"
            assert len(set(train_idx) & val_idx) == 0
            assert len(set(train_idx) & test_idx) == 0
            assert len(train_idx) + len(val_idx) + len(test_idx) == len(X_rem)

            X_train = X_rem[train_idx]
            X_val   = X_rem[list(val_idx)]
            X_test  = X_rem[list(test_idx)]

            # 保存
            write_list(X_train, os.path.join(seed_dir, f"fold{f}_train.txt"))
            write_list(X_val,   os.path.join(seed_dir, f"fold{f}_val.txt"))
            write_list(X_test,  os.path.join(seed_dir, f"fold{f}_test.txt"))

        # 参考情報: 分布ログ
        summary = {
            "seed": seed,
            "classes": sorted(Counter(y_all).keys()),
            "n_all": len(X_all),
            "n_rem": len(X_rem),
            "n_ood": 0 if not ood_ratio else len(X_ood),
            "n_splits": n_splits
        }
        with open(os.path.join(seed_dir, "_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[split] seed {seed:03d} done. rem={len(X_rem)} ood={summary['n_ood']}")

    print(f"[done] outputs under: {split_dir}")


if __name__ == "__main__":
    # 1) メタ生成（初回だけでOK）
    if not os.path.exists(META_PATH):
        make_meta()

    # 2) 分割生成（5-fold×3-seed、val/test 完全分離）
    make_splits()
