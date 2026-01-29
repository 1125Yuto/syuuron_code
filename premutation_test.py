"""
使い方:
  pip install numpy pandas scikit-learn

  # 例1: 予測CSVを与えて、Permutation testで差のp値を出す（paired sign-flip）
  python "C:\\pythonのコード\\statistics\\premutation_test.py" \
    --raw_csv "D:\\Study\\crossvalidation_result2\\raw\\raw_resukt.csv" \
    --pre_csv "D:\\Study\\crossvalidation_result2\\CLAHE\\CLAHE_reslt.csv" \
    --label_col y_true \
    --pred_col y_pred \
    --id_col fold \
    --metric macro_f1 \
    --n_perm 200 \
    --seed 42

  # 例2: 連続スコア（softmax確率など）がある場合（pred_colに確率列名）
  python perm_test_paired.py \
    --raw_csv raw_pred.csv \
    --pre_csv pre_pred.csv \
    --label_col y_true \
    --score_col p1 \
    --id_col sample_id \
    --metric macro_f1 \
    --threshold 0.5

  # 例3: 同等性の補助（ブートストラップ90%CIが[-Δ,+Δ]に入るか）
  python perm_test_paired.py \
    --raw_csv raw_pred.csv \
    --pre_csv pre_pred.csv \
    --label_col y_true \
    --pred_col y_pred \
    --id_col sample_id \
    --metric macro_f1 \
    --n_perm 20000 \
    --equiv_margin 0.02 \
    --n_boot 5000

想定するCSV形式:
  - sample_id: サンプルID（同一データをraw/preで突合するため必須）
  - y_true: 正解ラベル（整数 or 文字列）
  - y_pred: 予測ラベル（整数 or 文字列） もしくは score_col で確率スコア
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compute_metric(y_true, y_pred, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    elif metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    elif metric == "micro_f1":
        return float(f1_score(y_true, y_pred, average="micro"))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def load_and_align(
    raw_csv: str,
    pre_csv: str,
    id_col: str,
    label_col: str,
    pred_col: str | None,
    score_col: str | None,
    threshold: float | None,
):
    raw = pd.read_csv(raw_csv)
    pre = pd.read_csv(pre_csv)

    # ここで同一サンプルをIDで突合（欠損があると検定が歪むので、両方に存在するIDのみ使う）
    merged = raw.merge(pre, on=id_col, suffixes=("_raw", "_pre"), how="inner")
    if merged.empty:
        raise ValueError("No overlapping sample IDs between raw and pre CSVs.")

    y_true_raw = merged[f"{label_col}_raw"].to_numpy()
    y_true_pre = merged[f"{label_col}_pre"].to_numpy()

    # 正解ラベルが一致しているか確認（不一致ならデータ作りがおかしい）
    if not np.array_equal(y_true_raw, y_true_pre):
        # どこがズレているかを少し出す
        mismatch = np.where(y_true_raw != y_true_pre)[0][:10]
        raise ValueError(
            f"y_true differs between raw and pre for some samples (first mismatches idx={mismatch.tolist()}). "
            "Ensure the same ground-truth labels for the same sample_id."
        )

    y_true = y_true_raw

    if pred_col is not None:
        y_raw = merged[f"{pred_col}_raw"].to_numpy()
        y_pre = merged[f"{pred_col}_pre"].to_numpy()
        return y_true, y_raw, y_pre

    if score_col is not None:
        if threshold is None:
            raise ValueError("--threshold is required when using --score_col.")
        s_raw = merged[f"{score_col}_raw"].to_numpy().astype(float)
        s_pre = merged[f"{score_col}_pre"].to_numpy().astype(float)
        # 二値分類想定：確率→ラベル化（必要なら拡張）
        y_raw = (s_raw >= threshold).astype(int)
        y_pre = (s_pre >= threshold).astype(int)
        return y_true, y_raw, y_pre

    raise ValueError("Either --pred_col or --score_col must be provided.")


def permutation_test_paired_signflip(
    y_true,
    y_raw,
    y_pre,
    metric: str,
    n_perm: int,
    seed: int,
    alternative: str = "two-sided",
):
    """
    paired permutation via sign-flip on per-sample paired differences:
      各サンプルで (raw, pre) を入れ替えるかどうかをランダムに決める
    帰無仮説: rawとpreは交換可能（差はゼロ）
    """
    rng = np.random.default_rng(seed)

    # 観測統計量（差）
    stat_raw = compute_metric(y_true, y_raw, metric)
    stat_pre = compute_metric(y_true, y_pre, metric)
    obs = stat_pre - stat_raw

    # 置換分布
    perm_stats = np.empty(n_perm, dtype=float)

    # ここが核心：サンプルごとにswapするか決める（Bernoulli）
    y_raw = np.asarray(y_raw)
    y_pre = np.asarray(y_pre)

    for i in range(n_perm):
        swap = rng.random(len(y_true)) < 0.5
        # swap=True のところはrawとpreを入れ替える
        yA = np.where(swap, y_pre, y_raw)
        yB = np.where(swap, y_raw, y_pre)

        statA = compute_metric(y_true, yA, metric)
        statB = compute_metric(y_true, yB, metric)
        perm_stats[i] = statA - statB

    # p値計算（連続統計量の置換）
    if alternative == "two-sided":
        p = (np.sum(np.abs(perm_stats) >= np.abs(obs)) + 1) / (n_perm + 1)
    elif alternative == "greater":
        p = (np.sum(perm_stats >= obs) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(perm_stats <= obs) + 1) / (n_perm + 1)
    else:
        raise ValueError("alternative must be one of: two-sided, greater, less")

    return {
        "metric": metric,
        "raw": stat_raw,
        "pre": stat_pre,
        "diff(pre-raw)": obs,
        "p_value": float(p),
        "alternative": alternative,
    }


def bootstrap_ci_diff(
    y_true,
    y_raw,
    y_pre,
    metric: str,
    n_boot: int,
    seed: int,
    ci: float = 0.90,
):
    """
    サンプル（画像）単位のブートストラップで差のCIを作る
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot, dtype=float)

    y_true = np.asarray(y_true)
    y_raw = np.asarray(y_raw)
    y_pre = np.asarray(y_pre)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stat_raw = compute_metric(y_true[idx], y_raw[idx], metric)
        stat_pre = compute_metric(y_true[idx], y_pre[idx], metric)
        diffs[i] = stat_pre - stat_raw

    alpha = (1 - ci) / 2
    lo = float(np.quantile(diffs, alpha))
    hi = float(np.quantile(diffs, 1 - alpha))
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--pre_csv", required=True)

    parser.add_argument("--id_col", default="sample_id")
    parser.add_argument("--label_col", default="y_true")

    parser.add_argument("--pred_col", default="y_pred", help="Predicted label column (classification).")
    parser.add_argument("--score_col", default=None, help="Score/probability column (binary). If used, pred_col is ignored.")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for score_col (binary).")

    parser.add_argument("--metric", choices=["accuracy", "macro_f1", "micro_f1"], default="macro_f1")
    parser.add_argument("--n_perm", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alternative", choices=["two-sided", "greater", "less"], default="two-sided")

    # 同等性（任意）
    parser.add_argument("--equiv_margin", type=float, default=None, help="Equivalence margin Δ for diff(pre-raw).")
    parser.add_argument("--n_boot", type=int, default=0, help="If >0, compute bootstrap CI for diff(pre-raw).")

    args = parser.parse_args()

    # score_colが指定されたらpred_colは無視
    pred_col = None if args.score_col is not None else args.pred_col

    y_true, y_raw, y_pre = load_and_align(
        raw_csv=args.raw_csv,
        pre_csv=args.pre_csv,
        id_col=args.id_col,
        label_col=args.label_col,
        pred_col=pred_col,
        score_col=args.score_col,
        threshold=args.threshold,
    )

    res = permutation_test_paired_signflip(
        y_true=y_true,
        y_raw=y_raw,
        y_pre=y_pre,
        metric=args.metric,
        n_perm=args.n_perm,
        seed=args.seed,
        alternative=args.alternative,
    )

    print("=== Paired Permutation Test (sign-flip / swap within pair) ===")
    for k, v in res.items():
        print(f"{k}: {v}")

    # ブートストラップCI（任意）
    if args.n_boot and args.n_boot > 0:
        lo, hi = bootstrap_ci_diff(
            y_true=y_true,
            y_raw=y_raw,
            y_pre=y_pre,
            metric=args.metric,
            n_boot=args.n_boot,
            seed=args.seed,
            ci=0.90,
        )
        print("\n=== Bootstrap CI for diff(pre-raw) ===")
        print(f"90% CI: [{lo:.6f}, {hi:.6f}]")

        if args.equiv_margin is not None:
            delta = float(args.equiv_margin)
            equiv = (lo >= -delta) and (hi <= delta)
            print("\n=== Equivalence (via 90% CI within [-Δ, +Δ]) ===")
            print(f"Δ: {delta}")
            print(f"Equivalent: {equiv}")

    # 最低限の整合性チェック
    if len(y_true) < 20:
        print("\n[WARN] sample size is small; permutation p-values may be unstable.", file=sys.stderr)


if __name__ == "__main__":
    main()
