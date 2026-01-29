# -*- coding: utf-8 -*-
"""
使い方：
    1. 事前準備
        pip install pillow

    2. このファイルを split_grid_dataset.py などの名前で保存する。

    3. コマンドラインから実行例：

        (1) 単一画像を 3×3 に分割
            python split_grid_dataset.py --input path/to/image.png --rows 3 --cols 3

        (2) 単一画像を 4×4 に分割
            python split_grid_dataset.py --input path/to/image.png --rows 4 --cols 4

        (3) フォルダ直下の画像だけを 5×2 に分割（サブフォルダは無視）
            python split_grid_dataset.py --input_dir path/to/dir --rows 5 --cols 2

        (4) データセット全体（サブフォルダ含む）を 3×3 に分割し、構造を保って出力
            例：dataset_root/train/classA/img1.png → output_root/train/classA/img1_row0_col0.png
            python "C:\\pythonのコード\\画像編集\\image_split_dataset.py" --input_dir "D:\\fungi\\rawdataset_main_folded" \
                --rows 5 --cols 5 --output_dir "D:\\fungi\\split\\rawdataset_main_folded_5_5split" --preserve_tree

    注意：
        ・--input と --input_dir はどちらか片方だけ指定してください。
        ・rows（縦分割数）と cols（横分割数）は 1 以上の整数を指定してください。
        ・元画像サイズは 2448×1920 などを想定していますが、任意サイズで動作します。
          端数が出る場合、最後の行／列が余りを吸収する形で分割します。
        ・--preserve_tree を付けると、input_dir 配下のフォルダ構造をそのまま output_dir に複製します。
"""

import os
import argparse
from PIL import Image


def split_image_grid(input_path, rows=3, cols=3, output_dir=None):
    """
    1枚の画像を rows x cols 分割して保存する関数

    rows: 縦方向の分割数
    cols: 横方向の分割数
    """
    # 画像を開く
    img = Image.open(input_path)
    width, height = img.size  # (横, 縦)

    # 分割数のチェック
    if rows <= 0 or cols <= 0:
        raise ValueError("rows と cols は 1 以上の整数を指定してください。")

    # 各タイルの標準サイズ（端数は最後の行・列で調整）
    tile_w = width // cols
    tile_h = height // rows

    # 出力先ディレクトリ
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    # 元ファイル名と拡張子
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)

    # rows x cols で分割
    for r in range(rows):
        for c in range(cols):
            # 左上座標
            left = c * tile_w
            upper = r * tile_h

            # 右下座標（最後の行・列は余りを吸収）
            if c == cols - 1:
                right = width
            else:
                right = (c + 1) * tile_w

            if r == rows - 1:
                lower = height
            else:
                lower = (r + 1) * tile_h

            # 画像をクロップ
            crop = img.crop((left, upper, right, lower))

            # 出力ファイル名（例: image_row0_col1.png）
            out_name = f"{name}_row{r}_col{c}{ext}"
            out_path = os.path.join(output_dir, out_name)

            crop.save(out_path)


def split_directory_grid(input_dir, rows=3, cols=3, output_dir=None, preserve_tree=False):
    """
    ディレクトリ内の画像をすべて rows x cols に分割する関数

    preserve_tree:
        False : input_dir 直下の画像のみ処理し、すべて同じ出力ディレクトリに保存
        True  : input_dir 配下を再帰的に探索し、フォルダ構造を維持したまま出力
    """
    # 出力ベースディレクトリ決定
    if output_dir is None:
        base_out_dir = os.path.join(input_dir, f"split_{rows}x{cols}")
    else:
        base_out_dir = output_dir
    os.makedirs(base_out_dir, exist_ok=True)

    # フラット（非再帰）モード：元の動作を維持
    if not preserve_tree:
        for fname in os.listdir(input_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                in_path = os.path.join(input_dir, fname)
                split_image_grid(in_path, rows=rows, cols=cols, output_dir=base_out_dir)
        return

    # データセット向け：再帰的にサブフォルダをたどる
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                continue

            in_path = os.path.join(root, fname)

            # input_dir からの相対パスを計算して構造を反映
            rel_dir = os.path.relpath(root, input_dir)  # 例: "train/classA"
            out_dir = os.path.join(base_out_dir, rel_dir)
            os.makedirs(out_dir, exist_ok=True)

            split_image_grid(in_path, rows=rows, cols=cols, output_dir=out_dir)


def main():
    parser = argparse.ArgumentParser(description="画像を任意のグリッド (rows×cols) に分割するスクリプト")
    parser.add_argument(
        "--input",
        type=str,
        help="単一画像ファイルのパス（--input_dir とどちらか片方だけ指定）",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="画像が入ったディレクトリ（データセットルートを含む）のパス（--input とどちらか片方だけ指定）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "出力ディレクトリ\n"
            "  ・未指定かつ --input 使用時  : 入力ファイルと同じディレクトリ\n"
            "  ・未指定かつ --input_dir 使用時: input_dir/ split_rowsxcols 配下に保存"
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="縦方向の分割数（デフォルト: 3）",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="横方向の分割数（デフォルト: 3）",
    )
    parser.add_argument(
        "--preserve_tree",
        action="store_true",
        help=(
            "input_dir 配下のフォルダ構造を保ったまま再帰的に処理します。\n"
            "データセット (train/val/test やクラス別フォルダ) を丸ごと処理したい場合に指定してください。"
        ),
    )

    args = parser.parse_args()

    # 入力指定チェック
    if (args.input is None and args.input_dir is None) or (
        args.input is not None and args.input_dir is not None
    ):
        print("エラー: --input か --input_dir のどちらか片方だけを指定してください。")
        return

    # rows, cols のバリデーション
    if args.rows <= 0 or args.cols <= 0:
        print("エラー: --rows と --cols は 1 以上の整数を指定してください。")
        return

    # 単一画像モード
    if args.input is not None:
        if not os.path.isfile(args.input):
            print(f"エラー: 指定されたファイルが見つかりません: {args.input}")
            return
        split_image_grid(args.input, rows=args.rows, cols=args.cols, output_dir=args.output_dir)
        print(f"完了: {args.rows}×{args.cols} 分割画像を出力しました ({args.input})")

    # ディレクトリ／データセットモード
    else:
        if not os.path.isdir(args.input_dir):
            print(f"エラー: 指定されたディレクトリが見つかりません: {args.input_dir}")
            return
        split_directory_grid(
            args.input_dir,
            rows=args.rows,
            cols=args.cols,
            output_dir=args.output_dir,
            preserve_tree=args.preserve_tree,
        )
        mode_str = "（フォルダ構造を保持して再帰的に処理）" if args.preserve_tree else "（直下のみ処理）"
        print(f"完了: ディレクトリ内の画像を {args.rows}×{args.cols} に分割しました {mode_str} {args.input_dir}")


if __name__ == "__main__":
    main()
