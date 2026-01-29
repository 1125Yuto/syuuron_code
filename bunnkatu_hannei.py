# -*- coding: utf-8 -*-
"""
使い方:
1. このファイルを例えば rebuild_dataset_from_txt.py として保存する。
2. 下の「設定エリア」を自分の環境に合わせて書き換える:
   - SRC_ROOT: 元画像が入っているルートフォルダ
   - DEST_ROOT: 再構築した train/val/test データセットを作るルートフォルダ
   - SPLIT_TXT: train/val/test に対応する txt ファイルのパス
3. ターミナル or コマンドプロンプトで実行:
   python "C:\\pythonのコード\\k-fold\\bunnkatu_hannei.py"
4. 出力構成は以下のようになる:
   DEST_ROOT/
     train/
       クラス名/
         画像...
     val/
       クラス名/
         画像...
     test/
       クラス名/
         画像...

注意:
- txt に載っているファイルだけをコピーします。
- 元フォルダ(SRC_ROOT)に存在しないファイルはスキップし、コンソールに警告を出します。
- 既に DEST_ROOT に同名ファイルがあれば上書きされます。
"""

import os
import shutil

# ======== 設定エリア ========

# 元画像のルートフォルダ
# 例: r"D:\fungus_photo\Kabi_all_gray_scale2"
SRC_ROOT = r"D:\\fungi\\3ch_gray_clahe_meijering"

# 再構築データセットのルートフォルダ
# 例: r"D:\fungus_photo\Kabi_cv_rebuilt"
DEST_ROOT = r"D:\\fungi\\3ch_gray_clahe_meijering_folded"

# 各分割に対応する txt ファイルのパス
# 必要なものだけ指定すればOK (使わない split は None か "" にしておけば無視)
SPLIT_TXT = {
    "train": r"D:\\fungi\\k-fold\\fold5x3\\seed_000\\fold4_train.txt",  # 例
    "val":   r"D:\\fungi\\k-fold\\fold5x3\\seed_000\\fold4_val.txt",    # 例
    "test":  r"D:\\fungi\\k-fold\\fold5x3\\seed_000\\fold4_test.txt",   # 例: 今アップしてくれたやつ
}

# DEST_ROOT を最初に空にしたい場合は True
# (安全のためデフォルトは False。最初に1回だけ True にして実行→すぐ False に戻す運用が無難)
CLEAR_DEST_ROOT = False

# ============================


def load_list(txt_path):
    """
    txt ファイルから画像パスリストを読み込む。
    空行は無視する。
    """
    paths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(line)
    return paths


def build_src_path(rel_path):
    """
    txt に書かれている相対パス (例: 'Alternaria\\100.png')
    から、実際のファイルの絶対パスを作る。
    """
    # Windows の '\\' 区切りを想定しているので、os.sep に合わせて分解して join する
    parts = rel_path.replace("/", "\\").split("\\")
    return os.path.join(SRC_ROOT, *parts)


def get_class_name(rel_path):
    """
    相対パスからクラス名(最上位フォルダ名)を取り出す。
    例: 'Alternaria\\100.png' -> 'Alternaria'
    """
    parts = rel_path.replace("/", "\\").split("\\")
    return parts[0]


def rebuild_split(split_name, txt_path):
    """
    1つの split (train / val / test) に対して、
    txt の情報を元に DEST_ROOT/split_name/クラス名/ にコピーする。
    """
    if not txt_path or not os.path.exists(txt_path):
        print(f"[{split_name}] txt が指定されていないか存在しません: {txt_path}")
        return

    print(f"[{split_name}] txt からリストを読み込み中: {txt_path}")
    rel_paths = load_list(txt_path)
    print(f"[{split_name}] 画像数 (行数): {len(rel_paths)}")

    copied = 0
    missing = 0

    for rel_path in rel_paths:
        src_path = build_src_path(rel_path)
        cls_name = get_class_name(rel_path)
        dst_dir = os.path.join(DEST_ROOT, split_name, cls_name)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_path):
            # 元データから削除された or そもそも存在しない画像はスキップ
            print(f"  [WARN] 見つからないのでスキップ: {src_path}")
            missing += 1
            continue

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(f"[{split_name}] コピー完了: {copied} 枚 (missing: {missing} 枚)")


def main():
    # 出力ルートを空にするオプション
    if os.path.exists(DEST_ROOT) and CLEAR_DEST_ROOT:
        print(f"[INFO] DEST_ROOT を削除します: {DEST_ROOT}")
        shutil.rmtree(DEST_ROOT)

    os.makedirs(DEST_ROOT, exist_ok=True)

    # train / val / test それぞれ処理
    for split_name, txt_path in SPLIT_TXT.items():
        rebuild_split(split_name, txt_path)

    print("=== 全 split の処理が完了しました ===")


if __name__ == "__main__":
    main()
