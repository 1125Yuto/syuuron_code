# ---------- USER SETTINGS ----------
ROOTS = [
"D:\\fungi\\生データセット"# ← ここを書き換える（複数可）# r"E:\another_dataset",
]
RECURSIVE = True     # サブフォルダも処理するなら True
APPLY = True       # まずは False（ドライラン）。実行反映時に True にする
AUDIT_CSV = r""      # 監査ログをCSV保存するパス。例: r"D:\audit.csv" / "" で無効
# -----------------------------------

import csv
import sys
from pathlib import Path
from typing import Optional, List, Tuple


MAGIC_SIGNATURES = {
    "tiff": [b"II*\x00", b"MM\x00*"],              # little-endian / big-endian
    "jpeg": [b"\xFF\xD8\xFF"],                      # SOI
    "png":  [b"\x89PNG\r\n\x1a\n"],
    "gif":  [b"GIF87a", b"GIF89a"],
    "bmp":  [b"BM"],
}

# 正規化する拡張子（統一ルール）
DEFAULT_EXT = {
    "tiff": ".tif",
    "jpeg": ".jpg",
    "png":  ".png",
    "gif":  ".gif",
    "bmp":  ".bmp",
}

SUPPORTED_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".gif", ".bmp"}

def sniff_format(file: Path) -> Optional[str]:
    """先頭数バイトから実フォーマットを推定。未知/失敗時は None"""
    try:
        with file.open("rb") as f:
            head = f.read(16)
    except Exception:
        return None
    for fmt, sigs in MAGIC_SIGNATURES.items():
        if any(head.startswith(sig) for sig in sigs):
            return fmt
    return None

def is_ext_ok(actual_fmt: str, ext: str) -> bool:
    """実フォーマットと拡張子の整合をチェック"""
    ext = ext.lower()
    if actual_fmt == "jpeg":
        return ext in (".jpg", ".jpeg")
    if actual_fmt == "tiff":
        return ext in (".tif", ".tiff")
    desired = DEFAULT_EXT.get(actual_fmt)
    return ext == desired

def normalized_ext(actual_fmt: str) -> str:
    return DEFAULT_EXT.get(actual_fmt, "")

def safe_rename(src: Path, dst: Path) -> Path:
    """衝突回避しながら安全にrename（同名があれば _1, _2 を付与）"""
    if not dst.exists():
        src.rename(dst)
        return dst
    stem, suffix = dst.stem, dst.suffix
    i = 1
    while True:
        candidate = dst.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            src.rename(candidate)
            return candidate
        i += 1

def collect_files(root: Path, recursive: bool):
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    else:
        yield from (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def process_root(root: Path) -> List[Tuple[str, str, Optional[str], str, str]]:
    """1つのルートフォルダを処理してアクション一覧を返す"""
    actions = []  # (path, ext, actual_format, action, target)
    if not root.exists():
        print(f"[WARN] 入力ディレクトリが見つかりません: {root}", file=sys.stderr)
        return actions
    total=ok=need_fix=skipped=0
    for fp in collect_files(root, RECURSIVE):
        total += 1
        actual = sniff_format(fp)
        ext = fp.suffix.lower()
        if actual is None:
            actions.append((str(fp), ext, None, "skip_unknown_format", ""))
            skipped += 1
            continue
        if is_ext_ok(actual, ext):
            actions.append((str(fp), ext, actual, "ok", ""))
            ok += 1
            continue
        target_ext = normalized_ext(actual)
        target = fp.with_suffix(target_ext)
        actions.append((str(fp), ext, actual, "rename_ext", str(target)))
        need_fix += 1
        if APPLY:
            try:
                result = safe_rename(fp, target)
                print(f"[RENAMED] {fp} -> {result}")
            except Exception as e:
                print(f"[ERROR] rename失敗: {fp} -> {target} ({e})", file=sys.stderr)
    # サマリ
    print(f"\n[SUMMARY] root={root}")
    print(f"  total={total}, ok={ok}, need_fix={need_fix}, skipped={skipped}")
    if not APPLY:
        print("  [NOTE] ドライランです。反映するにはファイル冒頭の APPLY=True にしてください。")
    return actions

def main():    # ROOTSが空なら対話入力
    roots = [Path(p) for p in ROOTS if str(p).strip()]
    if not roots:
        user = input("処理するフォルダのパスを入力してください: ").strip().strip('"')
        if not user:
            print("[ERROR] パスが未入力です。", file=sys.stderr)
            sys.exit(1)
        roots = [Path(user)]
    all_actions: List[Tuple[str, str, Optional[str], str, str]] = []
    for r in roots:
        all_actions.extend(process_root(r))
    # 監査CSV
    if AUDIT_CSV:
        try:
            with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["path", "ext", "actual_format", "action", "target"])
                for row in all_actions:
                    w.writerow(row)
            print(f"[AUDIT] CSV出力: {AUDIT_CSV}")
        except Exception as e:
            print(f"[ERROR] 監査CSV書き込み失敗: {AUDIT_CSV} ({e})", file=sys.stderr)

if __name__ == "__main__":
    main()