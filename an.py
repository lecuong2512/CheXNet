import pandas as pd
from pathlib import Path, PurePosixPath
import csv
import re

# Thêm prefix Kaggle + chèn "images/" sau "images_###/"
PREFIX  = "/kaggle/input/datasets/data/"
IN_TXT  = Path(r"D:\Science_Of_AI\Pandas\complete-pandas-tutorial\data\test_list.txt")
OUT_TXT = IN_TXT.with_name(IN_TXT.stem + "_prefixed_new_2.txt")

# Đọc dòng, bỏ trống
lines = IN_TXT.read_text(encoding="utf-8", errors="ignore").splitlines()
lines = [ln.strip() for ln in lines if ln.strip()]

df = pd.DataFrame({"line": lines})

# Chuẩn hoá slash đầu dòng, đổi backslash -> slash
df["line"] = df["line"].str.replace(r"^[\\/]+", "", regex=True).str.replace("\\", "/")

# Chèn 'images/' sau 'images_###/' nếu CHƯA có sẵn
# VD: images_006/0001.png -> images_006/images/0001.png
df["line"] = df["line"].str.replace(r"(images_\d+/)(?!images/)", r"\1images/", regex=True)

# Thêm PREFIX nếu dòng chưa bắt đầu bằng PREFIX
df["line"] = df["line"].apply(
    lambda s: str(PurePosixPath(PREFIX) / s) if not s.startswith(PREFIX) else s
)

# Ghi ra file, giữ dạng space-separated như đầu vào
df["line"].to_csv(
    OUT_TXT,
    index=False,
    header=False,
    lineterminator="\n",
    quoting=csv.QUOTE_NONE,
    escapechar="\\",
    encoding="utf-8",
)
print(f"Đã tạo: {OUT_TXT}")
