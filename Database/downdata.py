import os
import zipfile
import json

# ==========================
# ⚙️ CẤU HÌNH
# ==========================

# ⚙️ Đường dẫn kaggle.json (tự động đúng cho mọi user Linux/VPS)
KAGGLE_JSON_PATH = os.path.expanduser("~/.kaggle/kaggle.json")

# 📦 Dataset trên Kaggle
DATASET = "cuonglevc/datatest"

# 📁 Thư mục đích
TARGET_DIR = "CheXNet/Database"

# ==========================
# ⚙️ TẠO FILE kaggle.json
# ==========================
def setup_kaggle():
    """Tạo file kaggle.json và phân quyền."""
    os.makedirs(os.path.dirname(KAGGLE_JSON_PATH), exist_ok=True)

    # 🧩 DÁN THÔNG TIN kaggle.json của bạn vào đây
    kaggle_json_content = {
        "username": "cuonglevc1",
        "key": "7b37d5c500c22682962b94dcadf8bad0"
    }

    with open(KAGGLE_JSON_PATH, "w") as f:
        json.dump(kaggle_json_content, f)

    os.chmod(KAGGLE_JSON_PATH, 0o600)
    print(f"✅ kaggle.json được tạo tại: {KAGGLE_JSON_PATH}")

# ==========================
# 📥 TẢI & GIẢI NÉN DATASET
# ==========================
def download_and_extract():
    """Tải và giải nén dataset Kaggle vào thư mục đích."""
    # Cài đặt thư viện Kaggle nếu chưa có
    os.system("pip install -q kaggle")

    # Tạo và thiết lập kaggle.json
    setup_kaggle()

    # Tạo thư mục đích
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Tải dataset
    print(f"⬇️  Đang tải dataset: {DATASET}")
    os.system(f"kaggle datasets download -d {DATASET} -p {TARGET_DIR}")

    # Giải nén file ZIP vừa tải
    for file in os.listdir(TARGET_DIR):
        if file.endswith(".zip"):
            zip_path = os.path.join(TARGET_DIR, file)
            print(f"📦 Giải nén: {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(TARGET_DIR)
            os.remove(zip_path)
            print("✅ Giải nén hoàn tất!")
            break
    else:
        print("❌ Không tìm thấy file ZIP trong thư mục tải.")

# ==========================
# 🚀 MAIN
# ==========================
if __name__ == "__main__":
    download_and_extract()
