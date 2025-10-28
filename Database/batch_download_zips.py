import os
import urllib.request
import tarfile
import time

base_dir = "CheXNet/Database"
os.makedirs(base_dir, exist_ok=True)

links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]


def show_progress(block_num, block_size, total_size):
    """Hiển thị tiến độ tải xuống với tốc độ (MB/s)."""
    global start_time
    if block_num == 0:
        start_time = time.time()
        return

    duration = time.time() - start_time
    downloaded = block_num * block_size
    speed = downloaded / 1024 / 1024 / duration if duration > 0 else 0
    percent = downloaded * 100 / total_size if total_size > 0 else 0

    print(f"\r📦 {percent:5.1f}% | {speed:6.2f} MB/s", end='')


for idx, link in enumerate(links):
    fn = os.path.join(base_dir, f"images_{idx+1:03d}.tar.gz")
    print(f"\n📥 Đang tải {fn} ...")

    urllib.request.urlretrieve(link, fn, reporthook=show_progress)
    print("\n✅ Tải xong!")

    extract_dir = os.path.splitext(os.path.splitext(fn)[0])[0]
    os.makedirs(extract_dir, exist_ok=True)

    print(f"📂 Đang giải nén {fn} vào {extract_dir} ...")
    with tarfile.open(fn, "r:gz") as tar:
        tar.extractall(path=extract_dir, filter="data")

    os.remove(fn)
    print(f"🗑️ Đã xóa {fn}")

print("\n🎉 Hoàn tất tải và giải nén tất cả file vào thư mục riêng trong CheXNet/Database")
