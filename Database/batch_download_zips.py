import os
import urllib.request
import tarfile

base_dir = "CheXNet/Database"


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

for idx, link in enumerate(links):
    fn = os.path.join(base_dir, f"images_{idx+1:03d}.tar.gz")
    print(f"📥 Đang tải {fn} ...")
    urllib.request.urlretrieve(link, fn)

    # Tạo thư mục cùng tên với file nén (bỏ .tar.gz)
    extract_dir = os.path.splitext(os.path.splitext(fn)[0])[0]
    os.makedirs(extract_dir, exist_ok=True)

    # Giải nén vào thư mục riêng
    print(f"📂 Đang giải nén {fn} vào {extract_dir} ...")
    with tarfile.open(fn, "r:gz") as tar:
        tar.extractall(path=extract_dir, filter="data")
    
    os.remove(fn)
    print(f"🗑️ Đã xóa {fn}")

print("✅ Hoàn tất tải và giải nén tất cả file vào thư mục riêng trong CheXNet/Database")