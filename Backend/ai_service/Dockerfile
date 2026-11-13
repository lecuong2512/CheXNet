FROM python:3.10-slim

WORKDIR /app

# Cài đặt các công cụ hệ thống cần thiết
# Lưu ý: libgl1-mesa-glx đã bị xóa trong Debian Trixie, dùng libgl1 thay thế
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài đặt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn dịch vụ AI
COPY . /app

EXPOSE 8000

# Khởi chạy FastAPI server
CMD ["python", "main.py"]
