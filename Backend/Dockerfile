FROM node:20-alpine AS builder

WORKDIR /app

# Cài đặt các công cụ build cần thiết
RUN apk --no-cache add g++ make python3 git && yarn global add node-gyp@9.4.0

# Copy package files
COPY package.json yarn.lock* /app/

# Cài đặt toàn bộ dependencies (bao gồm cả devDependencies để build)
RUN yarn install --pure-lockfile

# Copy mã nguồn
COPY . /app

# Build TypeScript sang JavaScript
RUN yarn build

# Cài đặt lại chỉ các production dependencies để giảm dung lượng
RUN yarn install --pure-lockfile --production

FROM node:20-alpine

WORKDIR /app

# Cài đặt curl để hỗ trợ docker health check
RUN apk --no-cache add curl

# Copy kết quả build từ stage builder
COPY --from=builder /app .

# Tạo thư mục uploads nếu chưa tồn tại
RUN mkdir -p /app/uploads

EXPOSE 3000

CMD ["yarn", "start"]
