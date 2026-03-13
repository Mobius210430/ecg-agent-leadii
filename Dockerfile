# 使用官方 PyTorch 镜像
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制 API 服务代码
COPY api_server.py .

# 从 COS 下载模型压缩包并解压
# 将下面的 URL 替换为你实际的对象地址
ADD https://ecg-model-1394114335.cos.ap-guangzhou.myqcloud.com/ecg-model-leadII.tar.gz /app/ecg-model-leadII.tar.gz
RUN tar -xzf ecg-model-LeadII.tar.gz -C /app && rm ecg-model-leadII.tar.gz

# 声明端口
EXPOSE 8000

# 启动命令
CMD ["python", "api_server.py"]
