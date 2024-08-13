# Use NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set working directory
WORKDIR /workspace/

# Set Python path environment variable
ENV PYTHONPATH=/workspace/

# Copy all project files
COPY . .

# Install dependencies
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt
