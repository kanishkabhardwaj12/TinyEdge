FROM nvcr.io/nvidia/l4t-pytorch:r32.7.1-py3

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create artifacts directory
RUN mkdir -p artifacts results

# Set entrypoint for benchmarking
ENTRYPOINT ["python3", "scripts/run_benchmark.py"]
