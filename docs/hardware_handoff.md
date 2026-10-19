# Hardware Handoff Guide (Jetson Nano)

This document provides the exact steps required to execute the benchmarks on the physical NVIDIA Jetson Nano.

## Prerequisites
- NVIDIA Jetson Nano with JetPack installed.
- Docker and NVIDIA Container Toolkit installed.
- **Verified JetPack Version**: The current Dockerfile targets `l4t-pytorch:r32.7.1-py3`. Please confirm if your Nano uses a different JetPack version (e.g., 4.6.1) and update the base image if necessary.

## Deployment Steps

1. **Build the Image**:
   ```bash
   docker build -t tinyml-benchmark .
   ```

2. **Run the Container**:
   Run the container with GPU access:
   ```bash
   docker run --runtime nvidia -it --rm -v $(pwd)/artifacts:/app/artifacts -v $(pwd)/results:/app/results tinyml-benchmark /bin/bash
   ```

3. **Execute Benchmarks**:
   Inside the container, run the benchmark script for each model:

   **Baseline (ResNet-18):**
   ```bash
   python3 scripts/run_benchmark.py --model resnet18 --path artifacts/resnet18_final.pth --output results/nano_resnet18_bench.json
   ```

   **Lightweight (MobileNetV2):**
   ```bash
   python3 scripts/run_benchmark.py --model mobilenetv2 --path artifacts/mobilenetv2_final.pth --output results/nano_mobilenetv2_bench.json
   ```

   **Quantized (ResNet-18 Quant):**
   ```bash
   python3 scripts/run_benchmark.py --model resnet18 --path artifacts/resnet18_quantized.pth --output results/nano_resnet18_quant_bench.json
   ```

## Data to Collect
For each run, please provide:
1. The resulting `.json` benchmark file.
2. The JetPack version.
3. The power mode used (e.g., 5W or 10W).
4. Peak memory usage (using `tegrastats`).
