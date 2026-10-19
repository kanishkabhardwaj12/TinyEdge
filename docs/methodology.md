# Experimental Methodology

## Research Question
"How do model compression and lightweight model design affect predictive performance, computational efficiency, memory requirements, inference latency, throughput, and deployability on resource-constrained edge hardware?"

## Dataset
- **CIFAR-10**: 60,000 32x32 color images in 10 classes.
- **Split**: 80% Training, 10% Validation, 10% Testing.
- **Preprocessing**: Random cropping, horizontal flipping, and normalization.

## Models
1. **Reference**: ResNet-18 adapted for CIFAR-10. It uses the torchvision ResNet-18 body with
   `conv1` changed to a 3x3, stride-1 convolution and the ImageNet max-pool removed;
   this is not the unmodified ImageNet stem.
2. **Lightweight**: MobileNetV2. Uses depthwise separable convolutions to reduce parameters and FLOPs.
3. **Custom CNN**: A conventional three-stage CNN with systematically scaled channel widths.
   The width variants are architecture scaling, not pruning.

## Evaluation Protocol
- **Predictive Metrics**: Accuracy, Precision, Recall, F1-score.
- **Efficiency Metrics**:
  - Model Size (Disk MB).
  - Parameter Count.
  - Inference Latency (ms/image).
  - Throughput (FPS).
  - Peak Memory Usage (MB).
  - MACs and FLOPs are counted from Conv2d and Linear forward hooks. FLOPs are reported as
    two operations per MAC and should be interpreted as estimates for the measured input shape.

## Hardware Comparison
- **Development Machine**: MacBook Pro M2. CPU and MPS measurements are separate; MPS is
  only used when PyTorch reports it as available.
- **Target Hardware**: NVIDIA Jetson Nano (Final Hardware Benchmarking).
