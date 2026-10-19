# TinyEdge-AI

Research project studying the trade-off between predictive accuracy and
inference efficiency when compressing CNNs for edge deployment.

The primary experiment trains one custom CNN on CIFAR-10 and progressively
reduces its channels using structured filter pruning. The repository also
contains historical ResNet-18/MobileNetV2 experiments and an ONNX export
workflow. Current measurements were collected on a MacBook Pro M2; they are
not NVIDIA Jetson Nano measurements.

## Primary result

The completed progressive experiment is documented in:

- [`results/experiments/custom_cnn_progressive_20261019T120338521794Z/REPORT.md`](results/experiments/custom_cnn_progressive_20261019T120338521794Z/REPORT.md)
- [`results/experiments/custom_cnn_progressive_20261019T120338521794Z/results/progressive_results.csv`](results/experiments/custom_cnn_progressive_20261019T120338521794Z/results/progressive_results.csv)
- [`results/experiments/custom_cnn_progressive_20261019T120338521794Z/results/progressive_results.json`](results/experiments/custom_cnn_progressive_20261019T120338521794Z/results/progressive_results.json)

The baseline achieved 85.81% CIFAR-10 test accuracy. After fine-tuning,
structured channel retention at approximately 75%, 50%, and 25% of the
original widths achieved 82.63%, 79.30%, and 69.19%, respectively, with
physically smaller models.

## Repository layout

```text
src/          Reusable data, models, training, evaluation, profiling, pruning
scripts/      Training, profiling, export, verification, and data utilities
configs/      Experiment configuration
tests/        Lightweight implementation tests
results/      Historical results and the curated primary experiment
artifacts/   Historical model checkpoints and ONNX files
docs/         Methodology and Jetson handoff notes
data/dummy/   Small test-only tensors
```

The full CIFAR-10 dataset is intentionally not committed. It is approximately
170 MB compressed and is ignored by Git.

## Setup

Use Python 3.9+ and install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Download and extract CIFAR-10:

```bash
python3 scripts/download_data.py
tar -xzf data/cifar-10-python.tar.gz -C data
```

The extracted directory must be:

```text
data/cifar-10-batches-py/
```

The downloaded archive can be removed after extraction.

## Tests

Run the implementation tests from the repository root:

```bash
PYTHONPATH=. python3 -m unittest tests.test_pipeline
```

Tests use the small tensors in `data/dummy/` for code-path checks. Dummy data
is never used as experimental evidence.

## Primary experiment

Run the full real-data baseline and progressive pruning experiment:

```bash
PYTHONPATH=. USE_DUMMY_DATA=false \
  python3 scripts/run_progressive_experiment.py
```

The runner:

1. Trains the full-width custom CNN for 30 epochs.
2. Selects the best checkpoint using validation accuracy.
3. Builds physically smaller 75%, 50%, and 25%-width stages from the preceding
   trained stage using deterministic L1 filter importance.
4. Evaluates each stage before fine-tuning.
5. Fine-tunes each stage for five epochs using only train and validation data.
6. Profiles the serialized checkpoints using a consistent batch-size-one
   protocol.
7. Writes a new timestamped directory under `results/experiments/`.

Set `evaluation.device` in [`configs/config.yaml`](configs/config.yaml) to
`mps` on a supported Apple Silicon PyTorch installation or to `cpu` for a
CPU-only run. CPU and MPS measurements must be reported separately.

## Other commands

Historical/general-purpose pipeline commands include:

```bash
PYTHONPATH=. USE_DUMMY_DATA=true python3 scripts/run_baseline.py
PYTHONPATH=. USE_DUMMY_DATA=true python3 scripts/run_experiment.py mobilenetv2
PYTHONPATH=. python3 scripts/run_profiling.py
PYTHONPATH=. python3 scripts/export_onnx_baseline.py
PYTHONPATH=. python3 scripts/verify_onnx.py
```

The existing quantization implementation is exploratory. The curated primary
experiment records that static INT8 quantization was unavailable in the
current PyTorch backend and does not claim a quantized result.

## Jetson Nano

Jetson instructions are in [`docs/hardware_handoff.md`](docs/hardware_handoff.md).
No Jetson Nano, TensorRT, or deployment performance should be inferred from the
current laptop results.
