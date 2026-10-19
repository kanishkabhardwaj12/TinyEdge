# TinyEdge-AI: Progressive Custom-CNN Compression

## Research question and scope

This experiment measures the accuracy/efficiency trade-off when one trained
custom CNN is progressively reduced using physically structured channel
pruning. These are laptop measurements on a MacBook Pro M2; no Jetson Nano
measurement or deployment claim is made.

The primary sequence is:

```text
stage 0 trained baseline
  -> stage 1: channels 75% of original
  -> stage 2: channels 50% of original
  -> stage 3: channels 25% of original
```

Each compressed stage was created from the preceding fine-tuned checkpoint,
evaluated immediately, then fine-tuned for five epochs using only the training
and validation loaders.

## Protocol

- Dataset: real CIFAR-10.
- Split: 45,000 train / 5,000 validation / 10,000 official test.
- Input: 3 x 32 x 32.
- Training augmentation: random crop and horizontal flip, training only.
- Evaluation preprocessing: tensor conversion and CIFAR-10 normalization.
- Baseline: `CustomCNN`, channels `32 / 64 / 128`.
- Optimizer: SGD, learning rate `0.01`, momentum `0.9`, weight decay `5e-4`.
- Batch size: `64`.
- Baseline budget: 30 epochs.
- Fine-tuning budget: 5 epochs per compressed stage.
- Seed: `42`.
- Device: `mps` for training and profiling.
- CPU thread setting: `8` (recorded; MPS was the execution device).
- Latency: batch size 1, 10 warm-up iterations, 100 timed iterations.
- Throughput: `1 / mean batch-size-one latency`.
- FLOPs convention: two FLOPs per counted MAC.
- Memory: process RSS before/after forward; this is an estimate, not
  accelerator peak tensor memory.

Complete metadata is in [`run_metadata.json`](results/run_metadata.json).
Training histories are in [`progressive_results.json`](results/progressive_results.json).

## Results

| Stage | Channels | Test accuracy | Parameters | Size MB | MACs | Mean latency ms | Throughput img/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 32/64/128 | 85.81% | 288,746 | 1.123 | 655,616 | 1.820 | 549.5 |
| 75% immediate | 24/48/96 | 14.32% | 162,994 | 0.643 | 498,624 | 1.649 | 606.5 |
| 75% fine-tuned | 24/48/96 | 82.63% | 162,994 | 0.642 | 498,624 | 1.729 | 578.5 |
| 50% immediate | 16/32/64 | 19.45% | 72,954 | 0.298 | 341,632 | 1.549 | 645.5 |
| 50% fine-tuned | 16/32/64 | 79.30% | 72,954 | 0.298 | 341,632 | 1.588 | 629.6 |
| 25% immediate | 8/16/32 | 10.00% | 18,626 | 0.090 | 184,640 | 1.501 | 666.1 |
| 25% fine-tuned | 8/16/32 | 69.19% | 18,626 | 0.090 | 184,640 | 1.474 | 678.5 |

The final fine-tuned stages therefore reduced parameters to approximately
56.5%, 25.3%, and 6.5% of the baseline, and reduced serialized checkpoint
size to approximately 57.2%, 26.5%, and 8.0% of the baseline. These are
parameter/storage outcomes, not claims that latency must scale identically.

Fine-tuning recovered most of the immediate pruning loss:

- 75% width retention: 14.32% -> 82.63%.
- 50% width retention: 19.45% -> 79.30%.
- 25% width retention: 10.00% -> 69.19%.

The complete measurements, including macro-F1, p95 latency, RSS estimates,
checkpoint paths, and parent links, are in
[`progressive_results.csv`](results/progressive_results.csv) and
[`progressive_results.json`](results/progressive_results.json).
The final profiling pass reloaded each checkpoint from disk before measuring
it; each record has `artifact_execution_verified: true`.

## Compression method and lineage

For each convolutional stage, deterministic L1 norms of the second
convolution's output filters selected the retained filters. The transfer
operation physically copied:

- selected output filters;
- corresponding BatchNorm weights, biases, running means, and variances;
- downstream convolution input channels;
- final classifier input channels.

The lineage is recorded in each JSON record. For example:

```text
stage_0_baseline_best_state.pth
  -> stage_1_pruned_immediate.pth
  -> stage_1_fine_tuned.pth
  -> stage_2_pruned_immediate.pth
  -> stage_2_fine_tuned.pth
  -> stage_3_pruned_immediate.pth
  -> stage_3_fine_tuned.pth
```

## Secondary experiments

### Static INT8 quantization

The intended static post-training INT8 attempt was run against the trained
baseline without fallback. It failed before conversion because this PyTorch
installation reports only `qnnpack` and `none` as supported quantized engines,
with the default engine set to `none`. The exact evidence is in
[`quantization_attempt.json`](results/quantization_attempt.json).

No quantized accuracy, size, or latency claim is made. The existing fallback
quantizer was not used because silently changing methods would invalidate the
comparison.

### ResNet-18

No new ResNet-18 run was launched in this execution. Historical ResNet-18
files remain preliminary and are not mixed into this primary result. The
repository's reference uses a CIFAR-10-adapted stem: 3x3 stride-1 `conv1`
and no ImageNet max-pool.

## Reproduction

From the repository root:

```bash
PYTHONPATH=. USE_DUMMY_DATA=false \
  python3 scripts/run_progressive_experiment.py
```

Each invocation creates a new timestamped directory under
`results/experiments/` and does not overwrite prior runs.

## Validation and limitations

The focused test suite completed successfully:

```text
Ran 9 tests
OK
```

The tests cover output shapes, model scaling, checkpoint save/load, physical
channel reduction, parameter reduction, split sizes/disjointness, profiler
MAC reporting, device discovery, and experiment configuration. They are
implementation checks and are not experimental evidence.

Known limitations:

- MPS latency is not comparable to CPU latency without a separate CPU run.
- RSS measurements are process-level estimates.
- MAC/FLOP counts cover Conv2d and Linear layers and omit non-arithmetic
  operation costs.
- No Jetson Nano or TensorRT measurements were performed.
- The interrupted earlier runs in sibling timestamped directories are
  incomplete and must not be used as final results.
