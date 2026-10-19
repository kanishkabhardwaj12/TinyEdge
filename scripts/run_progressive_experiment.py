"""Run the primary custom-CNN baseline and progressive pruning experiment."""

import csv
import json
import os
import time
from datetime import datetime, timezone

import torch
import yaml

from src.benchmarking.profiler import TinyMLProfiler
from src.compression.pruner import TinyMLPruner
from src.data.dataloader import get_dataloaders
from src.evaluation.evaluator import evaluate_model
from src.models.model_factory import get_model
from src.training.trainer import run_training
from src.utils.runtime import (
    configure_runtime,
    environment_metadata,
    resolve_device,
    save_json,
    set_reproducible_seed,
)


def save_model(model, path):
    torch.save(model, path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"Model artifact was not created: {path}")


def evaluate_and_profile(model, test_loader, config, device, artifact_path):
    # Reload the exact serialized artifact so the reported measurements are
    # for the checkpoint on disk, not only the in-memory training object.
    measured_model = torch.load(artifact_path, map_location=device, weights_only=False)
    measured_model.to(device).eval()
    metrics = evaluate_model(measured_model, test_loader, device, config)
    profile = TinyMLProfiler(measured_model, config).run_full_profile(artifact_path)
    return metrics, profile


def stage_record(stage, model, metrics, profile, metadata, **extra):
    return {
        "stage": stage,
        "parameters": profile["parameters"],
        "size": profile["size"],
        "complexity": profile["complexity"],
        "performance": profile["performance"],
        "memory": profile["memory"],
        "metrics": metrics,
        "execution_device": str(metadata["execution_device"]),
        **extra,
    }


def main():
    with open("configs/config.yaml") as handle:
        config = yaml.safe_load(handle)
    if os.environ.get("USE_DUMMY_DATA", "false").lower() == "true":
        raise RuntimeError("USE_DUMMY_DATA must be false for final experiments.")

    set_reproducible_seed(config["training"]["seed"])
    configure_runtime(config)
    device = resolve_device(config["evaluation"]["device"])
    config["evaluation"]["device"] = str(device)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = os.path.join(config["experiment"]["output_root"], f"custom_cnn_progressive_{run_id}")
    os.makedirs(root, exist_ok=False)
    artifacts = os.path.join(root, "artifacts")
    results = os.path.join(root, "results")
    os.makedirs(artifacts)
    os.makedirs(results)

    train_loader, val_loader, test_loader = get_dataloaders(config)
    split_sizes = {
        "train": len(train_loader.dataset),
        "validation": len(val_loader.dataset),
        "test": len(test_loader.dataset),
    }
    expected = {"train": 45000, "validation": 5000, "test": 10000}
    if split_sizes != expected:
        raise RuntimeError(f"Unexpected full CIFAR-10 split: {split_sizes}, expected {expected}")
    run_metadata = environment_metadata(config, split_sizes, device)
    save_json(run_metadata, os.path.join(results, "run_metadata.json"))

    records = []
    histories = {}
    start = time.time()
    baseline = get_model("custom_cnn")
    baseline_best, baseline_time, baseline_epoch, history = run_training(
        config, baseline, train_loader, val_loader, "custom_cnn",
        artifact_dir=artifacts, epochs=30, return_history=True
    )
    histories["stage_0_baseline"] = history
    baseline_best_path = os.path.join(artifacts, "stage_0_baseline_best_state.pth")
    baseline.load_state_dict(torch.load(
        os.path.join(artifacts, "custom_cnn_best.pth"),
        map_location=device, weights_only=False
    ))
    save_model(baseline, baseline_best_path)
    baseline_final_path = os.path.join(artifacts, "stage_0_baseline_final.pth")
    save_model(baseline, baseline_final_path)
    metrics, profile = evaluate_and_profile(
        baseline, test_loader, config, device, baseline_best_path
    )
    records.append(stage_record(
        "stage_0_baseline", baseline, metrics, profile, run_metadata,
        model_name="custom_cnn", parent_checkpoint=None,
        compression=None, best_validation_accuracy_percent=baseline_best,
        best_validation_epoch=baseline_epoch,
        training_duration_seconds=baseline_time,
        checkpoint=baseline_best_path,
    ))

    parent_model = baseline
    parent_checkpoint = baseline_best_path
    pruner = TinyMLPruner(parent_model, config)
    # Ratios are relative to the preceding stage while resulting widths target
    # approximately 75%, 50%, and 25% of the original baseline widths.
    for stage, retention in enumerate((0.75, 2 / 3, 0.5), start=1):
        child, compression = pruner.prune_custom_cnn(retention, parent_model)
        compression["target_original_retention"] = (0.75, 0.50, 0.25)[stage - 1]
        immediate_path = os.path.join(artifacts, f"stage_{stage}_pruned_immediate.pth")
        save_model(child, immediate_path)
        immediate_metrics, immediate_profile = evaluate_and_profile(
            child, test_loader, config, device, immediate_path
        )
        records.append(stage_record(
            f"stage_{stage}_pruned_immediate", child, immediate_metrics,
            immediate_profile, run_metadata, model_name="custom_cnn",
            parent_checkpoint=parent_checkpoint, compression=compression,
            checkpoint=immediate_path, fine_tuned=False,
        ))

        fine_dir = os.path.join(artifacts, f"stage_{stage}_fine_tuned")
        os.makedirs(fine_dir)
        fine_start = time.time()
        best_acc, fine_time, best_epoch, fine_history = run_training(
            config, child, train_loader, val_loader, f"stage_{stage}_custom_cnn",
            artifact_dir=fine_dir, epochs=config["training"]["fine_tune_epochs"],
            return_history=True
        )
        histories[f"stage_{stage}_fine_tuned"] = fine_history
        fine_best_state = os.path.join(fine_dir, f"stage_{stage}_custom_cnn_best.pth")
        child.load_state_dict(torch.load(
            fine_best_state, map_location=device, weights_only=False
        ))
        fine_path = os.path.join(artifacts, f"stage_{stage}_fine_tuned.pth")
        save_model(child, fine_path)
        fine_metrics, fine_profile = evaluate_and_profile(
            child, test_loader, config, device, fine_path
        )
        records.append(stage_record(
            f"stage_{stage}_fine_tuned", child, fine_metrics, fine_profile,
            run_metadata, model_name="custom_cnn", parent_checkpoint=immediate_path,
            compression=compression, checkpoint=fine_path, fine_tuned=True,
            best_validation_accuracy_percent=best_acc,
            best_validation_epoch=best_epoch,
            training_duration_seconds=fine_time,
        ))
        parent_model = child
        parent_checkpoint = fine_path
        pruner = TinyMLPruner(parent_model, config)

    save_json({"records": records, "training_histories": histories},
              os.path.join(results, "progressive_results.json"))
    with open(os.path.join(results, "progressive_results.csv"), "w", newline="") as handle:
        fields = [
            "stage", "accuracy", "macro_f1", "parameters", "size_mb",
            "macs", "flops", "mean_latency_s", "median_latency_s",
            "p95_latency_s", "throughput_fps", "memory_rss_mb",
            "parent_checkpoint", "checkpoint", "fine_tuned",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                "stage": record["stage"],
                "accuracy": record["metrics"]["accuracy"],
                "macro_f1": record["metrics"]["macro_f1"],
                "parameters": record["parameters"]["total_params"],
                "size_mb": record["size"]["size_mb"],
                "macs": record["complexity"]["macs"],
                "flops": record["complexity"]["flops"],
                "mean_latency_s": record["performance"]["mean_latency_s"],
                "median_latency_s": record["performance"]["median_latency_s"],
                "p95_latency_s": record["performance"]["p95_latency_s"],
                "throughput_fps": record["performance"]["throughput_fps"],
                "memory_rss_mb": record["memory"]["peak_memory_mb"],
                "parent_checkpoint": record["parent_checkpoint"],
                "checkpoint": record["checkpoint"],
                "fine_tuned": record["fine_tuned"] if "fine_tuned" in record else False,
            })

    report = [
        "# TinyEdge-AI Progressive Custom-CNN Experiment",
        "",
        "This report contains laptop measurements only; no Jetson Nano measurements were taken.",
        "",
        f"- Run directory: `{root}`",
        f"- Device: `{device}`",
        f"- Total wall time: `{time.time() - start:.2f}` seconds",
        "- Baseline: one 30-epoch custom CNN trained on real CIFAR-10.",
        "- Compression: progressive L1 filter-based structured channel transfer.",
        "- Fine-tuning: 5 epochs per compressed stage.",
        "- Test set: used only for final evaluation and profiling.",
        "- Memory: process RSS estimate, not accelerator peak tensor memory.",
        "",
        "Results are in `results/progressive_results.csv` and `results/progressive_results.json`.",
    ]
    with open(os.path.join(root, "REPORT.md"), "w") as handle:
        handle.write("\n".join(report) + "\n")
    print(f"Progressive experiment complete: {root}")


if __name__ == "__main__":
    main()
