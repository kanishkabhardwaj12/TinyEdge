import yaml
import torch
import os
import time
import sys
from datetime import datetime, timezone
from src.data.dataloader import get_dataloaders
from src.models.model_factory import get_model
from src.training.trainer import run_training
from src.evaluation.evaluator import evaluate_model, save_metrics
from src.utils.runtime import environment_metadata, save_json, set_reproducible_seed, resolve_device

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <model_name>")
        print("Example: python run_experiment.py resnet18")
        sys.exit(1)

    model_name = sys.argv[1]

    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    set_reproducible_seed(config['training']['seed'])
    device = resolve_device(config['evaluation']['device'])
    config['evaluation']['device'] = str(device)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = os.path.join(
        config['experiment']['output_root'], f"{model_name}_{run_id}"
    )
    artifact_dir = os.path.join(output_dir, "artifacts")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(artifact_dir, exist_ok=False)
    os.makedirs(results_dir, exist_ok=False)

    # Setup data
    print(f"Loading data for {model_name}...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Setup model
    print(f"Initializing model: {model_name}")
    model = get_model(model_name)

    # Train
    print(f"Starting training for {model_name}...")
    best_val_acc, train_time, best_epoch = run_training(
        config, model, train_loader, val_loader, model_name, artifact_dir=artifact_dir
    )
    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%, Time: {train_time:.2f}s")

    # Load best model for final evaluation
    best_model_path = os.path.join(artifact_dir, f"{model_name}_best.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=False))

    # Evaluate
    print("Evaluating on test set...")
    model.to(device)
    metrics = evaluate_model(model, test_loader, device, config)

    # Save metrics
    # Use a consistent name: modelname_metrics.json
    save_metrics(metrics, f"{model_name}_metrics.json", results_dir)

    # Record model size
    final_path = os.path.join(artifact_dir, f"{model_name}_final.pth")
    torch.save(model, final_path)
    model_size = os.path.getsize(final_path) / (1024 * 1024)
    print(f"Model size: {model_size:.2f} MB")

    split_sizes = {
        "train": len(train_loader.dataset),
        "validation": len(val_loader.dataset),
        "test": len(test_loader.dataset),
    }
    metadata = environment_metadata(config, split_sizes, device)
    metadata.update({
        "model_name": model_name,
        "training_duration_seconds": train_time,
        "best_validation_accuracy_percent": best_val_acc,
        "best_validation_epoch": best_epoch,
        "model_size_mb": model_size,
        "params": sum(p.numel() for p in model.parameters()),
    })
    save_json(metadata, os.path.join(results_dir, f"{model_name}_metadata.json"))
    print(f"Experiment outputs saved to {output_dir}")

if __name__ == "__main__":
    main()
