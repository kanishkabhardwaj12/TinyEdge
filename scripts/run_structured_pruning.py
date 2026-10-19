import torch
import os
import yaml
import json
from src.models.model_factory import get_model
from src.data.dataloader import get_dataloaders
from src.evaluation.evaluator import evaluate_model, save_metrics
from src.compression.pruner import TinyMLPruner
from src.benchmarking.profiler import TinyMLProfiler

def main():
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['models']['baseline']
    artifacts_dir = config['paths']['artifacts_dir']
    results_dir = config['paths']['results_dir']

    # 1. Load trained baseline
    print(f"Loading baseline model {model_name} for structured pruning...")
    model = get_model(model_name)
    best_model_path = os.path.join(artifacts_dir, f"{model_name}_best.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Baseline model not found at {best_model_path}")
        return
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()

    # 2. Setup Data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 3. Structured Pruning
    prune_amount = 0.5 # 50% reduction
    print(f"Applying structured pruning ({prune_amount*100}%)...")
    pruner = TinyMLPruner(model, config)
    pruned_model = pruner.prune_structured_fc(amount=prune_amount)

    # 4. Evaluate Pruned Model
    print("Evaluating Pruned Model...")
    device = torch.device(config['evaluation']['device'])
    pruned_model.to(device)
    metrics_pruned = evaluate_model(pruned_model, test_loader, device, config)
    save_metrics(metrics_pruned, f"{model_name}_structured_pruned_metrics.json", results_dir)

    # 5. Profile Pruned Model
    profiler = TinyMLProfiler(pruned_model, config)
    profile_results = profiler.run_full_profile()

    # 6. Save Artifact
    pruned_path = os.path.join(artifacts_dir, f"{model_name}_structured_pruned.pth")
    torch.save(pruned_model, pruned_path)

    # 7. Record Results
    with open(os.path.join(results_dir, f"{model_name}_structured_pruned_info.json"), 'w') as f:
        json.dump({
            "model_name": f"{model_name}_structured_pruned",
            "pruning_amount": prune_amount,
            "pruning_method": "structured_fc_conv",
            "metrics": metrics_pruned,
            "profile": profile_results
        }, f, indent=4)

    print(f"Structured pruned model saved to {pruned_path}")
    print(f"Pruned Accuracy: {metrics_pruned['accuracy']:.4f}")
    print(f"New Size: {profile_results['size']['size_mb']:.2f} MB")

if __name__ == "__main__":
    main()
