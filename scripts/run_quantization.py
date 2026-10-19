import torch
import os
import yaml
import json
from src.models.model_factory import get_model
from src.data.dataloader import get_dataloaders
from src.evaluation.evaluator import evaluate_model, save_metrics
from src.compression.quantizer import TinyMLQuantizer

def main():
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['models']['baseline']
    artifacts_dir = config['paths']['artifacts_dir']
    results_dir = config['paths']['results_dir']

    # 1. Load trained baseline model
    print(f"Loading baseline model {model_name} for quantization...")
    model = get_model(model_name)
    best_model_path = os.path.join(artifacts_dir, f"{model_name}_best.pth")
    if not os.path.exists(best_model_path):
        print(f"Error: Baseline model not found at {best_model_path}")
        return
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()

    # 2. Setup Data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # 3. Initialize Quantizer
    quantizer = TinyMLQuantizer(model, config)

    # 4. Perform Quantization
    print("Applying Quantization...")
    quant_model = quantizer.quantize_static_fallback(calibration_loader=train_loader)

    # 5. Evaluate Quantized Model
    print("Evaluating Quantized Model...")
    device = torch.device(config['evaluation']['device'])
    quant_model.to(device)
    metrics_quant = evaluate_model(quant_model, test_loader, device, config)
    save_metrics(metrics_quant, f"{model_name}_quantized_metrics.json", results_dir)

    # 6. Save Artifacts
    quant_path = os.path.join(artifacts_dir, f"{model_name}_quantized.pth")
    torch.save(quant_model, quant_path)

    # 7. Record and Compare Sizes
    baseline_path = os.path.join(artifacts_dir, f"{model_name}_final.pth")
    if not os.path.exists(baseline_path):
        baseline_path = best_model_path

    size_orig = os.path.getsize(baseline_path) / (1024 * 1024)
    size_quant = os.path.getsize(quant_path) / (1024 * 1024)
    print(f"Baseline Model size: {size_orig:.2f} MB")
    print(f"Quantized Model size: {size_quant:.2f} MB")
    print(f"Size Reduction: {(size_orig - size_quant) / size_orig * 100:.2f}%")

    with open(os.path.join(results_dir, f"{model_name}_quantized_info.json"), 'w') as f:
        json.dump({
            "model_name": f"{model_name}_quantized",
            "original_size_mb": size_orig,
            "quantized_size_mb": size_quant,
            "reduction_pct": (size_orig - size_quant) / size_orig * 100,
            "quantization_method": "static_int8_with_fallback"
        }, f, indent=4)

if __name__ == "__main__":
    main()
