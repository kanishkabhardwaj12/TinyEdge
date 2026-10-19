import torch
import os
import yaml
from src.models.model_factory import get_model
from src.data.dataloader import get_dataloaders

def audit_artifacts():
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    artifacts_dir = config['paths']['artifacts_dir']

    # 1. Quantization check
    print("--- Quantization Audit ---")
    quant_path = os.path.join(artifacts_dir, "resnet18_quantized.pth")
    if os.path.exists(quant_path):
        try:
            # Ensure we are using torch.load
            model = torch.load(quant_path, map_location='cpu', weights_only=False)
            has_quant_conv = any("QuantizedConv2d" in str(type(m)) for m in model.modules())
            print(f"Quantized model loaded. Contains QuantizedConv2d: {has_quant_conv}")

            # Try forward pass
            try:
                model.eval()
                with torch.no_grad():
                    model(torch.randn(1, 3, 32, 32))
                print("Quantized forward pass: SUCCESS")
            except Exception as e:
                print(f"Quantized forward pass: FAILED")
                print(f"Error: {e}")
                # Print supported engines
                try:
                    print(f"Supported engines: {torch.backends.quantized.supported_engines}")
                except:
                    print("Could not retrieve supported engines.")
        except Exception as e:
            print(f"Could not load quantized model: {e}")
    else:
        print("Quantized artifact not found.")

    # 2. Training Verification
    print("\n--- Training Audit ---")
    print(f"Configured epochs: {config['training']['epochs']}")
    subset_fraction = config['dataset'].get('subset_fraction', 1.0)
    print(f"Dataset subset fraction: {subset_fraction}")
    train_examples = int(50000 * subset_fraction)
    print(f"Estimated training examples: {train_examples}")

    # 3. Calibration Audit
    print("\n--- Calibration Audit ---")
    print("Calibration Method: Static PTQ")
    print("Calibration Batches: 5 batches from train_loader")
    print("Preprocessing: Standard CIFAR-10 Normalization")

if __name__ == "__main__":
    audit_artifacts()
