import torch
import os
import yaml
from src.models.model_factory import get_model

def export_baseline_onnx():
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['models']['baseline']
    artifacts_dir = config['paths']['artifacts_dir']
    onnx_path = os.path.join(artifacts_dir, f"{model_name}.onnx")

    model = get_model(model_name)
    best_model_path = os.path.join(artifacts_dir, f"{model_name}_best.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    export_baseline_onnx()
