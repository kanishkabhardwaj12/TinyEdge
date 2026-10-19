import torch
import numpy as np
import onnxruntime as ort
import os
import yaml
from src.models.model_factory import get_model

def compare_outputs(torch_model, onnx_path, input_shape=(1, 3, 32, 32), tolerance=1e-5):
    torch_model.eval()
    dummy_input = torch.randn(input_shape)

    # PyTorch output
    with torch.no_grad():
        torch_out = torch_model(dummy_input).numpy()

    # ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(torch_out - ort_out))
    return max_diff, torch_out, ort_out

def main():
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['models']['baseline']
    artifacts_dir = config['paths']['artifacts_dir']
    onnx_path = os.path.join(artifacts_dir, f"{model_name}.onnx")

    if not os.path.exists(onnx_path):
        print(f"ONNX file not found: {onnx_path}")
        return

    # Load PyTorch model
    model = get_model(model_name)
    best_model_path = os.path.join(artifacts_dir, f"{model_name}_best.pth")
    model.load_state_dict(torch.load(best_model_path, weights_only=False))

    try:
        diff, t_out, o_out = compare_outputs(model, onnx_path)
        print(f"ONNX Equivalence Check:")
        print(f"Max Absolute Difference: {diff:.8f}")
        print(f"Tolerance: 1e-5")
        if diff < 1e-5:
            print("Status: SUCCESS")
        else:
            print("Status: FAILED")
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    main()
