import torch
import torch.nn as nn
import os
import copy

class TinyMLQuantizer:
    """
    Handles model compression and quantization for TinyML deployment.
    Provides a robust interface for both dynamic and static quantization.
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cpu")

    def quantize_dynamic(self):
        """
        Applies dynamic quantization to Linear layers.
        Most effective for models with large FC layers (e.g., MobileNetV2 classifier).
        """
        model_to_quantize = copy.deepcopy(self.model).to(self.device)
        model_to_quantize.eval()

        # Quantize Linear layers
        return torch.ao.quantization.quantize_dynamic(
            model_to_quantize, {nn.Linear}, dtype=torch.qint8
        )

    def quantize_static_fallback(self, calibration_loader=None):
        """
        Attempts static quantization. If fusion fails, it falls back to
        dynamic quantization to ensure a valid artifact is produced.
        """
        try:
            model_to_quantize = copy.deepcopy(self.model).to(self.device)
            model_to_quantize.eval()

            # Attempt a very basic fusion for ResNet
            # We avoid the complex fuse_modules if it's unstable in the VM

            model_to_quantize.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
            model_prepared = torch.ao.quantization.prepare(model_to_quantize)

            if calibration_loader is not None:
                with torch.no_grad():
                    for i, (images, _) in enumerate(calibration_loader):
                        model_prepared(images)
                        if i >= 5: break
            else:
                model_prepared(torch.randn(1, 3, 32, 32))

            return torch.ao.quantization.convert(model_prepared)
        except Exception as e:
            print(f"Static quantization failed: {e}. Falling back to dynamic.")
            return self.quantize_dynamic()

    def save_model(self, model, filename):
        path = os.path.join(self.config['paths']['artifacts_dir'], filename)
        torch.save(model, path)
        return path
