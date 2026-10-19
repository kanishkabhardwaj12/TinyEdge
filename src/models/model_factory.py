import torch
import torch.nn as nn
from torchvision import models
from src.models.custom_cnn import CustomCNN

class QuantizableResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()

        # Use ResNet-18
        self.model = models.resnet18(weights=None)
        # Adjust for CIFAR-10
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def get_model(model_name, num_classes=10):
    if model_name == "resnet18":
        return QuantizableResNet(num_classes)

    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        model.features[0][0].stride = (1, 1)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model

    elif model_name in {"custom_cnn", "custom_cnn_w100"}:
        return CustomCNN(num_classes=num_classes, width_multiplier=1.0)
    elif model_name == "custom_cnn_w075":
        return CustomCNN(num_classes=num_classes, width_multiplier=0.75)
    elif model_name == "custom_cnn_w050":
        return CustomCNN(num_classes=num_classes, width_multiplier=0.5)
    elif model_name == "custom_cnn_w025":
        return CustomCNN(num_classes=num_classes, width_multiplier=0.25)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
