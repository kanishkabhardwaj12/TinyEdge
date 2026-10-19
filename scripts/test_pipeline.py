import torch
from src.data.dataloader import get_dataloaders
from src.models.model_factory import get_model
import yaml

with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

print("Testing dataloader...")
train_loader, val_loader, test_loader = get_dataloaders(config)
images, labels = next(iter(train_loader))
print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

print("Testing model...")
model = get_model(config['models']['baseline'])
outputs = model(images)
print(f"Outputs shape: {outputs.shape}")
print("Test successful!")
