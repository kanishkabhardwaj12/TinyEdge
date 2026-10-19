import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import yaml
import os

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total

def run_training(config, model, train_loader, val_loader, model_name,
                 artifact_dir=None, epochs=None, return_history=False):
    device = torch.device(config['evaluation']['device'])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(config['training']['learning_rate']),
                          momentum=0.9, weight_decay=float(config['training']['weight_decay']))

    best_val_acc = 0.0
    best_epoch = 0
    epochs = epochs or config['training']['epochs']
    artifact_dir = artifact_dir or config['paths']['artifacts_dir']
    os.makedirs(artifact_dir, exist_ok=True)

    start_time = time.time()
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy_percent": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy_percent": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(artifact_dir, f"{model_name}_best.pth"))

    total_time = time.time() - start_time
    result = (best_val_acc, total_time, best_epoch)
    return result + (history,) if return_history else result
