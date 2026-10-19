import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json
import os

def evaluate_model(model, loader, device, config):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {
            str(i): {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i])
            } for i in range(len(precision))
        },
        "confusion_matrix": cm.tolist()
    }

    return metrics

def save_metrics(metrics, filename, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(metrics, f, indent=4)
