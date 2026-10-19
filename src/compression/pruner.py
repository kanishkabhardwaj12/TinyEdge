import copy
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.models.custom_cnn import CustomCNN


class SliceLayer(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.register_buffer("indices", indices)

    def forward(self, x):
        return x[:, self.indices]


class TinyMLPruner:
    """Pruning helpers, including physical channel transfer for CustomCNN."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def prune_unstructured(self, amount=0.3):
        model_to_prune = copy.deepcopy(self.model)
        for _, module in model_to_prune.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
        return model_to_prune

    def prune_custom_cnn(self, retention, parent_model):
        """Physically transfer L1-selected filters from a trained CustomCNN."""
        if not isinstance(parent_model, CustomCNN):
            raise TypeError("CustomCNN channel pruning requires a CustomCNN parent.")
        if not 0 < retention < 1:
            raise ValueError("retention must be between zero and one.")

        parent_channels = parent_model.channels
        target_channels = tuple(max(1, round(c * retention)) for c in parent_channels)
        selected = []
        for stage, target in enumerate(target_channels):
            conv2 = parent_model.features[stage * 7 + 3]
            importance = conv2.weight.detach().abs().sum(dim=(1, 2, 3))
            indices = torch.topk(importance, target, largest=True, sorted=True).indices
            selected.append(indices.sort().values)

        child = CustomCNN(num_classes=parent_model.classifier.out_features,
                          channels=target_channels)
        previous_indices = torch.arange(3)
        for stage, output_indices in enumerate(selected):
            parent_offset = stage * 7
            child_offset = stage * 7
            parent_conv1 = parent_model.features[parent_offset]
            parent_bn1 = parent_model.features[parent_offset + 1]
            parent_conv2 = parent_model.features[parent_offset + 3]
            parent_bn2 = parent_model.features[parent_offset + 4]
            child_conv1 = child.features[child_offset]
            child_bn1 = child.features[child_offset + 1]
            child_conv2 = child.features[child_offset + 3]
            child_bn2 = child.features[child_offset + 4]

            with torch.no_grad():
                child_conv1.weight.copy_(
                    parent_conv1.weight[output_indices][:, previous_indices]
                )
                child_bn1.weight.copy_(parent_bn1.weight[output_indices])
                child_bn1.bias.copy_(parent_bn1.bias[output_indices])
                child_bn1.running_mean.copy_(parent_bn1.running_mean[output_indices])
                child_bn1.running_var.copy_(parent_bn1.running_var[output_indices])
                child_conv2.weight.copy_(
                    parent_conv2.weight[output_indices][:, output_indices]
                )
                child_bn2.weight.copy_(parent_bn2.weight[output_indices])
                child_bn2.bias.copy_(parent_bn2.bias[output_indices])
                child_bn2.running_mean.copy_(parent_bn2.running_mean[output_indices])
                child_bn2.running_var.copy_(parent_bn2.running_var[output_indices])
            previous_indices = output_indices

        with torch.no_grad():
            child.classifier.weight.copy_(parent_model.classifier.weight[:, previous_indices])
            child.classifier.bias.copy_(parent_model.classifier.bias)

        child.width_multiplier = None
        child.pruning_retention = retention
        child.parent_channels = parent_channels
        child.selected_channels = target_channels
        return child, {
            "method": "structured_channel_pruning",
            "importance": "L1 norm of second-convolution output filters per stage",
            "retention_target": retention,
            "parent_channels": list(parent_channels),
            "selected_channels": list(target_channels),
        }

    def save_pruned_model(self, model, filename):
        path = os.path.join(self.config["paths"]["artifacts_dir"], filename)
        torch.save(model, path)
        return path
