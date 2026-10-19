import torch.nn as nn


class CustomCNN(nn.Module):
    """A small CIFAR-10 CNN with systematically scalable channel widths."""

    BASE_CHANNELS = (32, 64, 128)

    def __init__(self, num_classes=10, width_multiplier=1.0, channels=None):
        super().__init__()
        if channels is None:
            channels = tuple(max(1, round(width * width_multiplier))
                             for width in self.BASE_CHANNELS)
        if len(channels) != len(self.BASE_CHANNELS) or any(c < 1 for c in channels):
            raise ValueError("CustomCNN requires three positive channel counts.")
        channels = tuple(int(c) for c in channels)

        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.width_multiplier = width_multiplier
        self.channels = channels

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)
