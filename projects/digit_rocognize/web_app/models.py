import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, in_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            # Conv layer 1: (96 filters, 11x11 kernel, stride 4, padding 2) + ReLU + MaxPool
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv layer 2: (256 filters, 5x5 kernel, stride 1, padding 2) + ReLU + MaxPool
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv layer 3: (384 filters, 3x3 kernel, stride 1, padding 1) + ReLU
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Conv layer 4: (384 filters, 3x3 kernel, stride 1, padding 1) + ReLU
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Conv layer 5: (256 filters, 3x3 kernel, stride 1, padding 1) + ReLU + MaxPool
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            logits = self(x)
        preds = logits.argmax(dim=-1)
        return preds