import torch
import torch.nn as nn
from torchvision.models import resnet50


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove the classification layer

    def forward(self, x):
        with torch.no_grad():
            return self.model(x).flatten(1)