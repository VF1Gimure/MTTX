from torch import nn
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights

class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg(x)
