import torch
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights


class ViTClassifier(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(ViTClassifier, self).__init__()
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = torch.nn.Linear(self.vit.hidden_dim, num_classes)  # Modify the classification head

    def forward(self, x):
        return self.vit(x)
