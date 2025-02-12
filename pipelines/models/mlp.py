import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, expansion_factor=4):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * expansion_factor)
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, dim)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x + res  # Residual connection

class MLPMixer(nn.Module):
    def __init__(self, in_channels=2, img_size=224, num_classes=8, patch_size=16, hidden_dim=256, num_layers=3):
        super(MLPMixer, self).__init__()
        num_patches = (img_size // patch_size) ** 2  # Correct calculation
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # Stack MLP Blocks
        self.mlp_blocks = nn.Sequential(*[
            MLPBlock(hidden_dim, hidden_dim, expansion_factor=4) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim * num_patches, num_classes)  # Corrected FC layer input size

    def forward(self, x):
        x = self.patch_embed(x)  # Convert image to patches
        x = x.flatten(2).transpose(1, 2)  # Correct reshaping order
        x = self.mlp_blocks(x)
        x = self.norm(x)
        x = x.flatten(1)  # Flatten before FC
        return self.fc(x)

