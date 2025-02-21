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


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class MLPBlockV2(nn.Module):
    def __init__(self, dim, hidden_dim, expansion_factor=4, dropout=0.1, activation='gelu'):
        super(MLPBlockV2, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * expansion_factor)
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, dim)

        self.act = (
            nn.GELU() if activation == 'gelu'
            else nn.ReLU() if activation == 'relu'
            else nn.LeakyReLU()
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)  # Nueva normalización
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)  # Normalización final
        return x + res  # Residual connection


class MLPMixerV2(nn.Module):
    def __init__(self, in_channels=2, img_size=224, num_classes=8, patch_size=16, hidden_dim=256, num_layers=3,
                 expansion_factor=4, dropout=0.1, activation='gelu'):
        super(MLPMixerV2, self).__init__()
        num_patches = (img_size // patch_size) ** 2  # Número de parches

        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_norm = nn.BatchNorm2d(hidden_dim)  # Nueva normalización

        self.mlp_blocks = nn.Sequential(*[
            MLPBlockV2(hidden_dim, hidden_dim, expansion_factor=expansion_factor, dropout=dropout, activation=activation) for _ in
            range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim * num_patches, num_classes)

        self.apply(init_weights)  # Inicialización de pesos

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.patch_norm(x)  # Aplicar BatchNorm
        x = x.flatten(2).transpose(1, 2)  # Reorganización de dimensiones
        x = self.mlp_blocks(x)
        x = self.norm(x)
        x = x.flatten(1)  # Aplanar antes de FC
        return self.fc(x)



class MLPBlockV3(nn.Module):
    def __init__(self, dim, hidden_dim, expansion_factor=4, dropout=0.1, activation='gelu'):
        super(MLPBlockV3, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * expansion_factor)
        self.fc2 = nn.Linear(hidden_dim * expansion_factor, dim)

        self.act = (
            nn.GELU() if activation == 'gelu'
            else nn.ReLU() if activation == 'relu'
            else nn.ELU() if activation =='elu'
            else nn.LeakyReLU()
        )

        self.norm1 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.apply(init_weights)  # Inicialización de pesos

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + res  # Residual connection


class MLPMixerV3(nn.Module):
    def __init__(self, in_channels=2, img_size=224, num_classes=8, patch_size=16, hidden_dim=256, num_layers=3,
                 expansion_factor=4, dropout=0.1, activation='gelu'):
        super(MLPMixerV3, self).__init__()
        num_patches = (img_size // patch_size) ** 2  # Número de parches

        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

        #   self.patch_norm = nn.LayerNorm(hidden_dim)# Nueva normalización

        self.mlp_blocks = nn.Sequential(*[
            MLPBlockV2(hidden_dim, hidden_dim, expansion_factor=expansion_factor, dropout=dropout, activation=activation) for _ in
            range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim * num_patches, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        #    x = self.patch_norm(x)
        x = self.mlp_blocks(x)
        x = self.norm(x)
        x = x.flatten(1)  # Aplanar antes de FC
        return self.fc(x)