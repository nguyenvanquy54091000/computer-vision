import torch
import torch.nn as nn

patch_size = 32
image_size = 224
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32] 

class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 4, 1) 
        return patches.flatten(2) 

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(patch_size * patch_size * 3, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patch):
        positions = torch.arange(0, self.num_patches, step=1).to(patch.device)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(projection_dim)
        self.mha = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout, batch_first=True)
        self.layernorm2 = nn.LayerNorm(projection_dim)
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x1 = self.layernorm1(x)
        attn_output, _ = self.mha(x1, x1, x1)
        x2 = x + attn_output
        x3 = self.layernorm2(x2)
        return x2 + self.mlp(x3)

class ViTObjectDetector(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        self.patch_layer = Patches(patch_size)
        self.patch_encoder = PatchEncoder(num_patches, projection_dim)
        
        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(projection_dim, num_heads) for _ in range(transformer_layers)]
        )
        
        self.shared_head = nn.Sequential(
            nn.LayerNorm(projection_dim),
            nn.Flatten(),
            nn.Linear(num_patches * projection_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.box_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 4), 
            nn.Sigmoid() 
        )

        self.class_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, num_classes) 
        )

    def forward(self, x):
        patches = self.patch_layer(x)
        encoded = self.patch_encoder(patches)
        features = self.transformer_layers(encoded)
        
        shared_features = self.shared_head(features)
        
        bboxes = self.box_head(shared_features)
        logits = self.class_head(shared_features)
        
        return bboxes, logits
