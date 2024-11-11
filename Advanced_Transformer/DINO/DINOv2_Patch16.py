import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm

class DINOv2_16Patch(nn.Module):
    def __init__(self,H,W) -> None:
        super().__init__()
        self.H=H
        self.W=W
        patch_size = 16 
        self.model = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m', 
            pretrained=True,
            num_classes=0,  
        )
        self.model.patch_embed=timm.models.vision_transformer.PatchEmbed(
            patch_size=patch_size,
            in_chans=3,
            embed_dim=self.model.embed_dim
        )
        num_patches = (H // patch_size) * (W // patch_size)
        self.model.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.model.embed_dim) * 0.02)
        self.model.patch_embed.img_size = (H, W) 


    def forward(self, x):
        features = self.model.get_intermediate_layers(x, 4)

        return features

def main():
    H, W = 320, 1024
    
    model=DINOv2_16Patch(H, W)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    x = torch.randn(1, 3, H, W)  
    x = x.to(device)

    feature=model(x)
    