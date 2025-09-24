import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, UNet3DConditionModel
from diffusers.models.attention_processor import XLAFlashAttnProcessor2_0

from ddim.config import IMAGE_SIZE, MODEL_DIM, NUM_TRAIN_TIMESTEPS
from monai.networks.nets import ViT

class ConditionalUNet(nn.Module):
    def __init__(self, sample_size=IMAGE_SIZE, cond_dim=256): # Increased cond_dim
        super().__init__()
        no_of_in_mri_channels = 4
        
        # MRI encoder now outputs a larger embedding
        self.mri_encoder = nn.Sequential(
            nn.Conv2d(no_of_in_mri_channels, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 192, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((sample_size // 8, sample_size // 8)),
            nn.Conv2d(192, cond_dim, 1), # Match the new cond_dim
        )
        
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            
            # 🚀 KEY CHANGE: Increased channel width for more capacity
            block_out_channels=(96, 192, 384, 512), 
            
            down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D', 'AttnDownBlock2D'),
            up_block_types=('AttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
            
            # 🚀 KEY CHANGE: Match the new, larger conditioning dimension
            cross_attention_dim=cond_dim 
        )

    def forward(self, x_t, timesteps, mri):
        context = self.mri_encoder(mri)
        B, C, H, W = context.shape
        context = context.view(B, C, -1).permute(0, 2, 1)
        
        # create faded sample to ensure model learns to output true structure 
        fade_factor = (timesteps / (NUM_TRAIN_TIMESTEPS - 1)).view(-1, 1, 1, 1)
        faded_mri = mri * fade_factor
        faded_x_t = x_t * (1 - fade_factor)
        inp = faded_x_t + faded_mri

        out = self.unet(
            sample=inp,
            timestep=timesteps,
            encoder_hidden_states=context
        ).sample
        return out

class ConditionalUNetNew(nn.Module):
    def __init__(self, sample_size=IMAGE_SIZE, cond_dim=256): # Increased cond_dim
        super().__init__()
        no_of_in_mri_channels = 4
        
        # MRI encoder with more channels to create a larger embedding
        self.mri_encoder = nn.Sequential(
            nn.Conv2d(no_of_in_mri_channels, 128, 3, padding=1), # Increased channels
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), # Increased channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((sample_size // 8, sample_size // 8)),
            nn.Conv2d(256, cond_dim, 1), # Match the new cond_dim
        )
        
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=8,
            out_channels=4,
            layers_per_block=2,
            #attention_type=XLAFlashAttnProcessor2_0,
            
            # 🚀 KEY CHANGE: Increased channel width for more capacity
            block_out_channels=(128, 256, 512, 768), 
            
            # down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D', 'AttnDownBlock2D'),
            # up_block_types=('AttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D', 'UpBlock2D'),
            
            # 🚀 KEY CHANGE: Match the new, larger conditioning dimension
            cross_attention_dim=cond_dim 
        )
        
    def make_unet_cool(self):
        self.unet.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        self.unet.fuse_qkv_projections()
        #self.unet.set_attn_processor(XLAFlashAttnProcessor2_0)
        
    def get_context(self, mri):
        context = self.mri_encoder(mri)
        B, C, H, W = context.shape
        print(f'{context.shape=}')
        context = context.view(B, C, -1).permute(0, 2, 1)
        return context
    
    def apply_unet(self, x_t, timesteps, mri, context):
        inp = torch.cat([x_t, mri], dim=1)  # Concatenate MRI with x_t
        
        out = self.unet(
            sample=inp,
            timestep=timesteps,
            encoder_hidden_states=context
        ).sample
        return out
    
    def forward(self, x_t, timesteps, mri):
        context = self.mri_encoder(mri)
        B, C, H, W = context.shape
        context = context.view(B, C, -1).permute(0, 2, 1)
        
        # # create faded sample to ensure model learns to output true structure 
        # fade_factor = (timesteps / (NUM_TRAIN_TIMESTEPS - 1)).view(-1, 1, 1, 1)
        # faded_mri = mri * fade_factor
        # faded_x_t = x_t * (1 - fade_factor)
        # inp = faded_x_t + faded_mri
        inp = torch.cat([x_t, mri], dim=1)  # Concatenate MRI with x_t
        out = self.unet(
            sample=inp,
            timestep=timesteps,
            encoder_hidden_states=context
        ).sample
        return out
    
class ConditionalUNet3D(nn.Module):
    def __init__(self, sample_size=IMAGE_SIZE, cond_dim=256):
        super().__init__()
        
        # 3D ViT encoder
        self.vit_encoder = ViT(
            in_channels=4,                       # e.g., MRI has 4 channels
            img_size=(32, 128, 128),  # cubic volume
            patch_size=(8, 8, 8),
            hidden_size=cond_dim,
            mlp_dim=cond_dim * 4,
            num_layers=4, num_heads=4,
            proj_type="conv",
            pos_embed_type="learnable",
            spatial_dims=3
        )
        
        # 3D U-Net denoiser
        self.unet = UNet3DConditionModel(
            sample_size=sample_size,
            in_channels=8,      # x_t + MRI
            out_channels=4,     # output volume channels
            cross_attention_dim=cond_dim,
            
            block_out_channels=(192, 256, 512, 768),
            down_block_types=(
                "DownBlock3D", 
                "CrossAttnDownBlock3D", 
                "DownBlock3D", 
                "CrossAttnDownBlock3D"
            ),
            up_block_types=(
                "CrossAttnUpBlock3D", 
                "UpBlock3D", 
                "CrossAttnUpBlock3D", 
                "UpBlock3D"
            ),
            layers_per_block=2,
        )
        self.unet.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        self.unet.fuse_qkv_projections()
    
    def get_context(self, mri):
        # ViT forward → output sequence (B, num_patches, cond_dim)
        context, _hidden_state = self.vit_encoder(mri)  # MONAI ViT returns (B, cond_dim, ...)

        return context


    def step(self, x_t, timesteps, mri, context):
        inp = torch.cat([x_t, mri], dim=1)  # (B, in_channels=8, D, H, W)
        
        out = self.unet(
            sample=inp,
            timestep=timesteps,
            encoder_hidden_states=context
        ).sample
        
        return out
    
    def forward(self, x_t, timesteps, mri):
        context = self.get_context(mri)        
        return self.step(x_t, timesteps, mri, context)

class ConditionalUNet2DViT(nn.Module):
    def __init__(self, sample_size=IMAGE_SIZE, cond_dim=128):
        super().__init__()
        
        # 2D ViT encoder
        self.vit_encoder = ViT(
            in_channels=4,                 # e.g., 4 MRI channels for a slice
            img_size=(128, 128),            # height, width only
            patch_size=(4, 4),            # 2D patches
            hidden_size=cond_dim,
            mlp_dim=cond_dim * 4,
            num_layers=2,
            num_heads=2,
            proj_type="conv",
            pos_embed_type="learnable",
            spatial_dims=2                  # <-- 2D mode
        )
        
        # 2D U-Net denoiser
        self.unet = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=8,       # x_t + MRI slices stacked along channel dim
            out_channels=4,      # predicted slice channels
            cross_attention_dim=cond_dim,
            
            block_out_channels=(128, 256, 512, 768),
            down_block_types=(
                "DownBlock2D", 
                "CrossAttnDownBlock2D", 
                "DownBlock2D", 
                "CrossAttnDownBlock2D"
            ),
            up_block_types=(
                "CrossAttnUpBlock2D", 
                "UpBlock2D", 
                "CrossAttnUpBlock2D", 
                "UpBlock2D"
            ),
            layers_per_block=2,
        )
        # Optional optimizations from diffusers
        self.unet.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        self.unet.fuse_qkv_projections()
    
    def get_context(self, mri):
        # MRI shape: (B, 4, 128, 128)
        context, _hidden_state = self.vit_encoder(mri)  # (B, num_patches, cond_dim)
        return context

    def step(self, x_t, timesteps, mri, context):
        # Concatenate noisy slice and conditioning MRI slice
        inp = torch.cat([x_t, mri], dim=1)  # (B, 8, H, W)
        
        out = self.unet(
            sample=inp,
            timestep=timesteps,
            encoder_hidden_states=context
        ).sample
        
        return out
    
    def forward(self, x_t, timesteps, mri):
        context = self.get_context(mri)
        return self.step(x_t, timesteps, mri, context)

