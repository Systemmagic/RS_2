import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ResNetDecoder(nn.Module):
    """解码器：将1D潜向量还原为2D地图"""
    def __init__(self, config: Config):
        super().__init__()
        self.latent_dim = config.LATENT_DIM
        self.init_h = config.IMG_SIZE // 32
        # 确定初始通道数（使其与潜向量维度相匹配进行上采样）
        # 对于256×256图像，32倍缩小后为8×8
        # 如果latent_dim=128: 128→256通道×8×8
        # 如果latent_dim=256: 256→512通道×8×8，但我们用更大的初始通道来保证上采样顺利
        self.init_channels = min(512, max(256, (self.latent_dim // 2)))
        self.flat_dim = self.init_channels * self.init_h * self.init_h
        
        # 潜向量映射回特征图维度
        self.fc = nn.Linear(config.LATENT_DIM, self.flat_dim)
        
        # 转置卷积上采样
        self.up_layers = nn.Sequential(
            nn.ConvTranspose2d(self.init_channels, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        # 最终卷积还原单通道
        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, z):
        # 1D向量→2D特征图
        x = self.fc(z)
        x = x.view(x.size(0), self.init_channels, self.init_h, self.init_h)
        # 上采样
        x = self.up_layers(x)
        # 还原单通道地图
        x = self.final_conv(x)
        # OUTPUT CONSTRAINT: Clamp to [0, 1] for better gradient flow
        # (Sigmoid causes saturation and vanishing gradients for long sequences)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x