import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ResNetDecoder(nn.Module):
    """解码器：将1D潜向量还原为2D地图"""
    def __init__(self, config: Config):
        super().__init__()
        self.init_h = config.IMG_SIZE // 32
        self.flat_dim = 256 * self.init_h * self.init_h
        
        # 潜向量映射回特征图维度
        self.fc = nn.Linear(config.LATENT_DIM, self.flat_dim)
        
        # 转置卷积上采样
        self.up_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
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
        x = x.view(x.size(0), 256, self.init_h, self.init_h)
        # 上采样
        x = self.up_layers(x)
        # 还原单通道地图
        return self.final_conv(x)