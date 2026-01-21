import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class SpatialAttention(nn.Module):
    """空间注意力模块：增强关键区域特征"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = torch.sigmoid(self.conv(x_cat))
        return x * attention_map

class ResidualBlock(nn.Module):
    """残差块：基础卷积单元"""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNetEncoder(nn.Module):
    """ResNet编码器：将2D地图编码为1D潜向量"""
    def __init__(self, config: Config):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = ResidualBlock(32, 32, 1)
        self.layer2 = ResidualBlock(32, 64, 2)
        self.layer3 = ResidualBlock(64, 128, 2)
        self.layer4 = ResidualBlock(128, 256, 2)
        self.att = SpatialAttention()
        
        # 计算扁平化维度
        final_h = config.IMG_SIZE // 32 
        self.flat_dim = 256 * final_h * final_h 
        
        # 全连接层：映射到潜变量维度
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, config.LATENT_DIM),
            nn.LayerNorm(config.LATENT_DIM),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 卷积特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 注意力增强
        x = self.att(x)
        # 扁平化+映射为1D向量
        z = self.fc(x)
        return z