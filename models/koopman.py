import torch
import torch.nn as nn
from config import Config
from models.encoder import ResNetEncoder
from models.decoder import ResNetDecoder

class ControlledKoopmanModel(nn.Module):
    """带气象控制的Koopman模型：核心动力学逻辑"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = ResNetEncoder(config)
        self.decoder = ResNetDecoder(config)
        
        # Koopman演化矩阵
        self.K = nn.Linear(config.LATENT_DIM, config.LATENT_DIM, bias=False)
        
        # 控制矩阵（按开关初始化）
        if config.USE_METEO_CONTROL:
            self.B = nn.Linear(config.METEO_DIM, config.LATENT_DIM, bias=False)
        else:
            self.B = None

    def compute_control(self, u_current):
        """计算控制项（兼容禁用控制的情况）"""
        if self.B is not None:
            return self.B(u_current)
        return 0.0

    def forward(self, x_current, u_current):
        """单次前向：从当前地图预测下一时刻地图"""
        # 编码为1D向量
        z_t = self.encoder(x_current)
        # Koopman动力学演化（K*z + B*u）
        z_next_pred = self.K(z_t) + self.compute_control(u_current)
        # 解码残差
        delta = self.decoder(z_next_pred)
        # 物理残差更新
        x_next_pred = torch.relu(x_current + delta)
        return x_next_pred, z_t, z_next_pred

    def predict_future(self, x_start, u_sequence, steps):
        """多步预测未来地图"""
        preds = []
        x_curr = x_start
        z_curr = self.encoder(x_curr)
        
        for t in range(steps):
            u_t = u_sequence[t].unsqueeze(0).to(x_start.device)
            # 动力学演化
            z_curr = self.K(z_curr) + self.compute_control(u_t)
            # 解码+残差更新
            delta = self.decoder(z_curr)
            x_next = torch.relu(x_curr + delta)
            
            preds.append(x_next)
            x_curr = x_next
        
        return preds