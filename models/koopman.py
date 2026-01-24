import torch
import torch.nn as nn
from config import Config
from models.encoder import ResNetEncoder
from models.decoder import ResNetDecoder

class SchurKoopmanLayer(nn.Module):
    """
    基于舒尔分解 (Schur Decomposition) 的 Koopman 层
    K = U T U^T
    - U: 正交矩阵 (Orthogonal)，负责坐标变换 (旋转)
    - T: 上三角矩阵 (Upper Triangular)，对角线控制稳定性，上三角控制短期增长
    """
    def __init__(self, dim, max_scale=0.99):
        super().__init__()
        self.dim = dim
        self.max_scale = max_scale
        
        # 构造正交矩阵 U 的参数（斜对称矩阵）
        self.U_param = nn.Parameter(torch.randn(dim, dim) * 0.01)
        
        # 构造上三角矩阵 T 的参数
        self.lambda_proxies = nn.Parameter(torch.randn(dim))  # 对角线部分（特征值）
        self.T_upper_param = nn.Parameter(torch.randn(dim, dim) * 0.01)  # 上三角部分

    def get_K(self):
        # 步骤 A: 构造正交基 U
        skew_symmetric = self.U_param - self.U_param.t()
        U = torch.linalg.matrix_exp(skew_symmetric)
        
        # 步骤 B: 构造上三角矩阵 T
        eigenvalues = torch.tanh(self.lambda_proxies) * self.max_scale  # 控制特征值在(-max_scale, max_scale)
        T_diag = torch.diag(eigenvalues)
        T_strict_upper = torch.triu(self.T_upper_param, diagonal=1)  # 严格上三角部分
        T = T_diag + T_strict_upper
        
        # 步骤 C: 重组 K = U T U^T
        K = U @ T @ U.t()
        return K

    def forward(self, x):
        K = self.get_K()
        return torch.nn.functional.linear(x, K)  # 等价于 x @ K.T

class ControlledKoopmanModel(nn.Module):
    """带气象控制的Koopman模型：核心动力学逻辑"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = ResNetEncoder(config)
        self.decoder = ResNetDecoder(config)
        
        # Koopman演化矩阵
        self.K = nn.Linear(config.LATENT_DIM, config.LATENT_DIM, bias=False)
        
        # 残差Koopman权重（0-1，控制线性vs残差的比例）
        # 当alpha=1时：z_next = K(z) （标准Koopman）
        # 当alpha=0时：z_next = z + ΔK(z) （纯残差，保留极值）
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5，让模型自己学习最优比例
        
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
        
        # 混合Koopman：线性 + 残差
        # 当alpha大时，更倾向于标准Koopman（z_next = K(z))
        # 当alpha小时，更倾向于残差形式（z_next = z + ΔK(z))
        alpha_clipped = torch.sigmoid(self.alpha)  # 限制在[0,1]
        z_linear = self.K(z_t)
        z_residual = z_t + (z_linear - z_t)  # 残差形式，保留原值
        z_next_pred = alpha_clipped * z_linear + (1 - alpha_clipped) * z_residual
        
        # 添加控制项
        z_next_pred = z_next_pred + self.compute_control(u_current)
        
        # 解码残差
        delta = self.decoder(z_next_pred)
        # 物理残差更新
        x_next_pred = torch.clamp(x_current + delta, min=0.0, max=1.0)
        return x_next_pred, z_t, z_next_pred

    def predict_future(self, x_start, u_sequence, steps):
        """多步预测未来地图"""
        preds = []
        x_curr = x_start
        z_curr = self.encoder(x_curr)
        
        alpha_clipped = torch.sigmoid(self.alpha)
        
        for t in range(steps):
            u_t = u_sequence[t].unsqueeze(0).to(x_start.device)
            # 混合动力学演化
            z_linear = self.K(z_curr)
            z_residual = z_curr + (z_linear - z_curr)
            z_curr = alpha_clipped * z_linear + (1 - alpha_clipped) * z_residual
            z_curr = z_curr + self.compute_control(u_t)
            
            # 解码+残差更新
            delta = self.decoder(z_curr)
            x_next = torch.clamp(x_curr + delta, min=0.0, max=1.0)
            
            preds.append(x_next)
            x_curr = x_next
        
        return preds