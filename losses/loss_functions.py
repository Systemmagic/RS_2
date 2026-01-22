import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class SharpnessAwareLoss(nn.Module):
    """带梯度/TopK约束及Koopman相关损失的综合损失函数"""
    def __init__(self, config: Config, koopman_layer=None, encoder=None, decoder=None, koopman_model=None):
        super().__init__()
        self.config = config
        self.l1 = nn.L1Loss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')  # 新增mse损失
        
        # 保存Koopman层及编解码器引用
        self.koopman_layer = koopman_layer  # SchurKoopmanLayer实例
        self.encoder = encoder
        self.decoder = decoder
        self.koopman_model = koopman_model  # 完整模型引用（用于获取K矩阵）

    def gradient_loss(self, pred, target, mask):
        """梯度一致性损失：保证预测图的空间梯度与真实图一致"""
        # 计算y方向梯度
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        # 计算x方向梯度
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        # 掩码：仅计算有效像素的梯度损失
        mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        
        loss_dy = torch.abs(dy_pred - dy_target) * mask_y
        loss_dx = torch.abs(dx_pred - dx_target) * mask_x
        
        return (loss_dy.sum() + loss_dx.sum()) / (mask_y.sum() + mask_x.sum() + 1e-8)
    
    def get_K_matrix(self):
        """获取Koopman演化矩阵K"""
        if self.koopman_layer is not None and hasattr(self.koopman_layer, 'get_K'):
            # 舒尔分解方式：从SchurKoopmanLayer动态生成
            return self.koopman_layer.get_K()
        elif self.koopman_model is not None and hasattr(self.koopman_model, 'K'):
            # 标准方式：从模型的K线性层获取权重矩阵
            return self.koopman_model.K.weight
        else:
            return None

    def forward(self, x_seq, pred_seq, mask):
        """
        总损失计算
        x_seq: 原始序列 [batch, T, channels, height, width]，包含真实状态x₁到x_T
        pred_seq: 预测序列 [batch, Sp, channels, height, width]，包含预测的x₂到x_{Sp+1}
        mask: 空间掩码 [batch, 1, height, width]
        支持Koopman损失计算
        """
        # --------------------------
        # 1. 原有损失项
        # --------------------------
        # 基础L1损失（使用最后一步预测与目标的差异）
        target_last = x_seq[:, -1, ...]  # 最后一个真实状态作为目标
        pred_last = pred_seq[:, -1, ...]  # 最后一个预测状态
        abs_diff = self.l1(pred_last, target_last) * mask
        loss_base = abs_diff.sum() / (mask.sum() + 1e-8)
        
        # TopK损失：聚焦误差最大的像素
        valid_pixels = abs_diff[mask > 0.5]
        if valid_pixels.numel() > 0:
            k = int(valid_pixels.numel() * self.config.TOPK_RATIO)
            k = max(k, 1)
            topk_vals, _ = torch.topk(valid_pixels, k)
            loss_topk = topk_vals.mean()
        else:
            loss_topk = 0.0
        
        # 梯度损失
        loss_grad = self.gradient_loss(pred_last, target_last, mask)
        
        # --------------------------
        # 2. 新增Koopman相关损失项
        # --------------------------
        loss_recon = 0.0
        loss_lin = 0.0
        loss_pred = 0.0
        loss_inf = 0.0
        loss_l2_reg = 0.0
        
        # 仅当编码器、解码器、以及K矩阵都可用时，才计算Koopman损失
        K = self.get_K_matrix()
        if self.encoder is not None and self.decoder is not None and K is not None:
            # 2.1 重建损失 L_recon：编码器+解码器的重建误差
            x1 = x_seq[:, 0, ...]  # 初始状态x₁
            y1 = self.encoder(x1)  # 编码到不变子空间 y₁=φ(x₁)
            x1_recon = self.decoder(y1)  # 解码重建 x₁_recon=φ⁻¹(y₁)
            recon_diff = self.mse(x1_recon, x1) * mask
            loss_recon = recon_diff.sum() / (mask.sum() + 1e-8)
            
            # 2.2 线性动力学损失 L_lin：不变子空间中的线性性约束（优化版）
            T = x_seq.shape[1]  # 序列长度
            if T > 1:
                y_list = [self.encoder(x_seq[:, t, ...]) for t in range(T)]  # 所有状态的编码
                y_current = y1.flatten(1)  # 初始化当前状态为y1（展平为向量）
                lin_losses = []
                
                for m in range(1, T):
                    # 核心优化：递归计算 K^m y1 = K*(K^(m-1) y1)，避免矩阵幂运算
                    # 批量矩阵乘法：(batch, dim) @ (dim, dim) -> (batch, dim)
                    y_current = torch.matmul(y_current, K.t())  # 等价于 K @ y_current（因维度适配）
                    
                    # 恢复形状并计算与真实编码的误差
                    y_pred_reshaped = y_current.view_as(y_list[m])
                    lin_diff = self.mse(y_pred_reshaped, y_list[m])
                    lin_losses.append(lin_diff.mean())
                
                loss_lin = torch.mean(torch.stack(lin_losses))
            else:
                loss_lin = 0.0
            
            # 2.3 预测损失 L_pred：原空间中的多步预测误差（同样采用递归优化）
            Sp = pred_seq.shape[1]  # 预测步数
            pred_losses = []
            if Sp > 0:
                # 重新初始化用于预测的y_current
                y_pred_current = y1.flatten(1)
                for m in range(Sp):
                    # 递归计算 K^(m+1) y1
                    y_pred_current = torch.matmul(y_pred_current, K.t())
                    
                    # 解码得到原空间预测
                    x_pred_m = self.decoder(y_pred_current.view_as(y1))
                    
                    # 获取对应目标（超过序列长度则用最后一个状态）
                    target_m = x_seq[:, m+1, ...] if (m+1) < T else x_seq[:, -1, ...]
                    
                    # 计算预测误差
                    pred_diff = self.mse(x_pred_m, target_m) * mask
                    pred_losses.append(pred_diff.sum() / (mask.sum() + 1e-8))
                
                loss_pred = torch.mean(torch.stack(pred_losses)) / Sp
            else:
                loss_pred = 0.0
            
            # 2.4 L∞损失：防止异常值干扰
            loss_inf = torch.norm(abs_diff, p=float('inf')) if valid_pixels.numel() > 0 else 0.0
            
            # 2.5 L2正则项：防止过拟合（Koopman矩阵的L2范数）
            loss_l2_reg = torch.norm(K, p=2)
        
        # --------------------------
        # 总损失加权求和
        # --------------------------
        total_loss = (
            # 原有损失
            self.config.W_BASE_L1 * loss_base +
            self.config.W_TOPK * loss_topk +
            self.config.W_GRAD * loss_grad +
            # 新增Koopman损失
            self.config.ALPHA1 * (loss_recon + loss_pred) +
            loss_lin +
            self.config.ALPHA2 * loss_inf +
            self.config.ALPHA3 * loss_l2_reg
        )
        
        return total_loss