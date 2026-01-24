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
        mask: 空间掩码 [batch, 1, height, width] 或 [batch, height, width]
        支持Koopman损失计算
        """
        # --------------------------
        # 0. 掩码维度标准化
        # --------------------------
        if mask.dim() == 3:
            # [batch, H, W] -> [batch, 1, H, W]
            mask = mask.unsqueeze(1)
        
        # --------------------------
        # 1. 原有损失项
        # --------------------------
        # 基础L1损失（使用最后一步预测与目标的差异）
        target_last = x_seq[:, -1, ...]  # 最后一个真实状态作为目标 [batch, C, H, W]
        pred_last = pred_seq[:, -1, ...]  # 最后一个预测状态 [batch, C, H, W]
        abs_diff = self.l1(pred_last, target_last) * mask  # [batch, C, H, W] * [batch, 1, H, W]
        loss_base = abs_diff.sum() / (mask.sum() + 1e-8)
        
        # TopK损失：聚焦误差最大的像素
        # 展平并选择有效像素
        abs_diff_flat = abs_diff.view(abs_diff.shape[0], -1)  # [batch, C*H*W]
        mask_flat = mask.view(mask.shape[0], -1)  # [batch, 1*H*W]
        valid_pixels = abs_diff_flat[mask_flat > 0.5]
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
            x1 = x_seq[:, 0, ...]  # 初始状态x₁ [batch, C, H, W]
            y1 = self.encoder(x1)  # 编码到不变子空间 y₁=φ(x₁)
            x1_recon = self.decoder(y1)  # 解码重建 x₁_recon=φ⁻¹(y₁) [batch, C, H, W]
            recon_diff = self.mse(x1_recon, x1) * mask  # [batch, C, H, W] * [batch, 1, H, W]
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
                    
                    # 计算预测误差 [batch, C, H, W] * [batch, 1, H, W]
                    pred_diff = self.mse(x_pred_m, target_m) * mask
                    pred_losses.append(pred_diff.sum() / (mask.sum() + 1e-8))
                
                loss_pred = torch.mean(torch.stack(pred_losses)) / Sp if pred_losses else 0.0
            else:
                loss_pred = 0.0
            
            # 2.4 L∞损失：防止异常值干扰
            # 只计算掩码有效区域的范数
            abs_diff_masked = abs_diff[mask > 0.5]
            loss_inf = torch.norm(abs_diff_masked, p=float('inf')) if abs_diff_masked.numel() > 0 else 0.0
            
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


class WeightedMSELoss(nn.Module):
    """加权MSE损失，强调极值样本"""
    def __init__(self, extreme_weight=2.0, global_min=0.0, global_max=229.13):
        super().__init__()
        self.extreme_weight = extreme_weight
        self.global_min = global_min
        self.global_max = global_max
        
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: 预测值 [batch, C, H, W]
            target: 真实值 [batch, C, H, W]
            mask: 掩码 [batch, 1, H, W] 或 [batch, C, H, W]
        """
        # 归一化到[0,1]
        norm_target = (target - self.global_min) / (self.global_max - self.global_min + 1e-8)
        
        # 计算分位数阈值
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            valid_vals = norm_target[mask > 0.5]
        else:
            valid_vals = norm_target.reshape(-1)
        
        if valid_vals.numel() > 0:
            q90 = torch.quantile(valid_vals, 0.90)
            q10 = torch.quantile(valid_vals, 0.10)
        else:
            q90 = 0.9
            q10 = 0.1
        
        # 创建权重矩阵
        weights = torch.ones_like(target)
        weights[(norm_target > q90) | (norm_target < q10)] = self.extreme_weight
        
        # 计算加权MSE
        se = (pred - target) ** 2
        if mask is not None:
            weighted_loss = (se * weights * mask).sum() / (mask.sum() + 1e-8)
        else:
            weighted_loss = (se * weights).mean()
        
        return weighted_loss


class GradientConstraintLoss(nn.Module):
    """梯度约束损失，强制预测梯度接近真实梯度"""
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: 预测值 [batch, C, H, W]
            target: 真实值 [batch, C, H, W]
            mask: 掩码 [batch, 1, H, W] 或 [batch, C, H, W]
        """
        # 计算梯度 (沿H和W维度)
        pred_gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]   # [batch, C, H-1, W]
        pred_gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]   # [batch, C, H, W-1]
        
        target_gy = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_gx = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # 梯度幅值（调和到相同尺寸）
        pred_grad = torch.sqrt(pred_gx[:, :, :-1, :]**2 + pred_gy[:, :, :, :-1]**2 + 1e-6)
        target_grad = torch.sqrt(target_gx[:, :, :-1, :]**2 + target_gy[:, :, :, :-1]**2 + 1e-6)
        
        # 梯度损失
        grad_loss = F.mse_loss(pred_grad, target_grad)
        
        return self.weight * grad_loss