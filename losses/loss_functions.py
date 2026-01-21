import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class SharpnessAwareLoss(nn.Module):
    """带梯度/TopK约束的损失函数"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.l1 = nn.L1Loss(reduction='none')

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

    def forward(self, pred, target, mask):
        """总损失：L1基础损失 + TopK损失 + 梯度损失"""
        # 基础L1损失（掩码加权）
        abs_diff = self.l1(pred, target) * mask
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
        loss_grad = self.gradient_loss(pred, target, mask)
        
        # 加权求和
        return (self.config.W_BASE_L1 * loss_base + 
                self.config.W_TOPK * loss_topk + 
                self.config.W_GRAD * loss_grad)