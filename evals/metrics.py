import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算均方误差 MSE"""
    return F.mse_loss(pred, target).item()

def calculate_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算均方根误差 RMSE"""
    return torch.sqrt(F.mse_loss(pred, target)).item()

def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算平均绝对误差 MAE"""
    return F.l1_loss(pred, target).item()

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> float:
    """计算峰值信噪比 PSNR（需将张量转为numpy）"""
    pred_np = pred.squeeze().cpu().detach().numpy()
    target_np = target.squeeze().cpu().detach().numpy()
    return peak_signal_noise_ratio(target_np, pred_np, data_range=data_range)

def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: int = 255) -> float:
    """计算结构相似性 SSIM（需将张量转为numpy）"""
    pred_np = pred.squeeze().cpu().detach().numpy()
    target_np = target.squeeze().cpu().detach().numpy()
    return structural_similarity(target_np, pred_np, data_range=data_range)

def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """一键计算所有评估指标"""
    metrics = {
        "MSE": calculate_mse(pred, target),
        "RMSE": calculate_rmse(pred, target),
        "MAE": calculate_mae(pred, target),
        "PSNR": calculate_psnr(pred, target),
        "SSIM": calculate_ssim(pred, target)
    }
    return metrics