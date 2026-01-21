import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def calculate_decoding_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    计算解码阶段的核心评估指标
    :param pred: 解码预测值 (B, H, W)
    :param target: 真实值 (B, H, W)
    :param mask: 有效区域掩码 (B, H, W)
    :return: 指标字典
    """
    # 仅计算有效区域（mask=1）
    pred_flat = pred[mask.bool()].cpu().numpy()
    target_flat = target[mask.bool()].cpu().numpy()
    
    # 基础误差指标
    mae = np.mean(np.abs(pred_flat - target_flat))
    mse = np.mean((pred_flat - target_flat) **2)
    rmse = np.sqrt(mse)
    
    # 结构/相关性指标
    ssim_score = ssim(pred_flat.reshape(1, -1), target_flat.reshape(1, -1), data_range=target_flat.max() - target_flat.min())
    corr, _ = pearsonr(pred_flat, target_flat) if len(pred_flat) > 0 else (0.0, 0.0)
    
    # 相对误差
    abs_target = np.abs(target_flat)
    rmae = np.mean(np.abs(pred_flat - target_flat) / (abs_target + 1e-6))  # 避免除0
    
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "SSIM": round(ssim_score, 4),
        "PearsonCorr": round(corr, 4),
        "RMAE": round(rmae, 4)
    }

def aggregate_metrics(metrics_list: list) -> dict:
    """聚合多批次/多窗口的指标（求均值+标准差）"""
    agg = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        agg[f"{key}_mean"] = round(np.mean(values), 4)
        agg[f"{key}_std"] = round(np.std(values), 4)
    return agg