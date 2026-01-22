# eval_metrics.py 完整修改
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

def calculate_decoding_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    计算解码阶段的核心评估指标（优化GPU利用率）
    :param pred: 解码预测值 (B, H, W) - GPU张量
    :param target: 真实值 (B, H, W) - GPU张量
    :param mask: 有效区域掩码 (B, H, W) - GPU张量
    :return: 指标字典
    """
    # 第一步：在GPU上完成掩码筛选（减少数据传输量）
    mask_bool = mask.bool()
    pred_masked = pred[mask_bool]
    target_masked = target[mask_bool]
    
    # 第二步：仅传输筛选后的数据到CPU（大幅减少IO）
    pred_flat = pred_masked.cpu().numpy()
    target_flat = target_masked.cpu().numpy()
    
    # 基础误差指标（批量计算）
    abs_diff = np.abs(pred_flat - target_flat)
    mae = np.mean(abs_diff)
    mse = np.mean((pred_flat - target_flat) **2)
    rmse = np.sqrt(mse)
    
    # 结构/相关性指标（处理空值）
    if len(pred_flat) == 0:
        ssim_score = 0.0
        corr = 0.0
    else:
        ssim_score = ssim(
            pred_flat.reshape(1, -1), 
            target_flat.reshape(1, -1), 
            data_range=target_flat.max() - target_flat.min()
        )
        corr, _ = pearsonr(pred_flat, target_flat)
    
    # 相对误差（避免除0）
    abs_target = np.abs(target_flat)
    rmae = np.mean(abs_diff / (abs_target + 1e-6))
    
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
    if not metrics_list:  # 空列表保护
        return {}
    agg = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        agg[f"{key}_mean"] = round(np.mean(values), 4)
        agg[f"{key}_std"] = round(np.std(values), 4)
    return agg