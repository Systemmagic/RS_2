"""
每日预测评估模块 - 为每一天的预测生成散点图和评估指标
计算指标：R², IA（Index of Agreement）、RMSE、MAE
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import linregress, pearsonr


def calculate_r2(y_true, y_pred):
    """计算R² (决定系数)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def calculate_ia(y_true, y_pred):
    """
    计算 IA (Index of Agreement - 一致性指数)
    IA = 1 - (Σ(pred-obs)²) / (Σ(|pred-mean(obs)| + |obs-mean(obs)|)²)
    范围 [0, 1]，1表示完美预测
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    mean_obs = np.mean(y_true)
    denominator = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2)
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    
    return 1.0 - (numerator / denominator)


def calculate_rmse(y_true, y_pred):
    """计算 RMSE (Root Mean Square Error)"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """计算 MAE (Mean Absolute Error)"""
    return np.mean(np.abs(y_true - y_pred))


def calculate_daily_metrics(y_true, y_pred, mask=None):
    """
    计算单日的所有评估指标
    
    Args:
        y_true: 真实值 [H, W] 或 [C, H, W]
        y_pred: 预测值，形状同y_true
        mask: 有效区域掩码 [H, W]
    
    Returns:
        dict: 包含R2, IA, RMSE, MAE的指标字典
    """
    # 展平处理
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 应用掩码
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        y_true_flat = y_true_flat[mask_flat]
        y_pred_flat = y_pred_flat[mask_flat]
    
    # 移除NaN值
    valid_idx = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_flat = y_true_flat[valid_idx]
    y_pred_flat = y_pred_flat[valid_idx]
    
    if len(y_true_flat) == 0:
        return {
            'r2': 0.0,
            'ia': 0.0,
            'rmse': np.inf,
            'mae': np.inf,
            'correlation': 0.0,
            'n_valid_pixels': 0
        }
    
    # 计算各指标
    metrics = {
        'r2': calculate_r2(y_true_flat, y_pred_flat),
        'ia': calculate_ia(y_true_flat, y_pred_flat),
        'rmse': calculate_rmse(y_true_flat, y_pred_flat),
        'mae': calculate_mae(y_true_flat, y_pred_flat),
        'n_valid_pixels': len(y_true_flat)
    }
    
    # 计算相关系数
    if len(y_true_flat) > 1:
        corr, _ = pearsonr(y_true_flat, y_pred_flat)
        metrics['correlation'] = corr
    else:
        metrics['correlation'] = 0.0
    
    return metrics


def plot_daily_scatter(y_true, y_pred, metrics, day_idx, output_dir, mask=None):
    """
    为单日预测绘制散点图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        metrics: 评估指标字典
        day_idx: 日期索引
        output_dir: 输出目录
        mask: 有效区域掩码
    """
    # 展平处理
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 应用掩码
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        y_true_flat = y_true_flat[mask_flat]
        y_pred_flat = y_pred_flat[mask_flat]
    
    # 移除NaN值
    valid_idx = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_flat = y_true_flat[valid_idx]
    y_pred_flat = y_pred_flat[valid_idx]
    
    if len(y_true_flat) == 0:
        print(f"Day {day_idx}: No valid data for plotting")
        return
    
    # 创建散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=20, c='steelblue', edgecolors='navy', linewidth=0.5)
    
    # 完美预测线
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 线性拟合线
    if len(y_true_flat) > 1:
        z = np.polyfit(y_true_flat, y_pred_flat, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min_val, max_val, 100)
        ax.plot(x_fit, p(x_fit), 'g-', linewidth=2, label=f'Linear Fit (slope={z[0]:.3f})')
    
    # 标题和标签
    title = f'Day {day_idx + 1} - Prediction Scatter Plot\n'
    title += f'R² = {metrics["r2"]:.4f}, IA = {metrics["ia"]:.4f}\n'
    title += f'RMSE = {metrics["rmse"]:.4f}, MAE = {metrics["mae"]:.4f}\n'
    title += f'Correlation = {metrics["correlation"]:.4f}, N = {metrics["n_valid_pixels"]}'
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('True Value', fontsize=11)
    ax.set_ylabel('Predicted Value', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'scatter_day_{day_idx + 1:02d}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def plot_daily_heatmap(y_true, y_pred, day_idx, output_dir, mask=None):
    """
    绘制单日的误差热力图
    
    Args:
        y_true: 真实值 [H, W]
        y_pred: 预测值 [H, W]
        day_idx: 日期索引
        output_dir: 输出目录
        mask: 有效区域掩码
    """
    # 计算误差
    error = np.abs(y_true - y_pred)
    
    # 应用掩码
    if mask is not None:
        error_masked = np.where(mask > 0.5, error, np.nan)
    else:
        error_masked = error
    
    # 创建3个子图：真实值、预测值、误差
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 真实值
    im1 = axes[0].imshow(y_true, cmap='viridis')
    axes[0].set_title(f'Day {day_idx + 1} - True Value', fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='PM2.5')
    
    # 预测值
    im2 = axes[1].imshow(y_pred, cmap='viridis')
    axes[1].set_title(f'Day {day_idx + 1} - Predicted Value', fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], label='PM2.5')
    
    # 误差
    im3 = axes[2].imshow(error_masked, cmap='hot')
    axes[2].set_title(f'Day {day_idx + 1} - Absolute Error', fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[2], label='|Error|')
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'heatmap_day_{day_idx + 1:02d}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def evaluate_daily_predictions(pred_list, true_list, mask_list, output_dir):
    """
    评估每一天的预测结果，生成散点图和指标
    
    Args:
        pred_list: 预测值列表，每个元素形状 [H, W] 或 [C, H, W]
        true_list: 真实值列表，形状同pred_list
        mask_list: 掩码列表，每个元素形状 [H, W]
        output_dir: 输出目录，用于保存图片和报告
    
    Returns:
        dict: 包含所有日期的评估结果
    """
    n_days = len(pred_list)
    daily_metrics = []
    scatter_paths = []
    heatmap_paths = []
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    scatter_dir = os.path.join(output_dir, 'scatter_plots')
    heatmap_dir = os.path.join(output_dir, 'heatmap_plots')
    
    print(f"\n{'='*80}")
    print(f"{'Daily Prediction Evaluation':^80}")
    print(f"{'='*80}")
    print(f"{'Day':<6} {'R2':<10} {'IA':<10} {'RMSE':<10} {'MAE':<10} {'Corr':<10} {'N_Pixels':<12}")
    print(f"{'-'*80}")
    
    for day_idx in range(n_days):
        pred = pred_list[day_idx]
        true = true_list[day_idx]
        mask = mask_list[day_idx] if day_idx < len(mask_list) else None
        
        # 如果是多通道，取第一通道
        if pred.ndim == 3:
            pred = pred[0]
        if true.ndim == 3:
            true = true[0]
        
        # 计算指标
        metrics = calculate_daily_metrics(true, pred, mask)
        daily_metrics.append(metrics)
        
        # 打印指标
        print(f"Day {day_idx + 1:<3} {metrics['r2']:<10.4f} {metrics['ia']:<10.4f} "
              f"{metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} "
              f"{metrics['correlation']:<10.4f} {metrics['n_valid_pixels']:<12d}")
        
        # 绘制散点图
        scatter_path = plot_daily_scatter(true, pred, metrics, day_idx, scatter_dir, mask)
        scatter_paths.append(scatter_path)
        
        # 绘制热力图
        heatmap_path = plot_daily_heatmap(true, pred, day_idx, heatmap_dir, mask)
        heatmap_paths.append(heatmap_path)
    
    # 计算统计信息
    r2_values = [m['r2'] for m in daily_metrics]
    ia_values = [m['ia'] for m in daily_metrics]
    rmse_values = [m['rmse'] for m in daily_metrics]
    mae_values = [m['mae'] for m in daily_metrics]
    
    print(f"{'-'*80}")
    print(f"{'Mean':<6} {np.mean(r2_values):<10.4f} {np.mean(ia_values):<10.4f} "
          f"{np.mean(rmse_values):<10.4f} {np.mean(mae_values):<10.4f} {' ':<10} {' ':<12}")
    print(f"{'Std':<6} {np.std(r2_values):<10.4f} {np.std(ia_values):<10.4f} "
          f"{np.std(rmse_values):<10.4f} {np.std(mae_values):<10.4f} {' ':<10} {' ':<12}")
    print(f"{'='*80}\n")
    
    # 生成汇总图表
    plot_metrics_summary(daily_metrics, output_dir)
    
    return {
        'daily_metrics': daily_metrics,
        'scatter_paths': scatter_paths,
        'heatmap_paths': heatmap_paths,
        'summary': {
            'r2_mean': float(np.mean(r2_values)),
            'r2_std': float(np.std(r2_values)),
            'ia_mean': float(np.mean(ia_values)),
            'ia_std': float(np.std(ia_values)),
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'mae_mean': float(np.mean(mae_values)),
            'mae_std': float(np.std(mae_values))
        }
    }


def plot_metrics_summary(daily_metrics, output_dir):
    """绘制指标总结图表"""
    n_days = len(daily_metrics)
    days = list(range(1, n_days + 1))
    
    r2_values = [m['r2'] for m in daily_metrics]
    ia_values = [m['ia'] for m in daily_metrics]
    rmse_values = [m['rmse'] for m in daily_metrics]
    mae_values = [m['mae'] for m in daily_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R² 曲线
    axes[0, 0].plot(days, r2_values, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_title('R² Score by Day', fontweight='bold', fontsize=12)
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # IA 曲线
    axes[0, 1].plot(days, ia_values, 'g-s', linewidth=2, markersize=8)
    axes[0, 1].set_title('Agreement Index (IA) by Day', fontweight='bold', fontsize=12)
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('IA')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # RMSE 曲线
    axes[1, 0].plot(days, rmse_values, 'r-^', linewidth=2, markersize=8)
    axes[1, 0].set_title('RMSE by Day', fontweight='bold', fontsize=12)
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE 曲线
    axes[1, 1].plot(days, mae_values, 'm-d', linewidth=2, markersize=8)
    axes[1, 1].set_title('MAE by Day', fontweight='bold', fontsize=12)
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'metrics_summary.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Metrics summary saved to {fig_path}")
