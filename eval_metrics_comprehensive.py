"""
科研规范的综合评估模块
========================

同时计算原始物理单位数据和归一化数据的指标
符合国际论文发表标准 (气象、遥感、环保领域)

指标包括：
  - R², IA, Pearson相关系数
  - RMSE (原始单位), MAE (原始单位)
  - NRMSE (相对误差%), MAPE (百分比误差%)
  - Bias (系统偏差)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


def denormalize_data(data_norm, global_min, global_max):
    """反归一化数据到原始物理单位"""
    return data_norm * (global_max - global_min) + global_min


def calculate_r2(y_true, y_pred):
    """R² (决定系数) - 无单位"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def calculate_ia(y_true, y_pred):
    """IA (一致性指数) - 范围[0,1]"""
    numerator = np.sum((y_true - y_pred) ** 2)
    mean_obs = np.mean(y_true)
    denominator = np.sum((np.abs(y_pred - mean_obs) + np.abs(y_true - mean_obs)) ** 2)
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return 1.0 - (numerator / denominator)


def calculate_rmse(y_true, y_pred):
    """RMSE (均方根误差) - 有单位"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """MAE (平均绝对误差) - 有单位"""
    return np.mean(np.abs(y_true - y_pred))


def calculate_nrmse(y_true, y_pred):
    """NRMSE (归一化RMSE) - %"""
    rmse = calculate_rmse(y_true, y_pred)
    mean_obs = np.mean(y_true)
    if mean_obs == 0:
        return 0.0
    return (rmse / mean_obs) * 100


def calculate_mape(y_true, y_pred):
    """MAPE (平均绝对百分比误差) - %"""
    # 避免除以零
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    mape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return np.mean(mape) * 100


def calculate_bias(y_true, y_pred):
    """Bias (系统偏差) - 有单位"""
    return np.mean(y_pred - y_true)


def calculate_correlation(y_true, y_pred):
    """Pearson相关系数"""
    if len(y_true) > 1:
        corr, _ = pearsonr(y_true, y_pred)
        return corr
    return 0.0


def calculate_comprehensive_metrics(y_true_norm, y_pred_norm, global_min, global_max, mask=None):
    """
    计算完整评估指标 - 同时返回原始单位和归一化结果
    
    Args:
        y_true_norm: 真实值 [0,1]
        y_pred_norm: 预测值 [0,1]
        global_min: 原始数据最小值 (e.g., 0.00)
        global_max: 原始数据最大值 (e.g., 229.13)
        mask: 有效区域掩码 [H, W]
    
    Returns:
        dict: 包含原始单位和归一化指标的字典
    """
    # 展平并应用掩码
    y_true_flat_norm = y_true_norm.flatten()
    y_pred_flat_norm = y_pred_norm.flatten()
    
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        y_true_flat_norm = y_true_flat_norm[mask_flat]
        y_pred_flat_norm = y_pred_flat_norm[mask_flat]
    
    # 移除NaN值
    valid_idx = ~(np.isnan(y_true_flat_norm) | np.isnan(y_pred_flat_norm))
    y_true_flat_norm = y_true_flat_norm[valid_idx]
    y_pred_flat_norm = y_pred_flat_norm[valid_idx]
    
    if len(y_true_flat_norm) == 0:
        return None
    
    # 反归一化到原始物理单位
    y_true_flat_orig = denormalize_data(y_true_flat_norm, global_min, global_max)
    y_pred_flat_orig = denormalize_data(y_pred_flat_norm, global_min, global_max)
    
    # ================== 原始单位指标 (科研论文必须) ==================
    r2_orig = calculate_r2(y_true_flat_orig, y_pred_flat_orig)
    ia_orig = calculate_ia(y_true_flat_orig, y_pred_flat_orig)
    rmse_orig = calculate_rmse(y_true_flat_orig, y_pred_flat_orig)
    mae_orig = calculate_mae(y_true_flat_orig, y_pred_flat_orig)
    nrmse_orig = calculate_nrmse(y_true_flat_orig, y_pred_flat_orig)
    mape_orig = calculate_mape(y_true_flat_orig, y_pred_flat_orig)
    bias_orig = calculate_bias(y_true_flat_orig, y_pred_flat_orig)
    corr = calculate_correlation(y_true_flat_orig, y_pred_flat_orig)
    
    # ================== 归一化指标 (用于对比) ==================
    r2_norm = calculate_r2(y_true_flat_norm, y_pred_flat_norm)
    ia_norm = calculate_ia(y_true_flat_norm, y_pred_flat_norm)
    rmse_norm = calculate_rmse(y_true_flat_norm, y_pred_flat_norm)
    mae_norm = calculate_mae(y_true_flat_norm, y_pred_flat_norm)
    
    metrics = {
        # 原始物理单位 (主要用于论文)
        'r2_original': r2_orig,
        'ia_original': ia_orig,
        'rmse_original': rmse_orig,
        'mae_original': mae_orig,
        'nrmse_percent': nrmse_orig,
        'mape_percent': mape_orig,
        'bias_original': bias_orig,
        'correlation': corr,
        
        # 归一化指标 (参考对比)
        'r2_normalized': r2_norm,
        'ia_normalized': ia_norm,
        'rmse_normalized': rmse_norm,
        'mae_normalized': mae_norm,
        
        # 统计信息
        'mean_obs': np.mean(y_true_flat_orig),
        'mean_pred': np.mean(y_pred_flat_orig),
        'std_obs': np.std(y_true_flat_orig),
        'std_pred': np.std(y_pred_flat_orig),
        'n_valid_pixels': len(y_true_flat_norm),
        'min_obs': np.min(y_true_flat_orig),
        'max_obs': np.max(y_true_flat_orig),
    }
    
    return metrics


def print_metrics_table(daily_metrics, output_dir):
    """
    打印符合科研论文标准的评估指标表
    
    表格包含：
      - 原始物理单位的指标 (MUST HAVE)
      - 相对误差百分比 (便于理解)
      - 统计信息 (观测vs预测)
    """
    print("\n" + "="*100)
    print(f"{'评估指标汇总':^100} (PM2.5预测, 单位: μg/m³)")
    print("="*100)
    
    # 主表：原始物理单位指标
    print(f"\n{'日期':<8} {'R²':<8} {'IA':<8} {'Corr':<8} {'RMSE':<10} {'MAE':<10} {'NRMSE(%)':<10} {'MAPE(%)':<10}")
    print("-"*100)
    
    r2_list = []
    ia_list = []
    rmse_list = []
    mae_list = []
    nrmse_list = []
    
    for day_idx, metrics in enumerate(daily_metrics):
        if metrics is None:
            continue
        
        day_str = f"Day {day_idx+1}"
        r2 = metrics['r2_original']
        ia = metrics['ia_original']
        corr = metrics['correlation']
        rmse = metrics['rmse_original']
        mae = metrics['mae_original']
        nrmse = metrics['nrmse_percent']
        mape = metrics['mape_percent']
        
        r2_list.append(r2)
        ia_list.append(ia)
        rmse_list.append(rmse)
        mae_list.append(mae)
        nrmse_list.append(nrmse)
        
        # 质量评价
        quality = "优秀" if r2 > 0.8 else "很好" if r2 > 0.6 else "一般" if r2 > 0.4 else "差"
        
        print(f"{day_str:<8} {r2:>7.4f} {ia:>7.4f} {corr:>7.4f} {rmse:>9.2f} {mae:>9.2f} {nrmse:>9.2f} {mape:>9.2f}")
    
    # 统计行
    print("-"*100)
    if r2_list:
        print(f"{'平均':<8} {np.mean(r2_list):>7.4f} {np.mean(ia_list):>7.4f} {' '*8} "
              f"{np.mean(rmse_list):>9.2f} {np.mean(mae_list):>9.2f} {np.mean(nrmse_list):>9.2f} {' '*10}")
        print(f"{'标准差':<8} {np.std(r2_list):>7.4f} {np.std(ia_list):>7.4f} {' '*8} "
              f"{np.std(rmse_list):>9.2f} {np.std(mae_list):>9.2f} {np.std(nrmse_list):>9.2f} {' '*10}")
    
    print("="*100)
    
    # 详细统计信息
    if daily_metrics and daily_metrics[0]:
        m = daily_metrics[0]
        print(f"\n[统计信息]")
        print(f"  观测值范围: [{m['min_obs']:.2f}, {m['max_obs']:.2f}] μg/m³")
        print(f"  观测平均值: {m['mean_obs']:.2f} ± {m['std_obs']:.2f} μg/m³")
        print(f"  预测平均值: {m['mean_pred']:.2f} ± {m['std_pred']:.2f} μg/m³")
        print(f"  有效像素数: {m['n_valid_pixels']}")
    
    # 指标解释
    print(f"\n[指标解释]")
    print(f"  R²        > 0.8: 优秀  | 0.6-0.8: 很好  | <0.4: 差 ✗")
    print(f"  IA        > 0.9: 优秀  | 0.8-0.9: 很好  | <0.5: 差 ✗")
    print(f"  Corr      > 0.8: 强相关 | 0.6-0.8: 中等  | <0.4: 弱 ✗")
    print(f"  RMSE      < 20μg/m³: 优秀 | < 30μg/m³: 可接受 | > 50: 差 ✗")
    print(f"  NRMSE(%)  < 10%: 优秀  | < 20%: 很好   | > 30%: 差 ✗")
    print("="*100 + "\n")


def plot_comprehensive_metrics_summary(daily_metrics, output_dir):
    """生成科研标准的评估总图"""
    if not daily_metrics or all(m is None for m in daily_metrics):
        print("[WARN] No valid metrics for plotting")
        return
    
    # 过滤有效指标
    valid_metrics = [m for m in daily_metrics if m is not None]
    n_days = len(valid_metrics)
    days = np.arange(1, n_days + 1)
    
    # 提取数据
    r2_orig = [m['r2_original'] for m in valid_metrics]
    ia_orig = [m['ia_original'] for m in valid_metrics]
    rmse_orig = [m['rmse_original'] for m in valid_metrics]
    mae_orig = [m['mae_original'] for m in valid_metrics]
    nrmse = [m['nrmse_percent'] for m in valid_metrics]
    corr = [m['correlation'] for m in valid_metrics]
    
    # 创建4个子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PM2.5预测性能评估汇总 (原始物理单位)', fontsize=14, fontweight='bold')
    
    # 1. R² 和 IA
    ax = axes[0, 0]
    ax.plot(days, r2_orig, 'o-', label='R²', linewidth=2, markersize=8)
    ax.plot(days, ia_orig, 's-', label='IA', linewidth=2, markersize=8)
    ax.axhline(0.8, color='g', linestyle='--', alpha=0.5, label='优秀阈值')
    ax.axhline(0.6, color='orange', linestyle='--', alpha=0.5, label='可接受阈值')
    ax.set_ylabel('指数值', fontsize=11)
    ax.set_title('相关性指标 (R², IA)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # 2. RMSE 和 MAE
    ax = axes[0, 1]
    ax.plot(days, rmse_orig, 'o-', label='RMSE', linewidth=2, markersize=8)
    ax.plot(days, mae_orig, 's-', label='MAE', linewidth=2, markersize=8)
    ax.axhline(20, color='g', linestyle='--', alpha=0.5, label='优秀阈值')
    ax.axhline(30, color='orange', linestyle='--', alpha=0.5, label='可接受阈值')
    ax.set_ylabel('误差 (μg/m³)', fontsize=11)
    ax.set_title('绝对误差指标 (RMSE, MAE)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. 相对误差 (NRMSE %)
    ax = axes[1, 0]
    bars = ax.bar(days, nrmse, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)
    ax.axhline(10, color='g', linestyle='--', alpha=0.5, linewidth=2, label='优秀阈值')
    ax.axhline(20, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='可接受阈值')
    ax.set_ylabel('NRMSE (%)', fontsize=11)
    ax.set_title('相对误差 (NRMSE = RMSE/mean×100%)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bar, val in zip(bars, nrmse):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. 相关系数
    ax = axes[1, 1]
    ax.plot(days, corr, 'o-', color='purple', linewidth=2, markersize=8, label='Pearson r')
    ax.axhline(0.8, color='g', linestyle='--', alpha=0.5, linewidth=2, label='强相关阈值')
    ax.axhline(0.6, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='中等相关阈值')
    ax.set_ylabel('相关系数', fontsize=11)
    ax.set_title('线性相关强度 (Pearson相关系数)', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # X轴标签
    for ax in axes.flat:
        ax.set_xlabel('预测日期', fontsize=11)
        ax.set_xticks(days)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary_comprehensive.png'), 
                dpi=150, bbox_inches='tight')
    print(f"[OK] 科研标准评估图已保存: {output_dir}/metrics_summary_comprehensive.png")


def plot_daily_scatter(y_true, y_pred, metrics, day_idx, output_dir):
    """绘制单日散点图"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 移除NaN值
    valid_idx = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_flat = y_true_flat[valid_idx]
    y_pred_flat = y_pred_flat[valid_idx]
    
    if len(y_true_flat) == 0:
        return None
    
    # 创建散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=20, c='steelblue', 
               edgecolors='navy', linewidth=0.5)
    
    # 完美预测线
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 线性拟合线
    if len(y_true_flat) > 1:
        z = np.polyfit(y_true_flat, y_pred_flat, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(min_val, max_val, 100)
        ax.plot(x_fit, p(x_fit), 'g-', linewidth=2, label=f'Linear Fit')
    
    # 标题和标签
    title = f'Day {day_idx + 1} - Prediction Scatter\n'
    title += f'R² = {metrics["r2_original"]:.4f}, IA = {metrics["ia_original"]:.4f}\n'
    title += f'RMSE = {metrics["rmse_original"]:.2f} μg/m³, MAE = {metrics["mae_original"]:.2f} μg/m³\n'
    title += f'Corr = {metrics["correlation"]:.4f}, N = {len(y_true_flat)}'
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('True Value (μg/m³)', fontsize=11)
    ax.set_ylabel('Predicted Value (μg/m³)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'scatter_day_{day_idx + 1:02d}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def plot_daily_heatmap(y_true, y_pred, day_idx, output_dir):
    """绘制单日误差热力图"""
    # 计算误差
    error = np.abs(y_true - y_pred)
    
    # 创建3个子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 真实值
    im1 = axes[0].imshow(y_true, cmap='viridis')
    axes[0].set_title(f'Day {day_idx + 1} - True Value', fontweight='bold')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0], label='PM2.5 (μg/m³)')
    
    # 预测值
    im2 = axes[1].imshow(y_pred, cmap='viridis')
    axes[1].set_title(f'Day {day_idx + 1} - Predicted Value', fontweight='bold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[1], label='PM2.5 (μg/m³)')
    
    # 误差
    im3 = axes[2].imshow(error, cmap='hot')
    axes[2].set_title(f'Day {day_idx + 1} - Absolute Error', fontweight='bold')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[2], label='|Error| (μg/m³)')
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f'heatmap_day_{day_idx + 1:02d}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig_path


def evaluate_daily_predictions_scientific(pred_list, true_list, mask_list, 
                                          global_min, global_max, output_dir):
    """
    科研规范的日均评估主函数
    
    Args:
        pred_list: 预测值列表 (已归一化到[0,1])
        true_list: 真实值列表 (已归一化到[0,1])
        mask_list: 有效掩码列表
        global_min: 原始数据最小值 (e.g., 0.00)
        global_max: 原始数据最大值 (e.g., 229.13)
        output_dir: 输出目录
    
    Returns:
        dict: 包含所有日均指标的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    daily_metrics = []
    scatter_dir = os.path.join(output_dir, 'scatter_plots')
    heatmap_dir = os.path.join(output_dir, 'heatmap_plots')
    
    for day_idx, (pred, true, mask) in enumerate(zip(pred_list, true_list, mask_list)):
        if true is None or mask is None:
            daily_metrics.append(None)
            continue
        
        # 如果是多通道，取第一通道
        if pred.ndim == 3:
            pred = pred[0]
        if true.ndim == 3:
            true = true[0]
        
        # 计算完整指标
        metrics = calculate_comprehensive_metrics(true, pred, global_min, global_max, mask)
        daily_metrics.append(metrics)
        
        # 反归一化用于绘图
        true_denorm = denormalize_data(true, global_min, global_max)
        pred_denorm = denormalize_data(pred, global_min, global_max)
        
        # 生成散点图
        plot_daily_scatter(true_denorm, pred_denorm, metrics, day_idx, scatter_dir)
        
        # 生成热力图
        plot_daily_heatmap(true_denorm, pred_denorm, day_idx, heatmap_dir)
    
    # 打印表格
    print_metrics_table(daily_metrics, output_dir)
    
    # 绘制总图
    plot_comprehensive_metrics_summary(daily_metrics, output_dir)
    
    return daily_metrics


if __name__ == "__main__":
    print("科研规范评估模块已导入，使用方法:")
    print("  from eval_metrics_comprehensive import evaluate_daily_predictions_scientific")
    print("  metrics = evaluate_daily_predictions_scientific(pred_list, true_list, mask_list,")
    print("                                                  global_min, global_max, output_dir)")
