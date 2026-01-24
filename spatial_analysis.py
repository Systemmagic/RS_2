import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from osgeo import gdal
import os
import glob
from config import Config

config = Config()
base_dir = config.OUTPUT_DIR

# 禁用警告
gdal.DontUseExceptions()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def read_tif(file_path):
    """读取TIF文件"""
    ds = gdal.Open(file_path)
    if ds is None:
        return None
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    ds = None
    return data

def calculate_spatial_features(data, mask=None):
    """计算空间特征"""
    if mask is not None:
        data = data * mask
    
    # 1. 梯度（Gradient）- 检测边界和过渡
    gy, gx = np.gradient(data)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    
    # 2. 拉普拉斯（Laplacian）- 检测尖锐变化
    laplacian = ndimage.laplace(data)
    
    # 3. 局部方差（Local Variance）- 检测纹理复杂度
    kernel_size = 5
    local_mean = ndimage.uniform_filter(data, size=kernel_size)
    local_var = ndimage.uniform_filter(data**2, size=kernel_size) - local_mean**2
    
    # 4. 空间自相关（Moran's I）- 检测空间平滑度
    center = data[1:-1, 1:-1]
    neighbors = (data[:-2, 1:-1] + data[2:, 1:-1] + 
                 data[1:-1, :-2] + data[1:-1, 2:]) / 4
    moran_i = np.mean((center - np.mean(data)) * (neighbors - np.mean(data))) / np.var(data)
    
    return {
        'gradient': gradient_magnitude,
        'laplacian': np.abs(laplacian),
        'local_var': local_var,
        'moran_i': moran_i,
        'gradient_mean': np.mean(gradient_magnitude),
        'gradient_std': np.std(gradient_magnitude),
    }

def analyze_distribution(data, name="数据"):
    """分析数据分布"""
    valid = data[data > 0]
    print(f"\n{name}分布统计:")
    print(f"  最小值: {np.min(valid):.2f}")
    print(f"  最大值: {np.max(valid):.2f}")
    print(f"  均值: {np.mean(valid):.2f}")
    print(f"  中位数: {np.median(valid):.2f}")
    print(f"  标准差: {np.std(valid):.2f}")
    print(f"  偏度: {(np.mean(valid) - np.median(valid)) / np.std(valid):.3f}")
    
    # 峰值分析
    peak_ratio = np.sum(valid > np.percentile(valid, 95)) / len(valid)
    print(f"  95%分位数以上的像素占比: {peak_ratio*100:.1f}%")

def compare_predictions():
    """对比预测和真实数据"""

    
    # 读取第1、4、7天的数据（指标最好和最差）
    days = [1, 4, 7]
    
    fig, axes = plt.subplots(len(days), 5, figsize=(20, 4*len(days)))
    
    for idx, day in enumerate(days):
        pred_file = os.path.join(base_dir, f"Pred_Day_{day:02d}.tif")
        
        pred_data = read_tif(pred_file)
        
        print(f"\n{'='*80}")
        print(f"Day {day} 空间分析")
        print(f"{'='*80}")
        
        if pred_data is None:
            print(f"  [ERROR] 无法读取预测文件: {pred_file}")
            continue
        
        # 尝试读取对应的真实数据文件（如果存在）
        true_file = os.path.join(base_dir, f"daily_evaluation/True_Day_{day:02d}.tif")
        true_data = read_tif(true_file)
        
        if true_data is None:
            # 从数据文件夹直接读取
            from config import Config
            data_dir = Config().DATA_DIR
            all_files = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
            if day - 1 < len(all_files):
                true_data = read_tif(all_files[day - 1])
                # 反归一化（假设范围0-229.13）
                if true_data is not None:
                    true_data = np.nan_to_num(true_data, nan=0.0)
            
            if true_data is None:
                print(f"  [WARN] 无法获取真实数据")
                true_data = None
        
        # 计算空间特征
        pred_features = calculate_spatial_features(pred_data)
        if true_data is not None:
            true_features = calculate_spatial_features(true_data)
        
        # 分析分布
        analyze_distribution(pred_data, f"Day {day} 预测值")
        if true_data is not None:
            analyze_distribution(true_data, f"Day {day} 真实值")
            
            # 比较特征
            print(f"\n空间特征对比:")
            print(f"  梯度幅值 - 预测: {pred_features['gradient_mean']:.3f}±{pred_features['gradient_std']:.3f}")
            print(f"  梯度幅值 - 真实: {true_features['gradient_mean']:.3f}±{true_features['gradient_std']:.3f}")
            print(f"  梯度差异比: {pred_features['gradient_mean']/max(true_features['gradient_mean'], 1e-6):.2f}x")
            print(f"  空间自相关 - 预测: {pred_features['moran_i']:.3f}")
            print(f"  空间自相关 - 真实: {true_features['moran_i']:.3f}")
            
            # 计算频率域特征（FFT）
            pred_fft = np.abs(np.fft.fft2(pred_data))
            true_fft = np.abs(np.fft.fft2(true_data))
            pred_fft_norm = pred_fft / np.sum(pred_fft)
            true_fft_norm = true_fft / np.sum(true_fft)
            
            # 高频能量比
            h, w = pred_fft.shape
            mask_low = np.zeros((h, w))
            mask_low[h//4:3*h//4, w//4:3*w//4] = 1
            
            pred_high_energy = np.sum(pred_fft_norm * (1 - mask_low))
            true_high_energy = np.sum(true_fft_norm * (1 - mask_low))
            
            print(f"  高频能量比 - 预测: {pred_high_energy*100:.1f}%")
            print(f"  高频能量比 - 真实: {true_high_energy*100:.1f}%")
            print(f"  高频能量差异: {(pred_high_energy - true_high_energy)*100:.1f}%")
        
        # 可视化
        row = idx
        
        # 1. 预测地图
        im1 = axes[row, 0].imshow(pred_data, cmap='viridis')
        axes[row, 0].set_title(f'Day {day} 预测值\n(R²={[0.9819, 0.5587, 0.8979][idx]:.3f})')
        plt.colorbar(im1, ax=axes[row, 0])
        
        # 2. 梯度对比
        im2 = axes[row, 1].imshow(pred_features['gradient'], cmap='hot')
        axes[row, 1].set_title(f'预测梯度\n(Mean={pred_features["gradient_mean"]:.2f})')
        plt.colorbar(im2, ax=axes[row, 1])
        
        if true_data is not None:
            # 3. 真实地图
            im3 = axes[row, 2].imshow(true_data, cmap='viridis')
            axes[row, 2].set_title('真实值')
            plt.colorbar(im3, ax=axes[row, 2])
            
            # 4. 真实梯度
            im4 = axes[row, 3].imshow(true_features['gradient'], cmap='hot')
            axes[row, 3].set_title(f'真实梯度\n(Mean={true_features["gradient_mean"]:.2f})')
            plt.colorbar(im4, ax=axes[row, 3])
            
            # 5. 差异地图
            diff = np.abs(pred_data - true_data)
            im5 = axes[row, 4].imshow(diff, cmap='RdYlBu_r')
            axes[row, 4].set_title(f'绝对误差\n(MAE={np.mean(diff):.2f})')
            plt.colorbar(im5, ax=axes[row, 4])
        
        for ax in axes[row]:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/spatial_feature_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n[OK] 空间特征分析图已保存")
    plt.close()

def analyze_problem_patterns():
    """分析常见问题模式"""
    print("\n" + "="*80)
    print("问题模式诊断")
    print("="*80)
    
    print("""
【可能的问题1】- 过度平滑化
  症状: R²/IA指标好，但空间细节丢失
  检验: 预测梯度 << 真实梯度
  原因: 
    - 神经网络倾向于学习平均值（惩罚异常值）
    - 高斯核卷积作用
    - 训练数据不均衡（低值占多数）
    - MSE损失函数对峰值敏感性不足

【可能的问题2】- 空间不匹配
  症状: 空间模式与真实数据错位
  检验: 高频能量差异大
  原因:
    - 模型未能学习空间相关性
    - Koopman动力学假设过强
    - 卷积核感受野不足
    - 输入序列窗口太小

【可能的问题3】- 极值压制
  症状: 预测值在中等范围，缺少极端值
  检验: 预测分布偏度 < 真实分布偏度
  原因:
    - 归一化后极值稀疏
    - Clamp[0,1]限制了表达力
    - 模型对稀有事件学习不足
    - 权重平衡导致的均化效应

【可能的问题4】- 低秩近似
  症状: 预测显示低频主导模式，细节贫乏
  检验: 预测高频能量 << 真实高频能量
  原因:
    - Koopman算子只捕获主要模式
    - 隐层维度不足（128→64）
    - 损失函数未强制细节学习
    - 编码器特征提取不足
    """)

if __name__ == "__main__":
    print("开始空间特征诊断分析...")
    compare_predictions()
    analyze_problem_patterns()
