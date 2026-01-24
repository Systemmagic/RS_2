# 每日预测评估快速参考

## 一句话总结
为每天的预测生成散点图+热力图，并计算R²、IA、RMSE、MAE四个核心指标。

---

## 快速开始

### 方式1: 自动集成（推荐）
```bash
python train.py
# 训练完成后自动生成评估结果到 result/experiment_1/daily_evaluation/
```

### 方式2: 独立使用
```python
from eval_daily import evaluate_daily_predictions
import numpy as np

# 准备数据（示例）
pred_list = [np.random.rand(256, 256) for _ in range(7)]
true_list = [np.random.rand(256, 256) for _ in range(7)]
mask_list = [np.ones((256, 256)) for _ in range(7)]

# 运行评估
results = evaluate_daily_predictions(pred_list, true_list, mask_list, 
                                     output_dir="my_evaluation")
```

---

## 评估指标

| 指标 | 范围 | 最佳值 | 说明 |
|------|------|--------|------|
| **R²** | [-∞, 1] | 1 | 预测方差解释率 |
| **IA** | [0, 1] | 1 | 预测一致性 |
| **RMSE** | [0, ∞) | 0 | 均方根误差（与数据同单位） |
| **MAE** | [0, ∞) | 0 | 平均绝对误差 |

---

## 输出文件

```
daily_evaluation/
├── scatter_plots/
│   ├── scatter_day_01.png    ← 第1天: 预测值 vs 真实值
│   ├── scatter_day_02.png
│   └── ...
├── heatmap_plots/
│   ├── heatmap_day_01.png    ← 第1天: 空间分布对比
│   ├── heatmap_day_02.png
│   └── ...
└── metrics_summary.png       ← 汇总: 4个指标趋势
```

---

## 核心函数

### 1. 完整评估
```python
evaluate_daily_predictions(pred_list, true_list, mask_list, output_dir)
# 输入: 7天预测 + 7天真实值 + 7个掩码
# 输出: 散点图+热力图+汇总报告
```

### 2. 单日指标
```python
metrics = calculate_daily_metrics(y_true, y_pred, mask=None)
# 输出: {r2, ia, rmse, mae, correlation, n_valid_pixels}
```

### 3. 绘制散点
```python
plot_daily_scatter(y_true, y_pred, metrics, day_idx, output_dir, mask)
# 输出: scatter_day_{idx}.png
```

### 4. 绘制热力图
```python
plot_daily_heatmap(y_true, y_pred, day_idx, output_dir, mask)
# 输出: heatmap_day_{idx}.png
```

---

## 控制台输出示例

```
================================================================================
                              每日预测评估
================================================================================
Day      R²         IA         RMSE       MAE        Corr       N_Pixels    
--------------------------------------------------------------------------------
Day 1    0.6543     0.8234     12.3456    8.1234     0.7890     65536
Day 2    0.6721     0.8356     11.9876    7.8901     0.8012     65536
...
Day 7    0.6612     0.8267     12.2345    8.1234     0.7912     65536
--------------------------------------------------------------------------------
Mean     0.6579     0.8249     12.3506    8.2060     0.7875     
Std      0.0079     0.0062     0.2654     0.1876              
================================================================================
```

---

## 图表说明

### 散点图
- **蓝点**: 每个有效像素的预测值 vs 真实值
- **红虚线**: y=x（完美预测）
- **绿实线**: 线性拟合（显示偏差趋势）
- **标题**: 包含R²、IA、RMSE、MAE

### 热力图 (3个子图)
- **左**: 真实PM2.5浓度分布
- **中**: 预测PM2.5浓度分布  
- **右**: 绝对误差|true-pred|

### 指标汇总
- **左上**: R²趋势（越高越好 ↑）
- **右上**: IA趋势（越高越好 ↑）
- **左下**: RMSE趋势（越低越好 ↓）
- **右下**: MAE趋势（越低越好 ↓）

---

## 常见用途

### 检查单天异常
```python
from eval_daily import calculate_daily_metrics
metrics = calculate_daily_metrics(true, pred, mask)
if metrics['r2'] < 0.5:
    print(f"Day预测质量差: R²={metrics['r2']}")
    # 检查该天的气象数据或模型输入
```

### 分区域评估
```python
# 只评估某个区域
region_mask = np.zeros_like(true)
region_mask[100:200, 100:200] = 1  # 感兴趣区域
metrics = calculate_daily_metrics(true, pred, region_mask)
```

### 分浓度等级评估
```python
low_mask = true < 50      # 低污染
high_mask = true >= 150   # 高污染
low_metrics = calculate_daily_metrics(true, pred, low_mask)
high_metrics = calculate_daily_metrics(true, pred, high_mask)
```

---

## 参数说明

### 输入数据
- **pred_list**: List of ndarray, 形状 [H,W] 或 [C,H,W]
- **true_list**: List of ndarray, 同pred_list形状
- **mask_list**: List of ndarray, 形状 [H,W], 0/1表示无效/有效

### mask说明
- **mask=1**: 有效像素，参与计算
- **mask=0**: 无效像素（如海洋、山区），忽略

---

## 性能指标解释

### R² = 0.65 表示什么？
预测能解释真实值65%的方差，还有35%的偏差。

### IA = 0.82 表示什么？
预测结果与真实值有82%的一致性，较好。

### RMSE = 12.3 表示什么？
平均每个像素预测偏差12.3μg/m³（与PM2.5数据同单位）。

### MAE = 8.1 表示什么？
绝对误差平均值8.1μg/m³，对异常值不敏感。

---

## 文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 评估模块 | `eval_daily.py` | 270+行核心代码 |
| 使用文档 | `DAILY_EVALUATION_GUIDE.md` | 详细说明 |
| 集成代码 | `train.py` | 自动调用评估 |
| 输出目录 | `result/experiment_1/daily_evaluation/` | 所有图表 |

---

## 下一步

1. ✅ 运行 `python train.py` 生成评估结果
2. ✅ 查看 `result/experiment_1/daily_evaluation/` 下的图表
3. ✅ 根据指标调整模型参数
4. ✅ 重复训练-评估循环直到满意

---

**最后更新**: 2026年1月22日  
**版本**: v1.0
