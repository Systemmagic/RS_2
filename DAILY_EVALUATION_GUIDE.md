# 每日预测评估使用指南

## 概述

新添加的 `eval_daily.py` 模块提供了全面的每日预测评估功能，包括：

- **四大评估指标**: R²、IA（一致性指数）、RMSE、MAE
- **散点图可视化**: 每天的预测值 vs 真实值散点图，含完美预测线和线性拟合线
- **误差热力图**: 真实值、预测值、误差的空间分布
- **汇总报告**: 多日预测指标的统计汇总和趋势图

---

## 评估指标说明

### R² (决定系数)
- **范围**: [-∞, 1]
- **定义**: $R^2 = 1 - \frac{\sum(y_{pred} - y_{true})^2}{\sum(y_{true} - \bar{y}_{true})^2}$
- **解释**: 
  - R² = 1: 完美预测
  - R² = 0: 预测精度等同于常数预测（用平均值）
  - R² < 0: 预测效果差于常数预测

### IA (Index of Agreement - 一致性指数)
- **范围**: [0, 1]
- **定义**: $IA = 1 - \frac{\sum(y_{pred} - y_{true})^2}{\sum(|y_{pred} - \bar{y}_{true}| + |y_{true} - \bar{y}_{true}|)^2}$
- **解释**:
  - IA = 1: 完美预测
  - IA = 0.5: 中等预测精度
  - IA = 0: 预测能力为0

### RMSE (Root Mean Square Error)
- **定义**: $RMSE = \sqrt{\frac{1}{n}\sum(y_{pred} - y_{true})^2}$
- **解释**: 预测误差的均方根，与真实值单位相同
- **特点**: 对大误差更敏感

### MAE (Mean Absolute Error)
- **定义**: $MAE = \frac{1}{n}\sum|y_{pred} - y_{true}|$
- **解释**: 预测误差的平均绝对值
- **特点**: 对异常值不太敏感

---

## 输出文件结构

```
result/experiment_1/
├── daily_evaluation/
│   ├── scatter_plots/
│   │   ├── scatter_day_01.png   # Day 1 散点图
│   │   ├── scatter_day_02.png   # Day 2 散点图
│   │   └── ...
│   ├── heatmap_plots/
│   │   ├── heatmap_day_01.png   # Day 1 误差热力图
│   │   ├── heatmap_day_02.png   # Day 2 误差热力图
│   │   └── ...
│   └── metrics_summary.png      # 指标汇总图表
```

---

## 使用方式

### 方式1: 自动集成到train.py（推荐）

在训练完成后，会自动执行每日预测评估：

```bash
python train.py
# 训练完成后自动生成评估结果
```

### 方式2: 独立运行评估

```python
from eval_daily import evaluate_daily_predictions
import numpy as np

# 准备数据
pred_list = [np.random.rand(256, 256) for _ in range(7)]  # 7天预测
true_list = [np.random.rand(256, 256) for _ in range(7)]  # 7天真实值
mask_list = [np.ones((256, 256)) for _ in range(7)]       # 有效区域掩码

# 运行评估
eval_results = evaluate_daily_predictions(
    pred_list, 
    true_list, 
    mask_list, 
    output_dir="result/evaluation"
)

# 查看结果摘要
print(eval_results['summary'])
```

### 方式3: 仅评估特定日期

```python
from eval_daily import calculate_daily_metrics, plot_daily_scatter
import numpy as np

# 单日评估
pred = np.random.rand(256, 256)
true = np.random.rand(256, 256)
mask = np.ones((256, 256))

# 计算指标
metrics = calculate_daily_metrics(true, pred, mask)
print(f"R² = {metrics['r2']:.4f}")
print(f"IA = {metrics['ia']:.4f}")
print(f"RMSE = {metrics['rmse']:.4f}")
print(f"MAE = {metrics['mae']:.4f}")

# 绘制散点图
plot_daily_scatter(true, pred, metrics, day_idx=0, output_dir="output")
```

---

## 输出示例

### 控制台输出

```
================================================================================
                              每日预测评估
================================================================================
Day      R²         IA         RMSE       MAE        Corr       N_Pixels    
--------------------------------------------------------------------------------
Day 1    0.6543     0.8234     12.3456    8.1234     0.7890     65536
Day 2    0.6721     0.8356     11.9876    7.8901     0.8012     65536
Day 3    0.6512     0.8234     12.5678    8.3456     0.7934     65536
Day 4    0.6634     0.8290     12.1234    8.0567     0.7956     65536
Day 5    0.6456     0.8156     12.7890    8.5678     0.7823     65536
Day 6    0.6578     0.8212     12.4567    8.2345     0.7889     65536
Day 7    0.6612     0.8267     12.2345    8.1234     0.7912     65536
--------------------------------------------------------------------------------
Mean     0.6579     0.8249     12.3506    8.2060     0.7875     
Std      0.0079     0.0062     0.2654     0.1876              
================================================================================
```

### 散点图特征

- **蓝色散点**: 预测值 vs 真实值（每个有效像素一个点）
- **红色虚线**: 完美预测线（y=x）
- **绿色实线**: 线性拟合线，显示预测的总体趋势
- **标题**: 包含R²、IA、RMSE、MAE、相关系数和有效像素数

### 热力图说明

三个并排的热力图：
1. **左图**: 真实PM2.5浓度分布
2. **中图**: 预测PM2.5浓度分布
3. **右图**: 绝对误差分布（仅显示有效区域）

### 指标汇总图

四个子图显示各指标在多日的变化趋势：
- R² 曲线（越高越好）
- IA 曲线（越高越好）
- RMSE 曲线（越低越好）
- MAE 曲线（越低越好）

---

## 参数说明

### evaluate_daily_predictions()

```python
def evaluate_daily_predictions(
    pred_list,      # List[np.ndarray], 预测值列表，每个[H,W]或[C,H,W]
    true_list,      # List[np.ndarray], 真实值列表，形状同pred_list
    mask_list,      # List[np.ndarray], 掩码列表，每个[H,W]，1表示有效
    output_dir      # str, 输出目录
)
```

### calculate_daily_metrics()

```python
def calculate_daily_metrics(
    y_true,    # np.ndarray, 真实值[H,W]或[C,H,W]
    y_pred,    # np.ndarray, 预测值
    mask=None  # np.ndarray, 掩码[H,W]
)
```

返回字典包含:
- `r2`: R² 值
- `ia`: 一致性指数
- `rmse`: 均方根误差
- `mae`: 平均绝对误差
- `correlation`: 皮尔逊相关系数
- `n_valid_pixels`: 有效像素数

---

## 高级使用

### 自定义阈值分析

```python
from eval_daily import calculate_daily_metrics
import numpy as np

# 分别计算不同浓度范围的指标
pred = np.random.rand(256, 256) * 200
true = np.random.rand(256, 256) * 200

# 低浓度区域 (< 50)
low_mask = true < 50
low_metrics = calculate_daily_metrics(true, pred, low_mask)

# 中浓度区域 (50-150)
mid_mask = (true >= 50) & (true < 150)
mid_metrics = calculate_daily_metrics(true, pred, mid_mask)

# 高浓度区域 (>= 150)
high_mask = true >= 150
high_metrics = calculate_daily_metrics(true, pred, high_mask)

print(f"Low pollution R²: {low_metrics['r2']:.4f}")
print(f"Mid pollution R²: {mid_metrics['r2']:.4f}")
print(f"High pollution R²: {high_metrics['r2']:.4f}")
```

### 批量处理多个时间段

```python
from eval_daily import evaluate_daily_predictions

# 评估不同周期的预测
for week_start in range(0, 60, 7):
    pred_list = load_predictions(week_start, week_start + 7)
    true_list = load_ground_truth(week_start, week_start + 7)
    mask_list = load_masks(week_start, week_start + 7)
    
    results = evaluate_daily_predictions(
        pred_list, true_list, mask_list, 
        output_dir=f"evaluation/week_{week_start//7}"
    )
    
    # 记录周平均指标
    weekly_r2.append(results['summary']['r2_mean'])
    weekly_ia.append(results['summary']['ia_mean'])
```

---

## 常见问题

**Q: 为什么某天的指标为NaN？**
A: 通常是因为该天没有有效的预测或真实值。检查掩码和数据是否正确加载。

**Q: 如何只评估某个地理区域？**
A: 通过 `mask` 参数过滤。将不感兴趣的区域设置为0，有效区域设置为1。

**Q: 散点图中有离群点怎么办？**
A: 这通常表示预测在某些区域存在系统性偏差。可以：
1. 增加该地区的训练数据
2. 调整模型参数
3. 使用鲁棒指标（如中位数误差）

**Q: 如何提高多日预测精度？**
A: 
1. 增加气象特征输入
2. 使用滑动窗口集成
3. 微调ALPHA参数

---

## 最佳实践

1. **定期评估**: 在训练的不同阶段评估，监测模型改进
2. **多指标综合**: 不要只看单一指标，R²和IA结合判断
3. **空间分析**: 利用热力图识别模型的弱点区域
4. **阈值分析**: 分别评估不同浓度等级的预测能力
5. **长期趋势**: 收集多周期数据，分析模型稳定性

---

**最后更新**: 2026年1月22日
