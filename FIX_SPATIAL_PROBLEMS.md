# 空间细节丧失问题修复方案

## 修复策略概览

根据诊断，采用**三层递进式修复**：

### 第1层：数据层面（快速见效，1-2小时）
1. **极值重采样** - 增加高污染/低污染样本权重
2. **数据增强** - 生成极值边界样本
3. **归一化改进** - 使用分位数归一化而非min-max

### 第2层：损失函数（中等投入，2-3小时）
1. **梯度约束损失** - 强制预测梯度接近真实
2. **感知损失** - 高频细节保留
3. **极值聚焦损失** - 高权重处理极端值

### 第3层：模型架构（重投入，4-6小时）
1. **隐层扩展** - 128→256维
2. **跳连接** - 保留细节信息
3. **多尺度监督** - 不同分辨率同时学习

---

## ✅ 第1层修复：立即实施

### 方案1A：极值重采样 + 损失权重调整

**文件修改：** `losses/loss_functions.py`

```python
# 添加加权MSE损失，强调极值
class WeightedMSELoss(nn.Module):
    def __init__(self, extreme_weight=2.0, global_min=0.0, global_max=229.13):
        super().__init__()
        self.extreme_weight = extreme_weight
        self.global_min = global_min
        self.global_max = global_max
        
    def forward(self, pred, target, mask=None):
        # 计算权重：极值(>90%分位)给予2倍权重
        normalized = (target - self.global_min) / (self.global_max - self.global_min)
        q90 = torch.quantile(normalized[mask>0], 0.90) if mask is not None else torch.quantile(normalized, 0.90)
        q10 = torch.quantile(normalized[mask>0], 0.10) if mask is not None else torch.quantile(normalized, 0.10)
        
        # 极值样本权重提升
        weights = torch.ones_like(target)
        weights[normalized > q90] = self.extreme_weight
        weights[normalized < q10] = self.extreme_weight
        
        loss = (weights * (pred - target)**2 * mask).sum() / mask.sum()
        return loss
```

**在 `train.py` 中使用：**

```python
# 第45行左右，修改损失函数
from losses.loss_functions import WeightedMSELoss

# 初始化
pred_loss = WeightedMSELoss(extreme_weight=2.0, 
                            global_min=train_dataset.global_min,
                            global_max=train_dataset.global_max)

# 计算损失
pred_error = pred_loss(pred, target, mask)  # 替换原来的 F.mse_loss
```

**预期效果：** R²维持，RMSE±0%, 但梯度保留率 0.54→0.65, 极值表现改善

---

### 方案1B：梯度约束损失（最有效）

**文件修改：** `losses/loss_functions.py` 末尾添加

```python
class GradientConstraintLoss(nn.Module):
    """强制预测梯度接近真实梯度"""
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target, mask=None):
        # 计算梯度
        pred_gy, pred_gx = torch.gradient(pred, dim=[2, 3])
        target_gy, target_gx = torch.gradient(target, dim=[2, 3])
        
        # 计算梯度大小
        pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-6)
        target_grad = torch.sqrt(target_gx**2 + target_gy**2 + 1e-6)
        
        # 梯度一致性损失
        if mask is not None:
            grad_loss = F.mse_loss(pred_grad * mask, target_grad * mask)
        else:
            grad_loss = F.mse_loss(pred_grad, target_grad)
        
        return self.weight * grad_loss
```

**在 `train.py` 中使用（第185行）：**

```python
# 初始化
from losses.loss_functions import GradientConstraintLoss
grad_loss_fn = GradientConstraintLoss(weight=0.05)

# 在train loop中（第185行）
pred_mse = F.mse_loss(pred, target, reduction='none')
pred_mse = (pred_mse * mask).sum() / mask.sum()
grad_loss = grad_loss_fn(pred, target, mask)  # 新增

total_loss = pred_mse + grad_loss  # 替换原来的只有pred_mse
```

**预期效果：** 梯度保留率 0.54→0.85+, RMSE+2-3%, R²-0.02

---

### 方案1C：更新 config 参数（立即生效）

**文件修改：** `config/__init__.py`

```python
# 第45行左右，修改
WEIGHT_SPECTRAL = 0.00001  # 原来0.001，减小Koopman约束
LATENT_DIM = 256  # 原来128，增加表达力
CLAMP_OUTPUT = True  # 保持[0,1]范围

# 新增参数
GRADIENT_LOSS_WEIGHT = 0.05  # 梯度损失权重
EXTREME_VALUE_WEIGHT = 2.0   # 极值样本权重
```

---

## 📋 实施顺序（推荐）

### **步骤1：快速验证（15分钟）**
```bash
# 只修改config，立即训练
# 1. 编辑 config/__init__.py
#    - WEIGHT_SPECTRAL = 0.00001
#    - LATENT_DIM = 256
# 2. 运行
python train.py
# 3. 观察梯度是否改善
```

### **步骤2：添加梯度损失（30分钟）**
```bash
# 1. 编辑 losses/loss_functions.py，添加 GradientConstraintLoss
# 2. 编辑 train.py 第185行，集成梯度损失
# 3. 运行
python train.py
# 4. 检查指标输出
```

### **步骤3：极值重采样（45分钟）**
```bash
# 1. 编辑 losses/loss_functions.py，添加 WeightedMSELoss
# 2. 编辑 train.py 第45行，使用加权损失
# 3. 运行
python train.py
# 4. 对比预测最大值是否提升
```

---

## 代码实施细节

### 修改1：config/__init__.py

```python
# 第44-48行，替换为：
WEIGHT_SPECTRAL = 0.00001      # 减小Koopman谱约束
LATENT_DIM = 256               # 增加隐层维度
GRADIENT_LOSS_WEIGHT = 0.05    # 新增：梯度损失权重
EXTREME_VALUE_WEIGHT = 2.0     # 新增：极值权重
```

### 修改2：losses/loss_functions.py（末尾添加）

```python
# 文件末尾添加这两个类

class WeightedMSELoss(torch.nn.Module):
    """根据值域范围对样本进行权重化"""
    def __init__(self, extreme_weight=2.0, global_min=0.0, global_max=229.13):
        super().__init__()
        self.extreme_weight = extreme_weight
        self.q_min = global_min
        self.q_max = global_max
        
    def forward(self, pred, target, mask=None):
        # 归一化到[0,1]
        norm_target = (target - self.q_min) / (self.q_max - self.q_min + 1e-8)
        
        # 计算分位数阈值
        if mask is not None:
            valid_vals = norm_target[mask > 0.5]
        else:
            valid_vals = norm_target.reshape(-1)
        
        q90 = torch.quantile(valid_vals, 0.90)
        q10 = torch.quantile(valid_vals, 0.10)
        
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


class GradientConstraintLoss(torch.nn.Module):
    """梯度一致性损失：强制预测梯度接近真实梯度"""
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, pred, target, mask=None):
        # 计算梯度 (沿H和W维度)
        # 使用torch.gradient（PyTorch 2.4+）或手动计算
        pred_gy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_gx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_gy = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_gx = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # 梯度幅值
        pred_grad = torch.sqrt(pred_gx[:, :, :, :-1]**2 + pred_gy[:, :, :-1, :]**2 + 1e-6)
        target_grad = torch.sqrt(target_gx[:, :, :, :-1]**2 + target_gy[:, :, :-1, :]**2 + 1e-6)
        
        # 梯度损失
        grad_loss = torch.nn.functional.mse_loss(pred_grad, target_grad)
        
        return self.weight * grad_loss
```

### 修改3：train.py（第40-50行）

```python
# 原代码：
# from losses.loss_functions import SpectralLoss

# 修改为：
from losses.loss_functions import SpectralLoss, WeightedMSELoss, GradientConstraintLoss
```

### 修改4：train.py（第180-200行）训练循环

```python
# 原代码：
# pred_error = F.mse_loss((pred - true_data)**2 * valid_mask)

# 修改为：
# 使用加权MSE
pred_error = weighted_mse(pred, true_data, valid_mask)

# 添加梯度约束
grad_penalty = gradient_constraint(pred, true_data, valid_mask)

# 总损失
total_loss = pred_error + grad_penalty + spectral_loss
```

---

## ⚙️ 关键参数调优表

| 参数 | 保守设置 | 推荐设置 | 激进设置 |
|------|--------|--------|--------|
| LATENT_DIM | 128 | **256** | 512 |
| GRADIENT_LOSS_WEIGHT | 0.01 | **0.05** | 0.10 |
| EXTREME_VALUE_WEIGHT | 1.5 | **2.0** | 3.0 |
| WEIGHT_SPECTRAL | 0.0001 | **0.00001** | 0 |

---

## 📊 预期改进效果

| 指标 | 当前 | 目标 | 实现方式 |
|------|------|------|--------|
| 梯度保留率 | 54% | **85%** | 梯度约束损失 |
| 极值覆盖率 | 70% | **90%** | 隐层扩展+权重 |
| NRMSE% | 109.6% | **25%** | 多管齐下 |
| R² | 0.76 | **0.75-0.80** | 略微下降，但空间质量↑ |
| 梯度幅值 | 2.13 | **3.2+** | 梯度损失 |

---

## 🧪 验证方法

每次修改后运行：

```bash
# 1. 训练
python train.py

# 2. 分析空间特征
python spatial_analysis.py

# 3. 检查输出中的关键指标
# - 梯度幅值是否从2.13提升到3.0+
# - 极值(最大值)是否从112提升到140+
# - NRMSE%是否下降
```

---

## 🚀 快速启动（复制即用）

### 第1步：修改 config/__init__.py
```python
WEIGHT_SPECTRAL = 0.00001
LATENT_DIM = 256
GRADIENT_LOSS_WEIGHT = 0.05
EXTREME_VALUE_WEIGHT = 2.0
```

### 第2步：运行测试
```bash
python train.py
python spatial_analysis.py
```

### 第3步：观察改进
查看 `spatial_feature_analysis.png` 中的梯度对比

---

## ⚠️ 风险与规避

| 风险 | 症状 | 规避方案 |
|------|------|--------|
| 过拟合 | R²下降>0.10 | 降低 GRADIENT_LOSS_WEIGHT 到0.02 |
| 梯度爆炸 | 训练loss NaN | 使用 gradient_clipping |
| 内存溢出 | CUDA OOM | 降低 LATENT_DIM 到192 |
| 收敛困难 | 150 epoch后无改善 | 增加 WEIGHT_DECAY 或降低学习率 |

---

## 📝 实施检查清单

- [ ] 修改 config/__init__.py (LATENT_DIM, WEIGHT_SPECTRAL等)
- [ ] 添加 WeightedMSELoss 到 losses/loss_functions.py
- [ ] 添加 GradientConstraintLoss 到 losses/loss_functions.py
- [ ] 修改 train.py 导入语句
- [ ] 修改 train.py 训练循环使用新损失
- [ ] 运行 python train.py 验证训练正常
- [ ] 运行 python spatial_analysis.py 检查梯度改善
- [ ] 对比 metrics_summary_comprehensive.png

---

**预期总耗时：1-2小时完全实施 + 2小时训练 = 3-4小时**
