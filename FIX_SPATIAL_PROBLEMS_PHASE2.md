# 第2轮修复方案：增强梯度约束

## 问题诊断

从第一次运行的结果看：
- ❌ 预测梯度仍然固定在 2.133（完全没有变化）
- ❌ 预测极值仍然固定在 112.34
- ❌ 隐层维度增加到256也没有改善

**根本原因：**
1. 梯度损失权重0.05太弱，无法与主损失竞争
2. 梯度损失只应用于最后一步，缺乏连续性
3. 模型陷入局部最优（平滑输出），梯度惩罚无法逃脱

## 第2层修复：增强梯度约束

### 方案2A：增强梯度损失权重

```python
# config/__init__.py，修改：
GRADIENT_LOSS_WEIGHT = 0.2  # 从0.05提升到0.2 (4倍强度)
```

### 方案2B：改进梯度损失计算（多时间步应用）

在 `train.py` 第145行左右，修改为：

```python
# 应用梯度约束到整个序列
for i, pred_frame in enumerate(pred_sequence):
    pred_frame = pred_frame.squeeze(1)  # [batch, C, H, W]
    target_frame = seq_img[:, i+1]
    loss_gradient += gradient_criterion(pred_frame, target_frame, mask[:, 0, :, :, :])

loss_gradient = loss_gradient / max(len(pred_sequence), 1)
```

### 方案2C：替代方案——使用感知损失+频域匹配

如果梯度约束仍不足效果，考虑添加高阶方法：

```python
class PerceptualGradientLoss(nn.Module):
    """感知梯度损失：匹配图像的梯度结构"""
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        # 预定义梯度算子
        self.gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        self.gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        
    def forward(self, pred, target):
        # 计算Sobel梯度
        pred_gx = F.conv2d(pred, self.gx.to(pred.device), padding=1)
        pred_gy = F.conv2d(pred, self.gy.to(pred.device), padding=1)
        target_gx = F.conv2d(target, self.gx.to(target.device), padding=1)
        target_gy = F.conv2d(target, self.gy.to(target.device), padding=1)
        
        # 梯度匹配
        loss = F.mse_loss(pred_gx, target_gx) + F.mse_loss(pred_gy, target_gy)
        return self.weight * loss
```

## 第2次实施步骤

### 步骤1：快速尝试（5分钟）
只修改梯度损失权重：

```bash
# config/__init__.py 第52行
GRADIENT_LOSS_WEIGHT = 0.2  # 4倍增强
```

运行训练看效果。

### 步骤2：如果第1步无效，修改梯度损失应用范围

在 `train.py` train loop中应用多时间步梯度损失。

### 步骤3：终极方案——使用拉普拉斯损失

```python
class LaplacianConstraintLoss(nn.Module):
    """拉普拉斯（二阶导数）约束，强制平滑度匹配"""
    def __init__(self, weight=0.05):
        super().__init__()
        self.weight = weight
        self.laplacian_kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0) / 4.0
        
    def forward(self, pred, target):
        pred_lap = F.conv2d(pred, self.laplacian_kernel.to(pred.device), padding=1)
        target_lap = F.conv2d(target, self.laplacian_kernel.to(target.device), padding=1)
        
        loss = F.mse_loss(pred_lap, target_lap)
        return self.weight * loss
```

## 诊断对策

如果修改后仍无效，问题可能出在：

1. **模型是否被真正加载？** → 检查 `predict_and_export_n_days` 是否使用 `model.eval()`
2. **优化器是否收敛到局部最优？** → 降低学习率或增加warmup
3. **损失函数计算是否有bug？** → 打印loss值检查梯度损失是否>0
4. **数据是否标准化正确？** → 检查输入范围是否确实是[0,1]

## 下一步计划

根据第2次修复的结果决定：
- ✓ 如果梯度>3.0且极值>140 → 成功，进行第3层修复（多尺度学习）
- ⚠️ 如果梯度>2.5但未达标 → 继续调参（权重、学习率）
- ✗ 如果梯度仍~2.1 → 问题更深层，需要模型架构改变

## 快速启动

只修改一行配置运行测试：
```bash
# config/__init__.py
GRADIENT_LOSS_WEIGHT = 0.2

# 运行
python train.py
python spatial_analysis.py
```

观察梯度是否从2.133改善。
