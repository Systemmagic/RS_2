import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入模块化组件
from config import Config
from utils import set_seed, get_device
from data import EnhancedPM25Dataset
from data.utils import export_geotiff
from models import ControlledKoopmanModel, SchurKoopmanLayer
from losses import SharpnessAwareLoss
from losses.loss_functions import GradientConstraintLoss


def predict_and_export_n_days(model, dataset, device, output_dir, start_day_idx=180, n_days=7):
    """
    预测未来n天并导出GeoTIFF + 返回预测结果用于评估
    
    Returns:
        tuple: (pred_list, true_list, mask_list)
    """
    print(f"--- 正在预测未来 {n_days} 天(从索引 {start_day_idx}开始) ---")
    model.eval()
    
    # 准备启动图像
    x_current_raw = dataset.get_all_data()[start_day_idx]
    x_current_norm = np.nan_to_num(x_current_raw, nan=0.0)
    x_current_tensor = torch.tensor(x_current_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 准备未来气象数据
    future_meteo = dataset.meteo_data[start_day_idx+1 : start_day_idx+1+n_days]
    if len(future_meteo) < n_days:
        pad = np.tile(dataset.meteo_data[-1], (n_days - len(future_meteo), 1))
        future_meteo = np.vstack([future_meteo, pad]) if len(future_meteo) > 0 else pad
    
    u_sequence = torch.tensor(future_meteo, dtype=torch.float32).to(device)
    
    pred_list = []
    true_list = []
    mask_list = []
    
    with torch.no_grad():
        preds_list = model.predict_future(x_current_tensor, u_sequence, steps=n_days)
        for i, pred_tensor in enumerate(preds_list):
            day = i + 1
            pred_img = pred_tensor.squeeze().cpu().numpy()
            
            # 获取真实值（如果存在）
            true_day_idx = start_day_idx + day
            if true_day_idx < len(dataset.get_all_data()):
                true_img = dataset.get_all_data()[true_day_idx]
                true_list.append(true_img)
                
                # 获取对应的掩码
                mask = dataset.mask_data[true_day_idx] if hasattr(dataset, 'mask_data') else np.ones_like(true_img)
                mask_list.append(mask)
            else:
                true_list.append(None)
                mask_list.append(None)
            
            pred_list.append(pred_img)
            
            # Export GeoTIFF
            filename = os.path.join(output_dir, f"Pred_Day_{day:02d}.tif")
            export_geotiff(filename, pred_img, dataset)
            print(f"  [OK] Saved: {filename}")
    
    return pred_list, true_list, mask_list

def main():
    # 1. 初始化配置/设备/种子
    config = Config()
    set_seed(42)
    device = get_device()
    
    # 2. 初始化数据集
    os.makedirs(config.DATA_DIR, exist_ok=True)
    try:
        train_dataset = EnhancedPM25Dataset(config.DATA_DIR, config, augment=True)
    except FileNotFoundError as e:
        print(f"数据加载失败：{e}")
        return
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 3. 初始化模型/优化器/损失函数
    model = ControlledKoopmanModel(config).to(device)
    
    # 使用参数分组优化器，为K矩阵设置更高的学习率
    param_groups = [
        {'params': model.K.parameters(), 'lr': config.LEARNING_RATE * 2.0, 'weight_decay': 0.0},  # K矩阵2倍学习率
        {'params': [p for n, p in model.named_parameters() if 'K' not in n], 
         'lr': config.LEARNING_RATE, 'weight_decay': config.WEIGHT_DECAY}  # 其他参数正常学习率
    ]
    optimizer = optim.AdamW(param_groups)
    
    # 初始化舒尔分解Koopman层（可选，用于增强稳定性）
    # schur_layer = SchurKoopmanLayer(dim=config.LATENT_DIM).to(device)
    # 如果使用SchurKoopmanLayer，需要将其参数加入优化器
    # optimizer = optim.AdamW(
    #     list(model.parameters()) + list(schur_layer.parameters()),
    #     lr=config.LEARNING_RATE,
    #     weight_decay=config.WEIGHT_DECAY
    # )
    
    # 初始化损失函数，传入模型引用以获取K矩阵
    criterion = SharpnessAwareLoss(config, koopman_model=model).to(device)
    gradient_criterion = GradientConstraintLoss(weight=config.GRADIENT_LOSS_WEIGHT).to(device)
    
    print(f"\n>>> 启动训练 (Control={config.USE_METEO_CONTROL}, Latent Dim={config.LATENT_DIM}) <<<")
    print(f"    Gradient Loss Weight={config.GRADIENT_LOSS_WEIGHT}")
    
    # 4. 训练循环
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        total_pred_loss = 0
        total_spectral_loss = 0
        
        for seq_img, seq_meteo, mask in train_loader:
            seq_img = seq_img.to(device)
            seq_meteo = seq_meteo.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            x_curr = seq_img[:, 0] 
            z_curr = model.encoder(x_curr)
            
            # 收集预测序列用于损失计算
            pred_sequence = []
            loss_batch = 0 
            
            # 递归预测序列
            for t in range(1, config.SEQUENCE_LENGTH):
                u_curr = seq_meteo[:, t-1] 
                # 动力学演化
                control_term = model.compute_control(u_curr)
                z_next_pred = model.K(z_curr) + control_term
                # 解码+残差更新
                delta = model.decoder(z_next_pred)
                x_next_pred = torch.relu(x_curr + delta)
                
                pred_sequence.append(x_next_pred.unsqueeze(1))  # [batch, 1, C, H, W]
                
                # 更新状态
                x_curr = x_next_pred 
                z_curr = z_next_pred
            
            # 将预测序列堆叠 [batch, Sp, C, H, W]
            pred_seq = torch.cat(pred_sequence, dim=1) if pred_sequence else seq_img[:, 1:2]
            
            # 使用新的损失函数计算（包含Koopman相关损失）
            loss_batch = criterion(
                x_seq=seq_img,      # [batch, T, C, H, W] 真实序列
                pred_seq=pred_seq,  # [batch, Sp, C, H, W] 预测序列
                mask=mask[:, 0, :, :, :]   # [batch, H, W] 取第一时刻的掩码
            )
            
            # 多尺度梯度约束损失：应用到所有预测步骤
            loss_gradient = 0.0
            for t in range(min(len(pred_sequence), seq_img.shape[1] - 1)):
                pred_t = pred_sequence[t].squeeze(1)  # [batch, C, H, W]
                true_t = seq_img[:, t+1, :, :, :]    # [batch, C, H, W]
                loss_gradient += gradient_criterion(pred_t, true_t, mask[:, 0, :, :, :])
            loss_gradient = loss_gradient / max(len(pred_sequence), 1) if pred_sequence else 0.0
            
            # ✨ NEW: 多步差异性约束 - 强制相邻预测步有明显不同（解决日期间预测相同问题）
            loss_diversity = 0.0
            if len(pred_sequence) >= 2:
                for t in range(len(pred_sequence) - 1):
                    pred_t = pred_sequence[t].squeeze(1)        # [batch, C, H, W]
                    pred_t_next = pred_sequence[t+1].squeeze(1) # [batch, C, H, W]
                    # 计算相邻步的差异：应该尽可能大（鼓励动力演变）
                    diff = torch.abs(pred_t - pred_t_next)  # [batch, C, H, W]
                    # 目标：每步平均变化至少0.05（在[0,1]范围内）
                    # 如果变化太小，惩罚（目的是强制多样性）
                    mean_diff = diff.mean(dim=[1, 2, 3])  # [batch]
                    loss_diversity += torch.relu(0.05 - mean_diff).mean()
            loss_diversity = loss_diversity * 0.1  # 权重0.1
            
            # 谱正则化（约束K矩阵特征值）
            K_weight = model.K.weight
            eigs = torch.linalg.eigvals(K_weight)  # 这会自动追踪梯度！
            
            # ✨ CRITICAL: 确保特征值的损失直接取决于K_weight
            # 这强制梯度流回K矩阵
            loss_spectral = torch.mean(torch.relu(torch.abs(eigs) - config.EIGEN_MAX)) + \
                            torch.mean(torch.relu(config.EIGEN_MIN - torch.abs(eigs)))
            loss_spectral_weighted = config.W_SPECTRAL * loss_spectral
            
            # ✨ NEW: 强制梯度连接 - 直接对特征值取导数，即使W_SPECTRAL=0
            # 目的：无论config.W_SPECTRAL是多少，K矩阵都能收到梯度信号
            loss_k_spectral_forced = 0.01 * torch.mean(torch.abs(eigs - 0.95))  # 吸引特征值到0.95（稳定中心）
            
            # ✨ NEW: 显式K矩阵Frobenius范数正则化（确保K矩阵更新）
            K_frobenius = torch.norm(K_weight, p='fro') / (config.LATENT_DIM ** 2)
            loss_k_reg = 0.01 * K_frobenius  # Frobenius范数约束
            
            # 总损失反向传播（包含梯度约束）
            total_loss = loss_batch + loss_gradient + loss_spectral_weighted + loss_k_spectral_forced + loss_k_reg + loss_diversity
            total_loss.backward()
            
            # ✨ NEW: 监控K矩阵梯度（在裁剪之前获取）
            if model.K.weight.grad is not None:
                k_grad_norm = torch.norm(model.K.weight.grad).item()
            else:
                k_grad_norm = 0.0
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            
            # 累计损失
            total_loss_value = total_loss.item()
            total_loss += total_loss_value
            total_pred_loss += loss_batch.item()
            total_spectral_loss += loss_spectral_weighted.item()
        
        # 打印训练日志
        if (epoch+1) % 1 == 0:
            avg_loss = total_loss / len(train_loader)
            avg_pred = total_pred_loss / len(train_loader)
            avg_spec = total_spectral_loss / len(train_loader)
            max_eig = torch.max(torch.abs(eigs)).item()
            # ✨ NEW: 显示K矩阵梯度范数（用于诊断）
            print(f"Epoch {epoch+1:03d} | Total={avg_loss:.4f} | Pred={avg_pred:.4f} | Spec={avg_spec:.4f} | Max Eig={max_eig:.4f} | K_Grad={k_grad_norm:.6f}")

    # 5. 预测并导出结果
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    pred_list, true_list, mask_list = predict_and_export_n_days(
        model, train_dataset, device, config.OUTPUT_DIR, start_day_idx=config.PREDICT_START_DAY_IDX, n_days=7
    )
    print(f"\nDone! Results saved to: {config.OUTPUT_DIR}")
    
    # 6. 每日预测评估 - 科研规范 (原始物理单位)
    try:
        from eval_metrics_comprehensive import evaluate_daily_predictions_scientific
        
        # 过滤出有真实值的预测
        valid_pred = []
        valid_true = []
        valid_mask = []
        for pred, true, mask in zip(pred_list, true_list, mask_list):
            if true is not None and mask is not None:
                valid_pred.append(pred)
                valid_true.append(true)
                valid_mask.append(mask)
        
        if valid_pred:
            eval_output_dir = os.path.join(config.OUTPUT_DIR, "daily_evaluation")
            # 使用科研规范评估 - 同时计算原始物理单位和归一化指标
            eval_results = evaluate_daily_predictions_scientific(
                valid_pred, valid_true, valid_mask,
                global_min=train_dataset.global_min,
                global_max=train_dataset.global_max,
                output_dir=eval_output_dir
            )
            print(f"[OK] Scientific evaluation completed. Results saved to {eval_output_dir}")
        else:
            print("[WARN] No valid data for daily evaluation")
    except ImportError as e:
        print(f"[WARN] eval_metrics_comprehensive module not found: {e}")
    
    # 7. 保存模型
    model_path = os.path.join(config.OUTPUT_DIR, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[OK] Trained model saved to: {model_path}")

if __name__ == "__main__":
    main()