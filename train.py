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


def predict_and_export_n_days(model, dataset, device, output_dir, start_day_idx=0, n_days=7):
    """预测未来n天并导出GeoTIFF"""
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
    
    with torch.no_grad():
        preds_list = model.predict_future(x_current_tensor, u_sequence, steps=n_days)
        for i, pred_tensor in enumerate(preds_list):
            day = i + 1
            pred_img = pred_tensor.squeeze().cpu().numpy()
            filename = os.path.join(output_dir, f"Pred_Day_{day:02d}.tif")
            export_geotiff(filename, pred_img, dataset)
            print(f"  ✓ 已保存: {filename}")

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
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
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
    
    print(f"\n>>> 启动训练 (Control={config.USE_METEO_CONTROL}, Latent Dim={config.LATENT_DIM}) <<<")
    
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
                mask=mask[:, 0:1]   # [batch, 1, H, W] 空间掩码
            )
            
            # 谱正则化（约束K矩阵特征值）
            K_weight = model.K.weight
            eigs = torch.linalg.eigvals(K_weight)
            loss_spectral = torch.mean(torch.relu(torch.abs(eigs) - config.EIGEN_MAX)) + \
                            torch.mean(torch.relu(config.EIGEN_MIN - torch.abs(eigs)))
            loss_spectral_weighted = config.W_SPECTRAL * loss_spectral
            
            # 总损失反向传播
            total_loss = loss_batch + loss_spectral_weighted
            total_loss.backward()
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
            print(f"Epoch {epoch+1:03d} | Total={avg_loss:.4f} | Pred={avg_pred:.4f} | Spec={avg_spec:.4f} | Max Eig={max_eig:.4f}")

    # 5. 预测并导出结果
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    predict_and_export_n_days(model, train_dataset, device, config.OUTPUT_DIR, start_day_idx=0, n_days=7)
    print(f"\n任务完成! 结果保存在: {config.OUTPUT_DIR}")
    # 训练循环结束后添加
    model_path = os.path.join(config.OUTPUT_DIR, "trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ 训练好的模型已保存至: {model_path}")

if __name__ == "__main__":
    main()