import torch
import numpy as np
import os
from tqdm import tqdm
from eval_metrics import calculate_decoding_metrics, aggregate_metrics

# eval_strategies.py
import torch
import numpy as np
from tqdm import tqdm

def sliding_window_rolling_test(
    model, 
    dataset, 
    window_size=4,  
    step=1, 
    device="cuda", 
    start_idx=0
):
    all_metrics = []
    total_samples = len(dataset)
    total_windows = max(0, total_samples - window_size + 1)
    start_idx = max(0, min(start_idx, total_windows - 1)) if total_windows > 0 else 0
    
    if total_windows == 0:
        print(f"⚠️ 无有效窗口：样本数={total_samples}, 窗口大小={window_size}")
        return all_metrics
    
    print(f">>> 滚动测试 (窗口={window_size}, 步长={step}, 总窗口数={total_windows}) <<<")
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(start_idx, total_windows, step), desc="滚动测试进度"):
            # --- 修复1：窗口数据加载后直接移到GPU ---
            end_idx = i + window_size
            window_frames = [dataset[idx] for idx in range(i, end_idx)]
            seq_img_list = []
            for frame in window_frames:
                img = frame[0]
                if len(img.shape) == 2:
                    img = img.unsqueeze(0)
                img = img.to(device)  # 新增：单帧加载后立即移到GPU
                seq_img_list.append(img)
            seq_img = torch.stack(seq_img_list).unsqueeze(0)  # 移除原to(device)，避免重复移动
            seq_meteo = torch.stack([frame[1].to(device) for frame in window_frames]).unsqueeze(0)
            mask = torch.stack([frame[2].to(device) for frame in window_frames]).unsqueeze(0)
            
            # 2. 核心修复：强制确保x_curr为4维
            x_curr = seq_img[:, 0]  # 取第0帧
            # 兜底：无论中间步骤如何，最终确保维度是[batch, channels, H, W]
            while len(x_curr.shape) > 4:
                x_curr = x_curr.squeeze(1)  # 挤压多余的seq_len维度
            if len(x_curr.shape) == 3:  # 无batch维：[1,256,256] → [1,1,256,256]
                x_curr = x_curr.unsqueeze(0)
            print(f"✅ x_curr维度验证: {x_curr.shape}")  # 必须输出[1,1,256,256]
            
            # 3. 模型前向（此时维度必正确）
            z_curr = model.encoder(x_curr)
            preds = []
            
            # 4. 递归预测（匹配train.py的循环逻辑）
            for t in range(1, window_size):
                u_curr = seq_meteo[:, t-1]
                control_term = model.compute_control(u_curr)
                z_next_pred = model.K(z_curr) + control_term
                delta = model.decoder(z_next_pred)
                x_next_pred = torch.relu(x_curr + delta)
                
                preds.append(x_next_pred)
                z_curr = z_next_pred
                x_curr = x_next_pred
            
            # 5. 指标计算
            preds_tensor = torch.cat(preds, dim=0)
            targets_tensor = seq_img[:, 1:].squeeze(0)
            mask_tensor = mask[:, 1:].squeeze(0)
            
            metrics = calculate_decoding_metrics(
                preds_tensor,  # 已在GPU
                targets_tensor,  # 已在GPU
                mask_tensor  # 已在GPU
            )
            all_metrics.append(metrics)
    
    return all_metrics

def iterative_optimize_decoder(
    model, dataset, device, config, 
    eval_metrics: dict, optimize_epochs: int = 10, lr: float = 1e-4
) -> tuple:
    """
    迭代优化解码模块：针对解码误差反向优化decoder参数
    :param model: 训练好的模型
    :param eval_metrics: 初始评估指标（用于对比）
    :param optimize_epochs: 迭代优化轮数
    :param lr: 优化学习率
    :return: 优化后的模型、优化后指标
    """
    print(f"\n>>> 启动解码模块迭代优化 (优化轮数={optimize_epochs}) <<<")
    # 仅优化decoder参数
    decoder_optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()  # 聚焦解码误差
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    model.train()
    for epoch in range(optimize_epochs):
        total_decoder_loss = 0
        for seq_img, seq_meteo, mask in test_loader:
            seq_img = seq_img.to(device)  # 确保最终在GPU
            seq_meteo = seq_meteo.to(device)
            mask = mask.to(device)
            
            decoder_optimizer.zero_grad()
            loss = 0
            
            x_curr = seq_img[:, 0]
            z_curr = model.encoder(x_curr)
            for t in range(1, config.SEQUENCE_LENGTH):
                u_curr = seq_meteo[:, t-1]
                control_term = model.compute_control(u_curr)
                z_next_pred = model.K(z_curr) + control_term
                
                # 解码+计算误差（仅优化decoder）
                delta = model.decoder(z_next_pred)
                x_next_pred = torch.relu(x_curr + delta)
                x_target = seq_img[:, t]
                m_target = mask[:, t]
                
                # 仅优化解码损失
                decoder_loss = criterion(x_next_pred[m_target.bool()], x_target[m_target.bool()])
                loss += decoder_loss
                
                x_curr = x_next_pred
                z_curr = z_next_pred
            
            loss.backward()
            decoder_optimizer.step()
            total_decoder_loss += loss.item()
        
        avg_loss = total_decoder_loss / len(test_loader)
        print(f"优化Epoch {epoch+1:02d} | 解码损失={avg_loss:.4f}")
    
    # 优化后重新评估
    model.eval()
    optimized_metrics = sliding_window_rolling_test(model, dataset, device, config)
    return model, optimized_metrics