import os
import time
import torch
import json
from config import Config
from utils import set_seed, get_device
from data import EnhancedPM25Dataset
from models import ControlledKoopmanModel
from eval_strategies import sliding_window_rolling_test, iterative_optimize_decoder

def load_trained_model(config, device):
    """加载训练好的模型（需确保train.py训练后保存了模型）"""
    model = ControlledKoopmanModel(config).to(device)
    model_path = os.path.join(config.OUTPUT_DIR, "trained_model.pth")
    if os.path.exists(model_path):
        # 对齐train.py的权重加载逻辑，仅指定map_location
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 加载预训练模型: {model_path}")
    else:
        raise FileNotFoundError("请先运行train.py训练并保存模型！")
    return model

def main():
    # 1. 初始化配置/设备/种子（完全对齐train.py逻辑）
    config = Config()
    set_seed(42)
    device = get_device()
    
    # 2. 加载数据集和训练好的模型（GPU逻辑对齐train.py）
    os.makedirs(config.DATA_DIR, exist_ok=True)
    try:
        dataset = EnhancedPM25Dataset(config.DATA_DIR, config, augment=False)  # 评估时关闭数据增强
    except FileNotFoundError as e:
        print(f"数据加载失败：{e}")
        return
    
    # 【关键】先运行train.py并保存模型（需修改train.py最后添加模型保存逻辑）
    model = load_trained_model(config, device)
    
    # 3. 滑动窗口+滚动测试评估
    initial_metrics = sliding_window_rolling_test(
        model=model,
        dataset=dataset,
        window_size=config.SEQUENCE_LENGTH,
        step=1,
        device=device,
        start_idx=0
    )
    # --- 修复：聚合指标 + 空值保护 ---
    if not initial_metrics:
        print("\n=== 优化前解码效果指标 ===")
        print("⚠️ 无有效评估数据")
        initial_agg_metrics = {}
    else:
        initial_agg_metrics = aggregate_metrics(initial_metrics)
        print("\n=== 优化前解码效果指标 ===")
        for k, v in initial_agg_metrics.items():
            print(f"{k}: {v}")
    
    # 4. 迭代优化解码模块
    optimized_model, optimized_metrics = iterative_optimize_decoder(
        model, dataset, device, config,
        eval_metrics=initial_agg_metrics,  # 传入聚合后的指标
        optimize_epochs=config.OPTIMIZE_EPOCHS,
        lr=config.OPTIMIZE_LR
    )
    # --- 修复：聚合优化后指标 ---
    if not optimized_metrics:
        print("\n=== 优化后解码效果指标 ===")
        print("⚠️ 无有效优化后数据")
        optimized_agg_metrics = {}
    else:
        optimized_agg_metrics = aggregate_metrics(optimized_metrics)
        print("\n=== 优化后解码效果指标 ===")
        for k, v in optimized_agg_metrics.items():
            print(f"{k}: {v}")
    
    # 5. 保存评估结果（替换为聚合后的指标）
    eval_result = {
        "config": {
            "window_size": config.SEQUENCE_LENGTH,  # 原代码错用EVAL_WINDOW_SIZE，需对齐
            "step_size": 1,
            "optimize_epochs": config.OPTIMIZE_EPOCHS
        },
        "initial_metrics": initial_agg_metrics,
        "optimized_metrics": optimized_agg_metrics
    }
    result_path = os.path.join(config.OUTPUT_DIR, "decoding_evaluation.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=4)
    print(f"\n✅ 评估结果已保存至: {result_path}")
    
    # 可选：保存优化后的模型（对齐train.py的模型保存逻辑，指定map_location=device）
    optimized_model_path = os.path.join(config.OUTPUT_DIR, "optimized_model.pth")
    torch.save(optimized_model.state_dict(), optimized_model_path)
    print(f"✅ 优化后的模型已保存至: {optimized_model_path}")

if __name__ == "__main__":
    main()