import torch
import random
import numpy as np

def set_seed(seed=42):
    """设置随机种子，确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False

def get_device():
    """强制优先使用GPU，无GPU则用CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # 指定第一张GPU
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        # 清空GPU缓存，避免显存占用
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("⚠️ 无可用GPU，使用CPU（训练/评估会极慢）")
    return device