import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from config import Config
from models.decoder import ResNetDecoder
from .metrics import compute_all_metrics
from data.utils import load_test_data  # 假设data/utils有测试数据加载函数

class DecoderEvaluator:
    """解码器评估器：封装评估流程，包含指标计算、结果可视化、报告生成"""
    def __init__(self, config: Config, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.test_dataloader = self._load_test_data()
        self.eval_results = {}

    def _load_model(self, model_path: str) -> ResNetDecoder:
        """加载预训练解码器模型"""
        model = ResNetDecoder(self.config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["decoder_state_dict"])
        model.eval()
        return model

    def _load_test_data(self) -> torch.utils.data.DataLoader:
        """加载测试数据集（需匹配项目data模块的加载逻辑）"""
        # 需根据项目实际数据加载逻辑实现，示例如下：
        test_dataset = load_test_data(
            data_dir=self.config.TEST_DATA_DIR,
            img_size=self.config.IMG_SIZE,
            latent_dim=self.config.LATENT_DIM
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        return test_dataloader

    @torch.no_grad()
    def evaluate_batch(self, latent_z: torch.Tensor, target_map: torch.Tensor) -> Dict[str, float]:
        """单批次评估：输入潜向量，输出评估指标"""
        latent_z = latent_z.to(self.device)
        target_map = target_map.to(self.device)
        
        # 解码器前向推理
        pred_map = self.model(latent_z)
        
        # 计算所有指标
        metrics = compute_all_metrics(pred_map, target_map)
        return metrics

    def evaluate_full_dataset(self) -> Dict[str, float]:
        """全数据集评估：返回平均指标"""
        all_metrics = []
        for batch_idx, (latent_z, target_map) in enumerate(self.test_dataloader):
            batch_metrics = self.evaluate_batch(latent_z, target_map)
            all_metrics.append(batch_metrics)
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} evaluated | MSE: {batch_metrics['MSE']:.4f} | PSNR: {batch_metrics['PSNR']:.2f}")
        
        # 计算所有批次的平均指标
        avg_metrics = {
            metric: np.mean([bm[metric] for bm in all_metrics])
            for metric in all_metrics[0].keys()
        }
        self.eval_results = avg_metrics
        return avg_metrics

    def visualize_prediction(self, save_dir: str, num_samples: int = 5) -> None:
        """可视化预测结果与真实标签对比"""
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        
        with torch.no_grad():
            for idx, (latent_z, target_map) in enumerate(self.test_dataloader):
                if idx >= num_samples:
                    break
                latent_z = latent_z.to(self.device)
                target_map = target_map.to(self.device)
                pred_map = self.model(latent_z)
                
                # 可视化单样本（取批次第一个）
                pred = pred_map[0].squeeze().cpu().numpy()
                target = target_map[0].squeeze().cpu().numpy()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                ax1.imshow(pred, cmap="viridis")
                ax1.set_title(f"Predicted Map (Batch {idx})")
                ax1.axis("off")
                
                ax2.imshow(target, cmap="viridis")
                ax2.set_title(f"Target Map (Batch {idx})")
                ax2.axis("off")
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"pred_vs_target_batch_{idx}.png"))
                plt.close()

    def save_eval_report(self, report_path: str) -> None:
        """保存评估报告到文件"""
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("Decoder Evaluation Report\n")
            f.write("="*50 + "\n")
            f.write(f"Model Path: {self.config.MODEL_SAVE_PATH}\n")
            f.write(f"Test Data Dir: {self.config.TEST_DATA_DIR}\n")
            f.write(f"Evaluation Time: {torch.cuda.get_device_name() if self.device.type == 'cuda' else 'CPU'}\n")
            f.write("\nMetrics Summary:\n")
            for metric, value in self.eval_results.items():
                f.write(f"{metric}: {value:.4f}\n")