import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from osgeo import gdal
from config import Config

class EnhancedPM25Dataset(Dataset):
    def __init__(self, data_dir, config: Config, augment=False):    
        self.config = config
        self.augment = augment
        
        # 加载全年文件路径（用于预测时访问）
        self.all_file_paths = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
        if not self.all_file_paths:
            raise FileNotFoundError(f"No .tif files found in {data_dir}")
        
        # 按配置范围筛选训练数据
        self.train_start_idx = config.TRAIN_START_IDX
        self.train_end_idx = config.TRAIN_END_IDX + 1  # +1 使其包含 end_idx
        self.file_paths = self.all_file_paths[self.train_start_idx:self.train_end_idx]
        
        print(f"---Dataset: {len(self.all_file_paths)} files (全年) | 训练数据: {len(self.file_paths)} files [{config.TRAIN_START_IDX}-{config.TRAIN_END_IDX}]. Res={config.IMG_SIZE}, Control={config.USE_METEO_CONTROL}---")
        
        # 读取地理参考信息
        ref_ds = gdal.Open(self.all_file_paths[0])
        self.orig_w, self.orig_h = ref_ds.RasterXSize, ref_ds.RasterYSize
        self.geo_transform = ref_ds.GetGeoTransform()
        self.projection = ref_ds.GetProjection()
        
        # 加载训练数据
        self.data = self._load_data()
        self.global_min, self.global_max = np.min(self.data), np.max(self.data)
        print(f"Data Loaded. Range: [{self.global_min:.2f}, {self.global_max:.2f}]")
        
        # 生成气象数据（按开关控制）
        self.meteo_data = self._generate_meteo_data()

    def _load_data(self):
        """加载并预处理TIFF数据（padding + resize）"""
        data = []
        max_side = max(self.orig_w, self.orig_h)
        for p in self.file_paths:  # 使用筛选后的训练数据文件
            ds = gdal.Open(p)
            arr = ds.GetRasterBand(1).ReadAsArray()
            # Padding到正方形
            canvas = np.zeros((max_side, max_side), dtype=np.float32)
            y_off, x_off = (max_side - self.orig_h) // 2, (max_side - self.orig_w) // 2
            canvas[y_off:y_off+self.orig_h, x_off:x_off+self.orig_w] = arr
            # Resize到目标尺寸
            resized = cv2.resize(canvas, (self.config.IMG_SIZE, self.config.IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            data.append(resized)
        data = np.array(data, dtype=np.float32)
        return np.nan_to_num(data, nan=0.0)

    def _generate_meteo_data(self):
        """生成气象控制数据（启用/禁用分支）"""
        if self.config.USE_METEO_CONTROL:
            # 启用：正弦波模拟周期性风场
            t = np.linspace(0, 4*np.pi, len(self.data))
            return np.stack([np.sin(t), np.cos(t)], axis=1).astype(np.float32)
        else:
            # 禁用：全0占位符
            return np.zeros((len(self.data), self.config.METEO_DIM), dtype=np.float32)

    def __len__(self):
        return len(self.data) - self.config.SEQUENCE_LENGTH

    def __getitem__(self, idx):
        """获取单样本：图像序列 + 气象序列 + 掩码"""
        # 1. 图像序列归一化
        seq = self.data[idx : idx + self.config.SEQUENCE_LENGTH]
        if self.global_max > self.global_min:
            seq = (seq - self.global_min) / (self.global_max - self.global_min)
        
        # 2. 气象序列
        meteo_seq = self.meteo_data[idx : idx + self.config.SEQUENCE_LENGTH]
        
        # 3. 数据增强
        if self.augment:
            if random.random() > 0.5:
                seq = np.flip(seq, axis=2).copy()
            if random.random() > 0.5:
                seq = np.flip(seq, axis=1).copy()
            if random.random() > 0.5:
                seq = np.rot90(seq, k=1, axes=(1,2)).copy()
        
        # 转Tensor
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(1)  # (T, 1, H, W)
        meteo_t = torch.tensor(meteo_seq, dtype=torch.float32)       # (T, 2)
        mask_t = (seq_t > self.config.VALID_DATA).float()
        
        return seq_t, meteo_t, mask_t

    def get_all_data(self):
        """获取全量归一化数据"""
        if self.global_max > self.global_min:
             return (self.data - self.global_min) / (self.global_max - self.global_min)
        return self.data