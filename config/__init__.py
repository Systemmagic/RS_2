class Config:
    """中央配置类 - 仅存储参数，解耦业务逻辑"""
    # 数据参数
    IMG_SIZE = 256           
    SEQUENCE_LENGTH = 4      
    VALID_DATA = 0.001 
    USE_METEO_CONTROL = True  # 气象控制开关
    METEO_DIM = 2            # 气象数据维度 (U-wind, V-wind)
    
    # 架构参数
    LATENT_DIM = 2048        # 潜变量维度
    
    # 训练参数
    BATCH_SIZE = 4           
    EPOCHS = 150             
    LEARNING_RATE = 1e-4     
    WEIGHT_DECAY = 1e-5      # 优化器权重衰减
    GRAD_CLIP_NORM = 1.0     # 梯度裁剪阈值
    
    # 损失函数权重
    W_BASE_L1 = 1.0          
    W_GRAD = 2.0             
    W_TOPK = 5.0             
    TOPK_RATIO = 0.15 
    
    # 动力学参数
    W_SPECTRAL = 0.005
    EIGEN_MAX = 2.0          
    EIGEN_MIN = 0.8          

    # 路径参数（解耦硬编码路径）
    DATA_DIR = "data\CZT_PM25_2023"
    OUTPUT_DIR = "result\CZT_PM25_2023_ControlledKoopman"

__all__ = ['Config']