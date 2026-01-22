class Config:
    """中央配置类 - 仅存储参数，解耦业务逻辑"""
    # 数据参数
    IMG_SIZE = 256           
    SEQUENCE_LENGTH = 4      
    VALID_DATA = 0.001 
    USE_METEO_CONTROL = True  # 气象控制开关
    METEO_DIM = 2            # 气象数据维度 (U-wind, V-wind)
    
    # 架构参数
    LATENT_DIM = 32        # 潜变量维度
    
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
    
    # Koopman损失权重
    ALPHA1 = 0.1            # 重建损失和预测损失权重
    ALPHA2 = 0.05           # L∞损失权重
    ALPHA3 = 0.01           # L2正则化权重
    
    # 评估配置
    EVAL_WINDOW_SIZE = 7    # 滚动测试窗口大小
    EVAL_STEP_SIZE = 1      # 窗口滑动步长
    TEST_START_IDX = 20    # 测试集起始索引（避开训练集）
    OPTIMIZE_EPOCHS = 10    # 解码模块迭代优化轮数
    OPTIMIZE_LR = 1e-4      # 解码优化学习率

    # 路径参数（解耦硬编码路径）
    DATA_DIR = "data\PM25_1_2"
    OUTPUT_DIR = "result\experiment_1"

__all__ = ['Config']