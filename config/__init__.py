class Config:
    """Central configuration - parameter storage only, decoupled from business logic"""
    # Data parameters
    IMG_SIZE = 256           
    SEQUENCE_LENGTH = 4      
    VALID_DATA = 0.001 
    USE_METEO_CONTROL = False  # Meteo control switch
    METEO_DIM = 2            # Meteo data dimension (U-wind, V-wind)
    
    # Architecture parameters
    LATENT_DIM = 256        # Increased from 128 to capture spatial details
    
    # Training parameters
    BATCH_SIZE = 4           
    EPOCHS = 10            # 10 epochs optimal - higher causes overfitting
    LEARNING_RATE = 5e-4     # Increased from 1e-4 for faster convergence
    WEIGHT_DECAY = 1e-5      # Optimizer weight decay
    GRAD_CLIP_NORM = 1.0     # Gradient clipping threshold
    
    # Loss function weights
    W_BASE_L1 = 0.5          # Reduced from 1.0 to allow larger value fluctuations
    W_GRAD = 2.0             
    W_TOPK = 2.0             # Focus on top errors
    TOPK_RATIO = 0.25        # Top 25% pixel errors
    
    # Dynamics parameters
    W_SPECTRAL = 0.15        # Enhanced stability constraint
    EIGEN_MAX = 10000         # Tightened from 2.0
    EIGEN_MIN = 0
    
    # Koopman loss weights
    ALPHA1 = 0.001          # Light regularization on Koopman
    ALPHA2 = 0.001          # Weaker L-inf constraint
    ALPHA3 = 0.0001         # Minimal regularization
    
    # Spatial detail preservation
    GRADIENT_LOSS_WEIGHT = 1.0     # Strong gradient constraint for spatial structure
    EXTREME_VALUE_WEIGHT = 10.0     # Weight boost for extreme values
    CLAMP_OUTPUT = True            # Keep output in [0,1]
    
    # Evaluation config
    #EVAL_WINDOW_SIZE = 7    # Rolling test window size
    #EVAL_STEP_SIZE = 1      # Window sliding step
    #TEST_START_IDX = 20     # Test set start index
    #OPTIMIZE_EPOCHS = 10    # Decode module optimization rounds
    #OPTIMIZE_LR = 1e-4      # Decode optimization learning rate
    
    # Training data range config
    TRAIN_START_IDX = 0      # Training data start index (0=2023-01-01)
    TRAIN_END_IDX = 363      # Training data end index (363=2023-12-31, inclusive)
                             # Includes prediction range to avoid extrapolation
    
    # Prediction config
    PREDICT_START_DAY_IDX = 0  # Start day for prediction (180=2023-06-30)

    # Path parameters
    DATA_DIR = "data\\CZT_PM25_2023"
    OUTPUT_DIR = "result\\experiment_3"

__all__ = ['Config']