import torch
import os

class Config:
    # Dataset Configuration
    DATASET_NAME = "cifar100"  # Options: cifar100, imagenet
    DATA_ROOT = "./data"
    NUM_CLASSES = 100
    
    # Dataset size in GB (approximate)
    DATASET_SIZE_GB = 1.6
    
    # Model Configuration
    MODEL_NAME = "resnet18"
    INPUT_SIZE = 32
    BATCH_SIZE = 128
    
    # Training Configuration
    EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    # Mixed Precision Training
    USE_AMP = True
    
    # Paths
    SAVE_DIR = "./checkpoints"
    FIGURES_DIR = "./figures"
    LOGS_DIR = "./logs"
    
    # Create directories
    for dir_path in [SAVE_DIR, FIGURES_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def display_info(cls):
        print("="*60)
        print("CONFIGURATION SETTINGS")
        print("="*60)
        print(f"Dataset: {cls.DATASET_NAME} ({cls.DATASET_SIZE_GB} GB)")
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print("="*60)
