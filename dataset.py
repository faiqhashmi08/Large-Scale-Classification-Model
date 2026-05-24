import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from config import Config

class DataProcessor:
    """Handle dataset collection, preprocessing, and loading"""
    
    def __init__(self):
        self.transform_train = None
        self.transform_test = None
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup data preprocessing pipelines"""
        # For CIFAR-100 (32x32 images)
        if Config.INPUT_SIZE == 32:
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
            
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
        
        # For larger datasets (224x224 images)
        else:
            self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def load_cifar100(self):
        """Load CIFAR-100 dataset (1.6 GB)"""
        print("\n📥 Downloading CIFAR-100 dataset (1.6 GB)...")
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=Config.DATA_ROOT,
            train=True,
            download=True,
            transform=self.transform_train
        )
        
        test_dataset = torchvision.datasets.CIFAR100(
            root=Config.DATA_ROOT,
            train=False,
            download=True,
            transform=self.transform_test
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Testing samples: {len(test_dataset)}")
        print(f"   Total size: ~{Config.DATASET_SIZE_GB} GB")
        
        return train_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, test_dataset):
        """Create data loaders for training and testing"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if Config.DEVICE == "cuda" else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if Config.DEVICE == "cuda" else False
        )
        
        return train_loader, test_loader
    
    def get_dataset_info(self):
        """Display dataset information"""
        return {
            'name': Config.DATASET_NAME,
            'size_gb': Config.DATASET_SIZE_GB,
            'num_classes': Config.NUM_CLASSES,
            'input_shape': (3, Config.INPUT_SIZE, Config.INPUT_SIZE)
        }
