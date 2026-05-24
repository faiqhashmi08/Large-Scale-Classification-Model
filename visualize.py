import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from config import Config
import os

class Visualizer:
    def __init__(self):
        self.figures_dir = Config.FIGURES_DIR
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Loss plot
        ax1.plot(epochs, history['train_losses'], 'o-', label='Training Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, history['val_losses'], 's-', label='Validation Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, history['train_accuracies'], 'o-', label='Training Accuracy', linewidth=2, markersize=4)
        ax2.plot(epochs, history['val_accuracies'], 's-', label='Validation Accuracy', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Training history plot saved to {self.figures_dir}/training_history.png")
    
    def plot_confusion_matrix(self, y_true, y_pred, max_classes=20):
        """Plot confusion matrix heatmap"""
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # For large number of classes, show only first N classes
        if cm.shape[0] > max_classes:
            cm = cm[:max_classes, :max_classes]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=[f'C{i}' for i in range(min(max_classes, cm.shape[0]))],
                    yticklabels=[f'C{i}' for i in range(min(max_classes, cm.shape[0]))],
                    cbar_kws={'label': 'Number of Predictions'})
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title(f'Confusion Matrix (First {max_classes} Classes)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrix saved to {self.figures_dir}/confusion_matrix.png")
    
    def plot_class_accuracy(self, y_true, y_pred):
        """Plot per-class accuracy bar chart"""
        cm = confusion_matrix(y_true, y_pred)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Sort by accuracy
        sorted_indices = np.argsort(class_accuracy)
        sorted_accuracy = class_accuracy[sorted_indices]
        
        # Show top 15 and bottom 15
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Worst performing classes
        worst_classes = sorted_indices[:15]
        worst_acc = sorted_accuracy[:15]
        ax1.barh(range(len(worst_classes)), worst_acc * 100, color='red', alpha=0.7)
        ax1.set_yticks(range(len(worst_classes)))
        ax1.set_yticklabels([f'Class {c}' for c in worst_classes])
        ax1.set_xlabel('Accuracy (%)')
        ax1.set_title('15 Worst Performing Classes', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Best performing classes
        best_classes = sorted_indices[-15:][::-1]
        best_acc = sorted_accuracy[-15:][::-1]
        ax2.barh(range(len(best_classes)), best_acc * 100, color='green', alpha=0.7)
        ax2.set_yticks(range(len(best_classes)))
        ax2.set_yticklabels([f'Class {c}' for c in best_classes])
        ax2.set_xlabel('Accuracy (%)')
        ax2.set_title('15 Best Performing Classes', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/class_accuracy.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Class accuracy plot saved to {self.figures_dir}/class_accuracy.png")
        
        # Print statistics
        print(f"\n📊 Class Accuracy Statistics:")
        print(f"   Mean Accuracy: {np.mean(class_accuracy)*100:.2f}%")
        print(f"   Median Accuracy: {np.median(class_accuracy)*100:.2f}%")
        print(f"   Std Deviation: {np.std(class_accuracy)*100:.2f}%")
        print(f"   Best Class: Class {best_classes[0]} ({best_acc[0]*100:.2f}%)")
        print(f"   Worst Class: Class {worst_classes[0]} ({worst_acc[0]*100:.2f}%)")
    
    def plot_sample_predictions(self, images, true_labels, pred_labels, probabilities, num_samples=8):
        """Display sample predictions with confidence scores"""
        fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
        axes = axes.flatten()
        
        # Denormalization parameters for CIFAR-100
        mean = np.array([0.5071, 0.4867, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
        
        for i in range(num_samples):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = std * img + mean  # Denormalize
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            color = 'green' if pred_labels[i] == true_labels[i] else 'red'
            title = f'True: {true_labels[i]}\nPred: {pred_labels[i]}\nConf: {probabilities[i]:.2f}'
            axes[i].set_title(title, color=color, fontsize=9)
            axes[i].axis('off')
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/sample_predictions.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Sample predictions saved to {self.figures_dir}/sample_predictions.png")
    
    def plot_performance_summary(self, history, results):
        """Create comprehensive performance summary dashboard"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Training History
        ax1 = plt.subplot(2, 2, 1)
        epochs = range(1, len(history['train_losses']) + 1)
        ax1.plot(epochs, history['train_losses'], 'o-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_losses'], 's-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy History
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(epochs, history['train_accuracies'], 'o-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, history['val_accuracies'], 's-', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance Metrics Table
        ax3 = plt.subplot(2, 2, 3)
        ax3.axis('tight')
        ax3.axis('off')
        metrics_data = [
            ['Metric', 'Value'],
            ['Best Validation Accuracy', f"{history['best_val_acc']:.2f}%"],
            ['Total Training Time', f"{history['training_time']/60:.2f} min"],
            ['Final Train Loss', f"{history['train_losses'][-1]:.4f}"],
            ['Final Val Loss', f"{history['val_losses'][-1]:.4f}"],
            ['Dataset Size', f"{Config.DATASET_SIZE_GB} GB"],
            ['Model', Config.MODEL_NAME],
            ['Device', Config.DEVICE.upper()]
        ]
        table = ax3.table(cellText=metrics_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax3.set_title('Performance Summary', fontweight='bold')
        
        # 4. Classification Report
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('tight')
        ax4.axis('off')
        report = classification_report(results['y_true'][:100], results['y_pred'][:100], 
                                       output_dict=True, zero_division=0)
        report_text = f"Overall Accuracy: {results['accuracy']:.2f}%\n"
        report_text += f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}\n"
        report_text += f"Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}"
        ax4.text(0.1, 0.5, report_text, fontsize=12, verticalalignment='center')
        ax4.set_title('Classification Metrics', fontweight='bold')
        
        plt.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.figures_dir}/performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Performance dashboard saved to {self.figures_dir}/performance_dashboard.png")
