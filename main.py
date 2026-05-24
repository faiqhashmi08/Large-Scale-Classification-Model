import torch
import numpy as np
import random
from config import Config
from dataset import DataProcessor
from model import create_model
from train import Trainer
from visualize import Visualizer
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("🎯 LARGE-SCALE IMAGE CLASSIFICATION PROJECT")
    print("="*60)
    
    # Display configuration
    Config.display_info()
    
    # Set seed
    set_seed(42)
    
    # Step 1-3: Load and preprocess dataset
    print("\n📊 STEP 1-3: Loading and Preprocessing Dataset")
    print("-"*40)
    data_processor = DataProcessor()
    train_dataset, test_dataset = data_processor.load_cifar100()
    train_loader, test_loader = data_processor.create_data_loaders(train_dataset, test_dataset)
    
    # Display dataset info
    dataset_info = data_processor.get_dataset_info()
    print(f"\n📁 Dataset Information:")
    print(f"   Name: {dataset_info['name']}")
    print(f"   Size: {dataset_info['size_gb']} GB")
    print(f"   Classes: {dataset_info['num_classes']}")
    print(f"   Input Shape: {dataset_info['input_shape']}")
    
    # Step 4: Create classification model
    print("\n🤖 STEP 4: Implementing Classification Model")
    print("-"*40)
    model = create_model(model_type=Config.MODEL_NAME, num_classes=Config.NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model Created: {Config.MODEL_NAME}")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    
    # Train the model
    trainer = Trainer(model, train_loader, test_loader)
    history = trainer.train()
    
    # Load best model for evaluation
    checkpoint = torch.load(f"{Config.SAVE_DIR}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\n📈 Final Evaluation on Test Set")
    print("-"*40)
    val_loss, val_acc, all_preds, all_labels = trainer.validate()
    
    # Step 5: Generate visualizations
    print("\n📊 STEP 5: Generating Results and Graphs")
    print("-"*40)
    visualizer = Visualizer()
    
    # Get sample images for visualization
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:8]
    sample_labels = sample_labels[:8]
    
    model.eval()
    with torch.no_grad():
        sample_outputs = model(sample_images.to(Config.DEVICE))
        sample_probs = torch.softmax(sample_outputs, dim=1)
        sample_preds = sample_outputs.argmax(dim=1)
        sample_confidences = sample_probs.max(dim=1)[0].cpu().numpy()
    
    # Generate all plots
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(all_labels, all_preds, max_classes=20)
    visualizer.plot_class_accuracy(all_labels, all_preds)
    visualizer.plot_sample_predictions(
        sample_images, sample_labels.cpu().numpy(), 
        sample_preds.cpu().numpy(), sample_confidences
    )
    
    # Prepare results for dashboard
    results = {
        'y_true': all_labels,
        'y_pred': all_preds,
        'accuracy': val_acc
    }
    visualizer.plot_performance_summary(history, results)
    
    # Step 6-7: Save results and summary
    print("\n💾 Saving Final Results")
    print("-"*40)
    
    # Save classification report
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    import json
    final_results = {
        'best_accuracy': float(history['best_val_acc']),
        'final_test_accuracy': float(val_acc),
        'total_training_time_minutes': float(history['training_time'] / 60),
        'model_parameters': total_params,
        'dataset_size_gb': Config.DATASET_SIZE_GB,
        'epochs_trained': Config.EPOCHS,
        'classification_report': {
            'accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }
    }
    
    with open(f"{Config.LOGS_DIR}/final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("✅ Results saved to:")
    print(f"   - Figures: {Config.FIGURES_DIR}/")
    print(f"   - Model: {Config.SAVE_DIR}/best_model.pth")
    print(f"   - Logs: {Config.LOGS_DIR}/final_results.json")
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📊 Final Results Summary:")
    print(f"   ✅ Dataset: {Config.DATASET_NAME} ({Config.DATASET_SIZE_GB} GB)")
    print(f"   ✅ Model: {Config.MODEL_NAME}")
    print(f"   ✅ Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    print(f"   ✅ Final Test Accuracy: {val_acc:.2f}%")
    print(f"   ✅ Total Training Time: {history['training_time']/60:.2f} minutes")
    print(f"\n📁 All outputs saved in:")
    print(f"   - Figures: {Config.FIGURES_DIR}")
    print(f"   - Model Checkpoints: {Config.SAVE_DIR}")
    print(f"   - Logs: {Config.LOGS_DIR}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
