import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from itertools import cycle

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Validate a trained medicinal herb classifier')
    parser.add_argument('--model_path', type=str, default="models/herb_classifier.h5",
                        help='Path to the trained model')
    parser.add_argument('--validation_data', type=str, default="dataset/medicinal_leaf_dataset",
                        help='Path to validation data directory')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for validation (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation (default: 32)')
    parser.add_argument('--output_dir', type=str, default="validation_results",
                        help='Directory to save validation results')
    return parser.parse_args()

def create_directories(args):
    """Create necessary directories if they don't exist"""
    os.makedirs(args.output_dir, exist_ok=True)

def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_validation_data(args):
    """Prepare validation data"""
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    try:
        validation_generator = validation_datagen.flow_from_directory(
            args.validation_data,
            target_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return validation_generator
    except Exception as e:
        print(f"Error preparing validation data: {e}")
        return None

def validate_model(model, validation_generator, args):
    """Validate the model"""
    print("Starting model validation...")
    
    # Evaluate the model
    evaluation = model.evaluate(validation_generator, verbose=1)
    
    # Print evaluation metrics
    print(f"\nEvaluation metrics:")
    print(f"Loss: {evaluation[0]:.4f}")
    print(f"Accuracy: {evaluation[1]:.4f}")
    
    # Get class labels
    class_labels = list(validation_generator.class_indices.keys())
    num_classes = len(class_labels)
    
    # Make predictions
    validation_generator.reset()  # Reset generator to beginning
    y_pred_prob = model.predict(validation_generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get true labels
    y_true = validation_generator.classes
    
    # Ensure we only use the valid part of the predictions 
    # (in case the number of samples is not a multiple of batch_size)
    n_valid_samples = validation_generator.n
    y_pred = y_pred[:n_valid_samples]
    y_pred_prob = y_pred_prob[:n_valid_samples]
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Save classification report
    report_path = os.path.join(args.output_dir, 'validation_classification_report.csv')
    report_df.to_csv(report_path)
    
    # Print classification report summary
    print("\nClassification Report Summary:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    # Save overall metrics as JSON
    metrics = {
        'loss': float(evaluation[0]),
        'accuracy': float(evaluation[1]),
        'macro_avg_precision': float(report['macro avg']['precision']),
        'macro_avg_recall': float(report['macro avg']['recall']),
        'macro_avg_f1': float(report['macro avg']['f1-score']),
    }
    
    metrics_path = os.path.join(args.output_dir, 'validation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, 'validation_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate per-class metrics visualization
    plt.figure(figsize=(12, 8))
    
    class_metrics = report_df.iloc[:-3]  # Exclude micro avg, macro avg, weighted avg
    
    # Sort by F1-score
    class_metrics = class_metrics.sort_values(by='f1-score', ascending=False)
    
    plt.barh(class_metrics.index, class_metrics['f1-score'], color='skyblue')
    plt.xlabel('F1-Score')
    plt.ylabel('Class')
    plt.title('F1-Score by Class')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, 1.0)
    
    # Save per-class metrics
    metrics_viz_path = os.path.join(args.output_dir, 'validation_class_f1_scores.png')
    plt.savefig(metrics_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # If number of classes is manageable, generate ROC curves
    if num_classes <= 15:  # limit for readability
        generate_roc_curves(y_true, y_pred_prob, class_labels, args)
    
    # Generate precision-recall curves
    if num_classes <= 15:  # limit for readability
        generate_precision_recall_curves(y_true, y_pred_prob, class_labels, args)
    
    return metrics

def generate_roc_curves(y_true, y_pred_prob, class_labels, args):
    """Generate ROC curves for each class"""
    # Convert true labels to one-hot encoding
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))
    
    # Set up plot
    plt.figure(figsize=(12, 10))
    
    # Plot ROC curve for each class
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(len(class_labels)), colors):
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot parameters
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    
    # Save ROC curves
    roc_path = os.path.join(args.output_dir, 'validation_roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_precision_recall_curves(y_true, y_pred_prob, class_labels, args):
    """Generate precision-recall curves for each class"""
    # Convert true labels to one-hot encoding
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))
    
    # Set up plot
    plt.figure(figsize=(12, 10))
    
    # Plot precision-recall curve for each class
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for i, color in zip(range(len(class_labels)), colors):
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        avg_precision = np.mean(precision)
        
        # Plot precision-recall curve
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{class_labels[i]} (AP = {avg_precision:.2f})')
    
    # Set plot parameters
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    
    # Save precision-recall curves
    pr_path = os.path.join(args.output_dir, 'validation_precision_recall_curves.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the validation pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    # Create necessary directories
    create_directories(args)
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        return
    
    print(f"Loaded model from {args.model_path}")
    
    # Prepare validation data
    validation_generator = prepare_validation_data(args)
    if validation_generator is None:
        return
    
    # Validate model
    metrics = validate_model(model, validation_generator, args)
    
    print("\nValidation complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()