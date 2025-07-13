import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
from PIL import Image
import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test a trained medicinal herb classifier on new images')
    parser.add_argument('--model_path', type=str, default="models/herb_classifier.h5",
                        help='Path to the trained model')
    parser.add_argument('--test_dir', type=str, default="test_images",
                        help='Directory containing test images')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for testing (default: 224)')
    parser.add_argument('--output_dir', type=str, default="results",
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for testing (default: 32)')
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

def load_class_labels():
    """Load class labels from JSON file"""
    try:
        with open('data/class_labels.json', 'r') as f:
            class_labels = json.load(f)
        return class_labels
    except Exception as e:
        print(f"Error loading class labels: {e}")
        return None

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image for prediction"""
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = np.array(img) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def predict_single_image(model, image_path, class_labels, image_size):
    """Make prediction on a single image"""
    try:
        # Preprocess the image
        img_array = preprocess_image(image_path, target_size=(image_size, image_size))
        
        if img_array is None:
            return None, None
        
        # Make prediction
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        if class_idx < len(class_labels):
            class_name = class_labels[class_idx]
        else:
            class_name = "Unknown"
        
        return class_name, confidence
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None, None

def test_on_directory(args):
    """Test the model on all images in a directory"""
    # Load the model
    model = load_model(args.model_path)
    if model is None:
        return
    
    # Load class labels
    class_labels = load_class_labels()
    if class_labels is None:
        return
    
    print(f"Loaded model from {args.model_path}")
    print(f"Testing on images in {args.test_dir}")
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Test directory {args.test_dir} does not exist.")
        return
    
    # Get all image files from test directory
    image_extensions = ['.jpg', '.jpeg', '.png']
    test_images = []
    
    # Walk through all subdirectories
    for root, _, files in os.walk(args.test_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print(f"No image files found in {args.test_dir}")
        return
    
    print(f"Found {len(test_images)} images for testing")
    
    # Create result dataframe
    results = []
    
    # Process each image
    for img_path in test_images:
        # Get true label from directory name if available
        dir_name = os.path.basename(os.path.dirname(img_path))
        true_label = dir_name if dir_name in class_labels else "Unknown"
        
        # Make prediction
        pred_label, confidence = predict_single_image(model, img_path, class_labels, args.image_size)
        
        if pred_label is not None:
            results.append({
                'image_path': img_path,
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    csv_path = os.path.join(args.output_dir, 'test_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Calculate accuracy if true labels are available
    if 'true_label' in results_df.columns and 'Unknown' not in results_df['true_label'].values:
        accuracy = (results_df['true_label'] == results_df['predicted_label']).mean()
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Generate classification report
        try:
            report = classification_report(
                results_df['true_label'], 
                results_df['predicted_label'],
                output_dict=True
            )
            
            # Save classification report
            report_path = os.path.join(args.output_dir, 'test_classification_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            # Print summary
            print("\nClassification Report Summary:")
            print(f"Accuracy: {report['accuracy']:.4f}")
            print(f"Macro Avg Precision: {report['macro avg']['precision']:.4f}")
            print(f"Macro Avg Recall: {report['macro avg']['recall']:.4f}")
            print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
            
            # Generate confusion matrix
            labels = sorted(results_df['true_label'].unique())
            cm = confusion_matrix(
                results_df['true_label'],
                results_df['predicted_label'],
                labels=labels
            )
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title('Confusion Matrix')
            
            # Save confusion matrix plot
            plt.tight_layout()
            cm_path = os.path.join(args.output_dir, 'test_confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {cm_path}")
            
        except Exception as e:
            print(f"Error generating classification report: {e}")
    
    # Visualize some predictions
    visualize_predictions(results_df, args)

def visualize_predictions(results_df, args):
    """Visualize some predictions with images"""
    try:
        # Sample up to 10 images to visualize
        sample_size = min(10, len(results_df))
        samples = results_df.sample(sample_size) if sample_size > 1 else results_df
        
        plt.figure(figsize=(15, 15))
        
        for i, (_, row) in enumerate(samples.iterrows()):
            img_path = row['image_path']
            true_label = row['true_label']
            pred_label = row['predicted_label']
            confidence = row['confidence']
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(3, 4, i+1)
            plt.imshow(img)
            
            # Add color based on correct/incorrect prediction
            if true_label != "Unknown":
                color = 'green' if true_label == pred_label else 'red'
                title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
            else:
                color = 'black'
                title = f"Pred: {pred_label}\nConf: {confidence:.2f}"
                
            plt.title(title, color=color)
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(args.output_dir, 'test_predictions_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to {viz_path}")
        
    except Exception as e:
        print(f"Error visualizing predictions: {e}")

def main():
    """Main function to run the testing pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    # Create necessary directories
    create_directories(args)
    
    # Test on directory
    test_on_directory(args)
    
    print("\nTesting complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()