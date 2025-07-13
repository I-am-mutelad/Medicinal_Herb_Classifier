import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from PIL import Image
import io
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle


def plot_training_history(history, output_path=None, figsize=(12, 5)):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: History object returned from model.fit()
        output_path: Path to save the plot (if None, plot is displayed)
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='#4CAF50')  # Green theme
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#2e7d32')
    plt.title('Model Accuracy', fontsize=12, color='#2e7d32')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='#4CAF50')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#2e7d32')
    plt.title('Model Loss', fontsize=12, color='#2e7d32')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt


def plot_confusion_matrix(y_true, y_pred, class_names, output_path=None, figsize=(10, 10), normalize=False):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot (if None, plot is displayed)
        figsize: Figure size as (width, height)
        normalize: Boolean to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Greens',  # Green-themed colormap
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.ylabel('True Label', fontsize=10, color='#2e7d32')
    plt.xlabel('Predicted Label', fontsize=10, color='#2e7d32')
    plt.title('Confusion Matrix', fontsize=12, color='#2e7d32')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt


def plot_roc_curves(y_true, y_score, class_names, output_path=None, figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels (integer encoded)
        y_score: Predicted probabilities from model.predict()
        class_names: List of class names
        output_path: Path to save the plot (if None, plot is displayed)
        figsize: Figure size as (width, height)
    """
    # Binarize the labels for one-vs-rest ROC calculation
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    plt.figure(figsize=figsize)
    
    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='#4CAF50', linestyle=':', linewidth=4)
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Only plot a subset of classes if there are too many
    max_classes_to_plot = 10
    if n_classes > max_classes_to_plot:
        indices = np.linspace(0, n_classes-1, max_classes_to_plot).astype(int)
    else:
        indices = range(n_classes)
    
    # Plot ROC curves for selected classes
    colors = cycle(['#2e7d32', '#66bb6a', '#a5d6a7', '#c8e6c9', '#81c784'])
    for i, color in zip(indices, colors):
        if len(fpr[i]):  # Check if we have points to plot
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10, color='#2e7d32')
    plt.ylabel('True Positive Rate', fontsize=10, color='#2e7d32')
    plt.title('Multi-class Receiver Operating Characteristic', fontsize=12, color='#2e7d32')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt


def plot_precision_recall_curves(y_true, y_score, class_names, output_path=None, figsize=(10, 8)):
    """
    Plot precision-recall curves for multi-class classification
    
    Args:
        y_true: True labels (integer encoded)
        y_score: Predicted probabilities from model.predict()
        class_names: List of class names
        output_path: Path to save the plot (if None, plot is displayed)
        figsize: Figure size as (width, height)
    """
    # Binarize the labels for one-vs-rest PR calculation
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute precision-recall curve for each class
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        avg_precision[i] = np.mean(precision[i])
    
    # Plot precision-recall curves
    plt.figure(figsize=figsize)
    
    # Only plot a subset of classes if there are too many
    max_classes_to_plot = 10
    if n_classes > max_classes_to_plot:
        indices = np.linspace(0, n_classes-1, max_classes_to_plot).astype(int)
    else:
        indices = range(n_classes)
    
    # Plot precision-recall curves for selected classes
    colors = cycle(['#2e7d32', '#66bb6a', '#a5d6a7', '#c8e6c9', '#81c784'])
    for i, color in zip(indices, colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR curve of {class_names[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=10, color='#2e7d32')
    plt.ylabel('Precision', fontsize=10, color='#2e7d32')
    plt.title('Multi-class Precision-Recall Curves', fontsize=12, color='#2e7d32')
    plt.legend(loc="lower left", prop={'size': 8})
    plt.grid(alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt


def visualize_model_layers(model):
    """
    Visualize the model architecture with layer shapes
    
    Args:
        model: Keras model
    
    Returns:
        String with model architecture information
    """
    output = "Model Architecture:\n"
    output += "="*80 + "\n"
    
    for i, layer in enumerate(model.layers):
        output += f"Layer {i}: {layer.name}\n"
        output += f"  Type: {layer.__class__.__name__}\n"
        output += f"  Input Shape: {layer.input_shape}\n"
        output += f"  Output Shape: {layer.output_shape}\n"
        output += f"  Parameters: {layer.count_params():,}\n"
        output += f"  Trainable: {layer.trainable}\n"
        output += "-"*80 + "\n"
    
    output += "="*80 + "\n"
    output += f"Total Parameters: {model.count_params():,}\n"
    trainable_params = sum([np.prod(layer.trainable_weights[0].shape) if layer.trainable_weights else 0 for layer in model.layers])
    output += f"Trainable Parameters: {trainable_params:,}\n"
    output += f"Non-trainable Parameters: {model.count_params() - trainable_params:,}\n"
    
    return output


def plot_model_architecture(model, output_path="model_architecture.png", show_shapes=True, show_dtype=False):
    """
    Plot model architecture using tf.keras.utils.plot_model
    
    Args:
        model: Keras model
        output_path: Path to save the plot
        show_shapes: Whether to display shapes
        show_dtype: Whether to display dtypes
    
    Returns:
        Path to the saved model architecture image
    """
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=output_path, show_shapes=show_shapes, show_dtype=show_dtype, 
                  show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)
        return output_path
    except ImportError:
        print("Could not import plot_model. Make sure pydot and graphviz are installed.")
        return None
    except Exception as e:
        print(f"Error plotting model: {e}")
        return None


def visualize_activations(model, img_array, layer_names=None, cols=3, figsize=(15, 12)):
    """
    Visualize activations of intermediate layers for a given input image
    
    Args:
        model: Keras model
        img_array: Input image as numpy array (1, height, width, channels)
        layer_names: List of layer names to visualize (if None, all conv layers are used)
        cols: Number of columns in the grid
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure with activations
    """
    if layer_names is None:
        layer_names = [layer.name for layer in model.layers if 'conv' in layer.name.lower()]
    
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    activations = activation_model.predict(img_array)
    
    rows = int(np.ceil(len(layer_names) / cols))
    fig = plt.figure(figsize=figsize)
    
    for i, (layer_name, layer_activation) in enumerate(zip(layer_names, activations)):
        plt.subplot(rows, cols, i + 1)
        
        n_features = min(layer_activation.shape[-1], 64)
        
        if n_features > 1:
            mean_activation = np.mean(layer_activation[0, :, :, :n_features], axis=-1)
        else:
            mean_activation = layer_activation[0, :, :, 0]
        
        plt.imshow(mean_activation, cmap='viridis')
        plt.title(f"{layer_name} (mean)", fontsize=10, color='#2e7d32')
        plt.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_feature_maps(model, img_array, layer_name, figsize=(15, 15), max_features=64):
    """
    Visualize individual feature maps for a specific layer
    
    Args:
        model: Keras model
        img_array: Input image as numpy array (1, height, width, channels)
        layer_name: Name of the layer to visualize
        figsize: Figure size as (width, height)
        max_features: Maximum number of features to display
    
    Returns:
        Matplotlib figure with feature maps
    """
    layer_output = model.get_layer(layer_name).output
    intermediate_model = Model(inputs=model.input, outputs=layer_output)
    
    feature_maps = intermediate_model.predict(img_array)
    
    n_features = min(feature_maps.shape[-1], max_features)
    size = int(np.ceil(np.sqrt(n_features)))
    
    fig = plt.figure(figsize=figsize)
    
    for i in range(n_features):
        plt.subplot(size, size, i + 1)
        feature_map = feature_maps[0, :, :, i]
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f"Feature Maps for Layer: {layer_name}", fontsize=12, color='#2e7d32')
    plt.tight_layout()
    return fig


def apply_gradcam(model, img_array, class_index, last_conv_layer_name=None):
    """
    Apply Grad-CAM to visualize areas of input that are important for classification
    
    Args:
        model: Keras model
        img_array: Input image as numpy array (1, height, width, channels)
        class_index: Index of the class to generate Grad-CAM for
        last_conv_layer_name: Name of the last conv layer (if None, tries to find it)
    
    Returns:
        Tuple containing original image and heatmap
    """
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("Could not find a convolutional layer in the model")
    
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    
    img_height, img_width = img_array.shape[1], img_array.shape[2]
    heatmap = cv2.resize(heatmap, (img_width, img_height))
    
    return img_array[0], heatmap


def plot_gradcam(img_array, class_index, model, last_conv_layer_name=None, output_path=None, 
                alpha=0.4, figsize=(12, 4), class_names=None):
    """
    Plot Grad-CAM visualization
    
    Args:
        img_array: Input image as numpy array (1, height, width, channels)
        class_index: Index of the class to generate Grad-CAM for
        model: Keras model
        last_conv_layer_name: Name of the last conv layer (if None, tries to find it)
        output_path: Path to save the plot (if None, plot is displayed)
        alpha: Transparency of heatmap overlay
        figsize: Figure size as (width, height)
        class_names: List of class names
    
    Returns:
        Matplotlib figure with Grad-CAM visualization
    """
    img, heatmap = apply_gradcam(model, img_array, class_index, last_conv_layer_name)
    
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    
    ax[0].imshow(img)
    ax[0].set_title("Original Image", fontsize=10, color='#2e7d32')
    ax[0].axis('off')
    
    ax[1].imshow(heatmap, cmap='jet')
    ax[1].set_title("Class Activation Map", fontsize=10, color='#2e7d32')
    ax[1].axis('off')
    
    heatmap_colored = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    if img.dtype != np.uint8:
        img_uint8 = np.uint8(img * 255)
    else:
        img_uint8 = img
    
    superimposed_img = cv2.addWeighted(img_uint8, 1-alpha, heatmap_colored, alpha, 0)
    ax[2].imshow(superimposed_img)
    ax[2].set_title("Grad-CAM Overlay", fontsize=10, color='#2e7d32')
    ax[2].axis('off')
    
    class_label = class_names[class_index] if class_names and class_index < len(class_names) else f"Class {class_index}"
    fig.suptitle(f"Grad-CAM for {class_label}", fontsize=14, color='#2e7d32')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig


def visualize_data_distribution(data_dir, output_path=None, figsize=(12, 8)):
    """
    Visualize class distribution in the dataset
    
    Args:
        data_dir: Directory containing class subdirectories
        output_path: Path to save the plot (if None, plot is displayed)
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure with data distribution
    """
    class_counts = {}
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            image_files = [f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(image_files)
    
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    plt.figure(figsize=figsize)
    
    ax = sns.barplot(x='Count', y='Class', data=df, palette='Greens')  # Green-themed palette
    
    for i, count in enumerate(df['Count']):
        ax.text(count + 1, i, str(count), va='center', color='#2e7d32')
    
    plt.title('Class Distribution in Dataset', fontsize=12, color='#2e7d32')
    plt.xlabel('Number of Images', fontsize=10, color='#2e7d32')
    plt.ylabel('Class Name', fontsize=10, color='#2e7d32')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt


def visualize_image_grid(images, labels=None, class_names=None, rows=5, cols=5, figsize=(15, 15)):
    """
    Display a grid of images with optional labels
    
    Args:
        images: List of images (numpy arrays)
        labels: List of labels (integer class indices)
        class_names: List of class names for label mapping
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure with image grid
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    max_images = rows * cols
    images = images[:max_images]
    
    if labels is not None:
        labels = labels[:max_images]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        ax.imshow(img)
        ax.axis('off')
        
        if labels is not None and i < len(labels):
            label_idx = labels[i]
            if class_names and label_idx < len(class_names):
                title = class_names[label_idx]
            else:
                title = f"Class {label_idx}"
            ax.set_title(title, fontsize=8, color='#2e7d32')
    
    for i in range(len(images), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig


def visualize_augmentations(image, datagen, n_augmentations=9, figsize=(12, 12)):
    """
    Visualize data augmentations on a single image
    
    Args:
        image: Input image as numpy array (height, width, channels)
        datagen: ImageDataGenerator object with augmentation parameters
        n_augmentations: Number of augmentations to display
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure with augmentations
    """
    image = image.reshape((1,) + image.shape)
    
    augmented_images = []
    for i, batch in enumerate(datagen.flow(image, batch_size=1)):
        augmented_images.append(batch[0])
        if i >= n_augmentations - 1:
            break
    
    rows = int(np.ceil(np.sqrt(n_augmentations + 1)))
    cols = rows
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    axes[0].imshow(image[0])
    axes[0].set_title("Original", fontsize=10, color='#2e7d32')
    axes[0].axis('off')
    
    for i, img in enumerate(augmented_images):
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"Aug #{i+1}", fontsize=10, color='#2e7d32')
        axes[i+1].axis('off')
    
    for i in range(n_augmentations + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig


def visualize_learning_rate_finder(learning_rates, losses, output_path=None, figsize=(10, 6)):
    """
    Visualize results from a learning rate finder
    
    Args:
        learning_rates: List of learning rates
        losses: List of loss values corresponding to learning rates
        output_path: Path to save the plot (if None, plot is displayed)
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure with learning rate vs loss plot
    """
    plt.figure(figsize=figsize)
    plt.plot(learning_rates, losses, color='#4CAF50')
    plt.xscale('log')
    plt.xlabel('Learning Rate', fontsize=10, color='#2e7d32')
    plt.ylabel('Loss', fontsize=10, color='#2e7d32')
    plt.title('Learning Rate vs. Loss', fontsize=12, color='#2e7d32')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    smooth_losses = np.array(losses)
    gradients = np.gradient(smooth_losses)
    min_grad_idx = np.argmin(gradients)
    suggested_lr = learning_rates[min_grad_idx]
    
    plt.axvline(x=suggested_lr, color='#2e7d32', linestyle='--', alpha=0.5)
    plt.text(suggested_lr, 
             (max(losses) + min(losses)) / 2, 
             f'Suggested LR: {suggested_lr:.2e}', 
             rotation=90, 
             verticalalignment='center', color='#2e7d32')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt


def visualize_class_predictions(img, predictions, class_names, output_path=None, figsize=(12, 4)):
    """
    Visualize top class predictions for an image
    
    Args:
        img: Input image as numpy array
        predictions: Array of class probabilities
        class_names: List of class names
        output_path: Path to save the plot (if None, plot is displayed)
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure with image and prediction bars
    """
    top_k = min(5, len(predictions))
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_values = predictions[top_indices]
    top_classes = [class_names[i] for i in top_indices]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1.5]})
    
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=10, color='#2e7d32')
    
    bars = ax2.barh(range(top_k), top_values, align='center', color='#4CAF50')
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(top_classes)
    ax2.set_xlabel('Probability', fontsize=10, color='#2e7d32')
    ax2.set_title('Top Predictions', fontsize=10, color='#2e7d32')
    
    for i, (value, bar) in enumerate(zip(top_values, bars)):
        ax2.text(min(value + 0.01, 0.99), i, f'{value:.4f}', va='center', color='#2e7d32')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return fig


def visualize_model_size_comparison(model_sizes, model_accuracies, model_names=None, output_path=None, figsize=(10, 6)):
    """
    Visualize trade-off between model size and accuracy
    
    Args:
        model_sizes: List of model sizes in MB
        model_accuracies: List of model accuracies (0-1)
        model_names: Optional list of model names
        output_path: Path to save the plot (if None, plot is displayed)
        figsize: Figure size as (width, height)
    
    Returns:
        Matplotlib figure with model size vs accuracy plot
    """
    # Input validation
    if len(model_sizes) != len(model_accuracies):
        raise ValueError("model_sizes and model_accuracies must have the same length")
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(model_sizes))]
    elif len(model_names) != len(model_sizes):
        raise ValueError("model_names must have the same length as model_sizes")

    # Create scatter plot
    plt.figure(figsize=figsize)
    
    # Scatter plot with green-themed colors
    scatter = plt.scatter(model_sizes, model_accuracies, c='#4CAF50', s=100, alpha=0.6, edgecolors='#2e7d32')
    
    # Annotate each point with model name
    for i, (size, acc, name) in enumerate(zip(model_sizes, model_accuracies, model_names)):
        plt.annotate(name, (size, acc), xytext=(5, 5), textcoords='offset points', fontsize=8, color='#2e7d32')
    
    # Plot trend line (optional, if more than 2 points)
    if len(model_sizes) > 2:
        z = np.polyfit(model_sizes, model_accuracies, 1)
        p = np.poly1d(z)
        plt.plot(model_sizes, p(model_sizes), linestyle='--', color='#2e7d32', alpha=0.5, label='Trend Line')
    
    # Customize plot
    plt.xlabel('Model Size (MB)', fontsize=10, color='#2e7d32')
    plt.ylabel('Accuracy', fontsize=10, color='#2e7d32')
    plt.title('Model Size vs. Accuracy', fontsize=12, color='#2e7d32')
    plt.grid(True, alpha=0.3)
    
    # Add legend if trend line is plotted
    if len(model_sizes) > 2:
        plt.legend()
    
    # Set axis limits with some padding
    plt.margins(0.1)
    plt.xlim(min(model_sizes) * 0.9, max(model_sizes) * 1.1)
    plt.ylim(min(model_accuracies) * 0.9, max(model_accuracies) * 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        return plt