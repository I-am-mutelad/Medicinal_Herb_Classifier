import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import argparse
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a medicinal herb classifier')
    parser.add_argument('--dataset_path', type=str, default="dataset/medicinal_leaf_dataset",
                        help='Path to the dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for training (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine-tune the model after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of epochs for fine-tuning (default: 10)')
    parser.add_argument('--output_dir', type=str, default="models",
                        help='Directory to save model and results')
    return parser.parse_args()

def create_directories(args):
    """Create necessary directories if they don't exist"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

def load_and_preprocess_data(args):
    """Load and preprocess the dataset"""
    print(f"Loading and preprocessing dataset from {args.dataset_path}...")
    
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation/test set
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    try:
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            args.dataset_path,
            target_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        validation_generator = test_datagen.flow_from_directory(
            args.dataset_path,
            target_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Save class indices
        class_indices = train_generator.class_indices
        class_names = list(class_indices.keys())
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Create a readable version of class labels
        readable_labels = {i: name for name, i in class_indices.items()}
        
        # Save class indices and names
        with open('data/class_labels.json', 'w') as f:
            json.dump(class_names, f)
            
        # Save a mapping of indices to class names
        with open('data/class_indices.json', 'w') as f:
            json.dump(readable_labels, f)
        
        # Generate herb information template
        herb_info = {}
        for herb_name in class_names:
            # Create placeholder data that can be manually edited later
            herb_info[herb_name] = {
                "scientific_name": f"Scientific name for {herb_name}",
                "medicinal_uses": [
                    "Medicinal use 1",
                    "Medicinal use 2",
                    "Medicinal use 3"
                ],
                "description": f"Description of {herb_name} and its properties."
            }
        
        # Save herb information template
        with open('data/herb_info.json', 'w') as f:
            json.dump(herb_info, f, indent=4)
            print("Created herb information template at data/herb_info.json")
            print("You can edit this file to add accurate information about each herb.")
        
        return train_generator, validation_generator, len(class_names)
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Please check if the dataset path {args.dataset_path} is correct.")
        exit(1)

def build_model(num_classes, args):
    """Build and compile the model"""
    print("Building the model...")
    
    # Load the MobileNetV2 model without the top layer
    base_model = MobileNetV2(
        input_shape=(args.image_size, args.image_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model summary to a file
    model_summary_path = os.path.join(args.output_dir, "model_summary.txt")
    with open(model_summary_path, 'w') as f:
        # Save original stdout
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        model.summary()
        sys.stdout = original_stdout
    
    print(f"Model architecture saved to {model_summary_path}")
    return model

def train_model(model, train_generator, validation_generator, args):
    """Train the model"""
    print("Training the model...")
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, 'herb_classifier_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Log directory for TensorBoard
    log_dir = os.path.join(args.output_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        write_graph=True
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // args.batch_size + (1 if train_generator.samples % args.batch_size else 0),
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // args.batch_size + (1 if validation_generator.samples % args.batch_size else 0),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )
    
    # Save the final model
    model.save(os.path.join(args.output_dir, 'herb_classifier.h5'))
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        history_dict = {
            'accuracy': [float(acc) for acc in history.history['accuracy']],
            'val_accuracy': [float(acc) for acc in history.history['val_accuracy']],
            'loss': [float(loss) for loss in history.history['loss']],
            'val_loss': [float(loss) for loss in history.history['val_loss']]
        }
        json.dump(history_dict, f)
    
    return history, model

def unfreeze_and_fine_tune(model, train_generator, validation_generator, args):
    """Unfreeze some layers and fine-tune the model"""
    print("Fine-tuning the model...")
    
    # Unfreeze the last few layers of the base model
    base_model = model.layers[0]  # First layer is the MobileNetV2 base
    
    # Unfreeze the last 30 layers of the base model
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(args.output_dir, 'herb_classifier_fine_tuned_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Log directory for TensorBoard
    log_dir = os.path.join(args.output_dir, "logs", f"fine_tune_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // args.batch_size + (1 if train_generator.samples % args.batch_size else 0),
        epochs=args.fine_tune_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // args.batch_size + (1 if validation_generator.samples % args.batch_size else 0),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )
    
    # Save the fine-tuned model
    model.save(os.path.join(args.output_dir, 'herb_classifier_fine_tuned.h5'))
    
    # Update the default model to the fine-tuned version
    model.save(os.path.join(args.output_dir, 'herb_classifier.h5'))
    
    # Save fine-tuning history
    with open(os.path.join(args.output_dir, 'fine_tuning_history.json'), 'w') as f:
        history_dict = {
            'accuracy': [float(acc) for acc in history.history['accuracy']],
            'val_accuracy': [float(acc) for acc in history.history['val_accuracy']],
            'loss': [float(loss) for loss in history.history['loss']],
            'val_loss': [float(loss) for loss in history.history['val_loss']]
        }
        json.dump(history_dict, f)
    
    return history, model

def evaluate_model(model, validation_generator, args):
    """Evaluate the model and generate reports"""
    print("Evaluating the model...")
    
    # Load class labels
    with open('data/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Convert string keys to integers
    class_indices = {int(k): v for k, v in class_indices.items()}
    
    # Reset the generator
    validation_generator.reset()
    
    # Make predictions
    predictions = model.predict(
        validation_generator,
        steps=validation_generator.samples // args.batch_size + (1 if validation_generator.samples % args.batch_size else 0),
        verbose=1
    )
    
    # Get true labels
    true_labels = validation_generator.classes
    
    # Get predicted labels
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Limit to the actual number of validation samples
    true_labels = true_labels[:validation_generator.samples]
    predicted_labels = predicted_labels[:validation_generator.samples]
    
    # Get class names in order
    class_names = [class_indices[i] for i in range(len(class_indices))]
    
    # Generate classification report
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True
    )
    
    # Save classification report
    with open(os.path.join(args.output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save readable classification report
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(true_labels, predicted_labels, target_names=class_names))
    
    # Generate and save confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    
    # Calculate and print overall metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save per-class metrics as a CSV
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(args.output_dir, 'class_metrics.csv'))
    
    return report, cm

def visualize_training_history(history, args, title_suffix=""):
    """Visualize and save training history plots"""
    print(f"Generating training history plots{' for ' + title_suffix if title_suffix else ''}...")
    
    # Accuracy plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save with appropriate filename
    filename = 'training_history'
    if title_suffix:
        filename = f"{filename}_{title_suffix.lower().replace(' ', '_')}"
    plt.savefig(os.path.join(args.output_dir, f'{filename}.png'), dpi=300)
    plt.close()

def main():
    """Main function to run the training pipeline"""
    # Parse command line arguments
    args = parse_args()
    
    # Create necessary directories
    create_directories(args)
    
    # Load and preprocess data
    train_generator, validation_generator, num_classes = load_and_preprocess_data(args)
    print(f"Dataset loaded with {num_classes} classes.")
    
    # Build the model
    model = build_model(num_classes, args)
    
    # Train the model
    history, model = train_model(model, train_generator, validation_generator, args)
    
    # Visualize training history
    visualize_training_history(history, args)
    
    # Evaluate the initial model
    report, cm = evaluate_model(model, validation_generator, args)
    
    # Fine-tune if requested
    if args.fine_tune:
        print("\n\n" + "="*50)
        print("Starting fine-tuning process")
        print("="*50 + "\n")
        
        # Unfreeze and fine-tune
        fine_tune_history, model = unfreeze_and_fine_tune(model, train_generator, validation_generator, args)
        
        # Visualize fine-tuning history
        visualize_training_history(fine_tune_history, args, "Fine-Tuned")
        
        # Evaluate the fine-tuned model
        fine_tuned_report, fine_tuned_cm = evaluate_model(model, validation_generator, args)
        
        print("\nFine-tuning complete! The model has been saved to:")
        print(f"  - {os.path.join(args.output_dir, 'herb_classifier.h5')}")
    
    print("\nTraining and evaluation complete!")
    print(f"Model saved to: {os.path.join(args.output_dir, 'herb_classifier.h5')}")
    print(f"Performance metrics saved to: {args.output_dir}")
    print("\nYou can now run the Streamlit app to use the model for prediction.")

if __name__ == "__main__":
    main()