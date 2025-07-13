import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_and_resize_image(image_path, target_size=(224, 224)):
    """
    Load an image from path and resize it to target size.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size (height, width)
        
    Returns:
        PIL.Image: Resized image
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert grayscale to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img = img.resize(target_size)
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image_for_prediction(image, target_size=(224, 224)):
    """
    Preprocess an image for prediction.
    
    Args:
        image: PIL Image or numpy array
        target_size (tuple): Target size (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image ready for model input
    """
    if isinstance(image, np.ndarray):
        # If image is already a numpy array (from webcam)
        img_array = cv2.resize(image, target_size)
        # Convert BGR to RGB if image is from OpenCV
        if img_array.shape[-1] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        # If image is a PIL Image (from file upload)
        img = image.resize(target_size)
        img_array = np.array(img)
    
    # Normalize the image
    img_array = img_array.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def create_data_generators(dataset_path, image_size=224, batch_size=32, validation_split=0.2):
    """
    Create train and validation data generators.
    
    Args:
        dataset_path (str): Path to the dataset directory
        image_size (int): Target image size
        batch_size (int): Batch size
        validation_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
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
        validation_split=validation_split
    )
    
    # Only rescaling for validation set
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator
    validation_generator = validation_datagen.flow_from_directory(
        dataset_path,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def apply_data_augmentation(image, seed=None):
    """
    Apply data augmentation to a single image.
    
    Args:
        image: PIL Image or numpy array
        seed (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Augmented image
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to numpy array if PIL Image
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Create data generator with augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Reshape for augmentation and normalize
    img = image.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Get augmented image
    augmented_img = next(datagen.flow(img, batch_size=1))[0]
    
    # Convert back to 0-255 range
    augmented_img = (augmented_img * 255).astype(np.uint8)
    
    return augmented_img

def prepare_image_batch(image_paths, target_size=(224, 224)):
    """
    Prepare a batch of images for prediction.
    
    Args:
        image_paths (list): List of image paths
        target_size (tuple): Target size (height, width)
        
    Returns:
        numpy.ndarray: Batch of preprocessed images
    """
    batch = []
    valid_paths = []
    
    for img_path in image_paths:
        try:
            img = load_and_resize_image(img_path, target_size)
            if img is not None:
                img_array = np.array(img)
                img_array = img_array.astype(np.float32) / 255.0
                batch.append(img_array)
                valid_paths.append(img_path)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    if not batch:
        return None, []
    
    return np.array(batch), valid_paths

def merge_datasets(dataset_paths, output_path):
    """
    Merge multiple datasets into one.
    
    Args:
        dataset_paths (list): List of dataset directory paths
        output_path (str): Output directory path
        
    Returns:
        dict: Class statistics
    """
    import shutil
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    class_stats = {}
    
    # Process each dataset
    for dataset_path in dataset_paths:
        # Iterate through classes in the dataset
        for class_name in os.listdir(dataset_path):
            class_dir = os.path.join(dataset_path, class_name)
            
            # Skip if not a directory
            if not os.path.isdir(class_dir):
                continue
            
            # Create class directory in output path if it doesn't exist
            output_class_dir = os.path.join(output_path, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Count existing files in output directory
            existing_files = len(os.listdir(output_class_dir)) if os.path.exists(output_class_dir) else 0
            
            # Copy image files from class directory to output directory
            file_count = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_path = os.path.join(class_dir, filename)
                    dst_path = os.path.join(output_class_dir, f"{class_name}_{existing_files + file_count + 1}.jpg")
                    shutil.copy2(src_path, dst_path)
                    file_count += 1
            
            # Update class statistics
            if class_name in class_stats:
                class_stats[class_name] += file_count
            else:
                class_stats[class_name] = file_count
            
            print(f"Copied {file_count} images for class {class_name}")
    
    return class_stats

def split_dataset(input_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        input_path (str): Input dataset directory path
        output_path (str): Output directory path
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Split statistics
    """
    import shutil
    
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    # Create output directories
    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'val')
    test_dir = os.path.join(output_path, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    np.random.seed(random_seed)
    
    split_stats = {'train': {}, 'val': {}, 'test': {}}
    
    # Process each class in the dataset
    for class_name in os.listdir(input_path):
        class_dir = os.path.join(input_path, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Create class directories in train, val, and test directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        np.random.shuffle(image_files)
        
        # Calculate split indices
        n_train = int(len(image_files) * train_ratio)
        n_val = int(len(image_files) * val_ratio)
        
        # Split the image files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # Copy files to respective directories
        split_stats['train'][class_name] = len(train_files)
        split_stats['val'][class_name] = len(val_files)
        split_stats['test'][class_name] = len(test_files)
        
        for f in train_files:
            shutil.copy2(os.path.join(class_dir, f), os.path.join(train_dir, class_name, f))
        
        for f in val_files:
            shutil.copy2(os.path.join(class_dir, f), os.path.join(val_dir, class_name, f))
        
        for f in test_files:
            shutil.copy2(os.path.join(class_dir, f), os.path.join(test_dir, class_name, f))
        
        print(f"Class {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    return split_stats

def validate_image_folder(folder_path):
    """
    Validate image folder by checking if all files are valid images.
    
    Args:
        folder_path (str): Path to the folder containing images
        
    Returns:
        tuple: (valid_count, invalid_count, invalid_files)
    """
    valid_count = 0
    invalid_count = 0
    invalid_files = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Verify the file is a valid image
                        valid_count += 1
                except Exception as e:
                    invalid_count += 1
                    invalid_files.append((file_path, str(e)))
    
    return valid_count, invalid_count, invalid_files

def detect_image_anomalies(folder_path, target_size=(224, 224)):
    """
    Detect anomalies in images like extremely small/large sizes, unusual aspect ratios, etc.
    
    Args:
        folder_path (str): Path to the folder containing images
        target_size (tuple): Target size for reference
        
    Returns:
        dict: Dictionary of anomalies by type
    """
    anomalies = {
        'small_images': [],
        'large_images': [],
        'unusual_aspect_ratio': [],
        'grayscale': [],
        'invalid_images': []
    }
    
    # Define thresholds
    min_dimension = 32  # Images smaller than this are considered small
    max_dimension = 4000  # Images larger than this are considered large
    aspect_ratio_threshold = 3.0  # Images with aspect ratio greater than this are considered unusual
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        # Check dimensions
                        if min(width, height) < min_dimension:
                            anomalies['small_images'].append((file_path, (width, height)))
                        
                        if max(width, height) > max_dimension:
                            anomalies['large_images'].append((file_path, (width, height)))
                        
                        # Check aspect ratio
                        aspect_ratio = max(width, height) / min(width, height)
                        if aspect_ratio > aspect_ratio_threshold:
                            anomalies['unusual_aspect_ratio'].append((file_path, aspect_ratio))
                        
                        # Check if grayscale
                        if img.mode not in ('RGB', 'RGBA'):
                            anomalies['grayscale'].append(file_path)
                
                except Exception as e:
                    anomalies['invalid_images'].append((file_path, str(e)))
    
    return anomalies

def analyze_dataset(dataset_path):
    """
    Analyze a dataset and return statistics.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        dict: Dataset statistics
    """
    stats = {
        'class_counts': {},
        'total_images': 0,
        'classes': [],
        'class_balance': {},
        'image_dimensions': {},
        'file_formats': {}
    }
    
    # Process each class in the dataset
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Count images in this class
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        stats['class_counts'][class_name] = len(images)
        stats['total_images'] += len(images)
        stats['classes'].append(class_name)
        
        # Sample some images to get dimensions
        sample_size = min(len(images), 10)
        sampled_images = np.random.choice(images, sample_size, replace=False) if sample_size > 0 else []
        
        for img_file in sampled_images:
            file_ext = os.path.splitext(img_file)[1].lower()
            
            # Update file format count
            if file_ext in stats['file_formats']:
                stats['file_formats'][file_ext] += 1
            else:
                stats['file_formats'][file_ext] = 1
            
            # Get image dimensions
            try:
                img_path = os.path.join(class_dir, img_file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    dims = f"{width}x{height}"
                    
                    if dims in stats['image_dimensions']:
                        stats['image_dimensions'][dims] += 1
                    else:
                        stats['image_dimensions'][dims] = 1
            except Exception:
                pass
    
    # Calculate class balance
    if stats['total_images'] > 0:
        for class_name, count in stats['class_counts'].items():
            stats['class_balance'][class_name] = count / stats['total_images']
    
    return stats

def normalize_histogram(image):
    """
    Normalize image histogram to improve contrast.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        numpy.ndarray: Normalized image
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Convert to YUV color space if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        
        # Equalize the Y channel (luminance)
        image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
        
        # Convert back to RGB
        result = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    else:
        # For grayscale images
        result = cv2.equalizeHist(image)
    
    return result