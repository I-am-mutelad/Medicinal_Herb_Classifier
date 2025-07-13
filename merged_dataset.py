import cv2
import os

train_dir = 'D:/Projects/Python/medicinal-herbs-detector/dataset/images/train'
valid_images = []

for filename in os.listdir(train_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            valid_images.append(filename)
        else:
            print(f"Corrupted or unreadable: {img_path}")
            os.remove(img_path)

print(f"Valid images remaining: {len(valid_images)}")
