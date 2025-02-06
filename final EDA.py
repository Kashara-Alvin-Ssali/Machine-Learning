import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageStat
from collections import Counter

# Define dataset path
dataset_path = r"C:\Users\KASHARA ALVIN SSALI\Desktop\Work\FakeCurrencyDetectionSystem\Dataset"
folders = ["Training", "Validation", "Testing"]
categories = ["Fake", "Real"]

# Function to iterate over dataset efficiently
def iterate_dataset(func):
    """Iterate over dataset and apply function to each image."""
    for folder in folders:
        for category in categories:
            category_path = os.path.join(dataset_path, folder, category)
            for img_name in filter(lambda f: f.lower().endswith(('.png', '.jpg', '.jpeg')), os.listdir(category_path)):
                img_path = os.path.join(category_path, img_name)
                yield func(img_path)

# Count images per category
image_counts = {
    f: {c: len(list(filter(lambda img: img.lower().endswith(('.png', '.jpg', '.jpeg')), os.listdir(os.path.join(dataset_path, f, c)))))
        for c in categories}
    for f in folders
}
print("Image Count Per Category:", image_counts)

# Plot class distribution
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
sns.barplot(
    x=[f"{f}-{c}" for f in image_counts for c in image_counts[f]],
    y=[image_counts[f][c] for f in image_counts for c in image_counts[f]],
    palette="viridis"
)
plt.xticks(rotation=45)
plt.ylabel("Number of Images")
plt.xlabel("Dataset Categories")
plt.title("Class Distribution in Dataset", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Show sample images
def show_sample_images():
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, category in enumerate(categories):
        category_path = os.path.join(dataset_path, "Training", category)
        sample_images = list(filter(lambda f: f.lower().endswith(('.png', '.jpg', '.jpeg')), os.listdir(category_path)))[:5]
        for j, img_name in enumerate(sample_images):
            axes[i, j].imshow(Image.open(os.path.join(category_path, img_name)))
            axes[i, j].axis("off")
            axes[i, j].set_title(category.capitalize())
    plt.show()

show_sample_images()

# Get image sizes
sizes_counter = Counter(iterate_dataset(lambda p: Image.open(p).size))
print("Most common image sizes:", sizes_counter.most_common(5))

# Image brightness analysis
brightness_values = list(iterate_dataset(lambda p: ImageStat.Stat(Image.open(p).convert("L")).mean[0]))
sns.histplot(brightness_values, bins=20, kde=True, color='purple')
plt.xlabel("Brightness Level")
plt.ylabel("Frequency")
plt.title("Image Brightness Distribution", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Pixel intensity histogram
sample_img_path = os.path.join(dataset_path, "Training", "Real")
sample_images = list(filter(lambda f: f.lower().endswith(('.png', '.jpg', '.jpeg')), os.listdir(sample_img_path)))
if sample_images:
    sample_img = os.path.join(sample_img_path, sample_images[0])
    img = cv2.imread(sample_img, cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(10, 6))
    plt.hist(img.ravel(), bins=256, color='gray', alpha=0.7)
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    plt.title("Pixel Intensity Histogram", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Detect corrupted images
def check_corruption(img_path):
    try:
        with Image.open(img_path) as img:
            img.load()
        return None  # Not corrupted
    except Exception:
        return img_path  # Corrupted

corrupted_images = [p for p in iterate_dataset(check_corruption) if p]
print(f"Found {len(corrupted_images)} corrupted images.")

# Noise analysis
noise_levels = list(iterate_dataset(lambda p: np.std(cv2.imread(p, cv2.IMREAD_GRAYSCALE))))
sns.histplot(noise_levels, bins=20, kde=True, color='brown')
plt.xlabel("Noise Level (Standard Deviation)")
plt.ylabel("Frequency")
plt.title("Image Noise Distribution", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print(f"Found {len(corrupted_images)} corrupted images.")
