import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageStat
from collections import Counter

# Define dataset path
dataset_path = r"C:\Users\KASHARA ALVIN SSALI\Desktop\FakeCurrencyDetectionSystem\Dataset"
folders = ["Training", "Validation", "Testing"]
categories = ["Fake", "Real"]

# Function to iterate over dataset
def iterate_dataset(func):
    results = []
    for folder in folders:
        for category in categories:
            category_path = os.path.join(dataset_path, folder, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                results.append(func(img_path))
    return results

# Count images per category
image_counts = {f: {c: len(os.listdir(os.path.join(dataset_path, f, c))) for c in categories} for f in folders}
print("Image Count Per Category:", image_counts)

# Plot class distribution
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=[f"{f}-{c}" for f in image_counts for c in image_counts[f]], 
            y=[image_counts[f][c] for f in image_counts for c in image_counts[f]], 
            palette="viridis")
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
        for j, img_name in enumerate(os.listdir(category_path)[:5]):
            axes[i, j].imshow(Image.open(os.path.join(category_path, img_name)))
            axes[i, j].axis("off")
            axes[i, j].set_title(category.capitalize())
    plt.show()

show_sample_images()

# Get image sizes
sizes_counter = Counter(iterate_dataset(lambda p: Image.open(p).size))
print("Most common image sizes:", sizes_counter.most_common(5))

# Image brightness analysis
brightness_values = iterate_dataset(lambda p: ImageStat.Stat(Image.open(p).convert("L")).mean[0])
sns.histplot(brightness_values, bins=20, kde=True, color='purple')
plt.xlabel("Brightness Level")
plt.ylabel("Frequency")
plt.title("Image Brightness Distribution", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Pixel intensity histogram
sample_img = os.path.join(dataset_path, "Training", "Real", os.listdir(os.path.join(dataset_path, "Training", "Real"))[0])
img = cv2.imread(sample_img, cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(10, 6))
plt.hist(img.ravel(), bins=256, color='gray', alpha=0.7)
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.title("Pixel Intensity Histogram", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Detect corrupted images
corrupted_images = [p for p in iterate_dataset(lambda p: p if not Image.open(p).verify() else None) if p]
print(f"Found {len(corrupted_images)} corrupted images.")

# Noise analysis
noise_levels = iterate_dataset(lambda p: np.std(cv2.imread(p, cv2.IMREAD_GRAYSCALE)))
sns.histplot(noise_levels, bins=20, kde=True, color='brown')
plt.xlabel("Noise Level (Standard Deviation)")
plt.ylabel("Frequency")
plt.title("Image Noise Distribution", fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print(f"Found {len(corrupted_images)} corrupted images.")
