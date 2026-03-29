"""
CS 429/529 Midterm Project - Tasks 1 & 2
Task 1: Download and import MNIST and Fashion-MNIST datasets
Task 2: Flatten images from 28x28 to 1-D arrays (784)
"""

import os
import gzip
import urllib.request
import numpy as np
import idx2numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Task 1: Download and Import Datasets
# =============================================================================

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- MNIST URLs ---
MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

# --- Fashion-MNIST URLs ---
FASHION_BASE = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def download_and_extract(base_url, files, dataset_name):
    """Download .gz files, decompress, and load with idx2numpy."""
    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    loaded = {}
    for key, filename in files.items():
        gz_path = os.path.join(dataset_dir, filename)
        idx_path = gz_path.replace(".gz", "")

        # Download if not already present
        if not os.path.exists(idx_path):
            if not os.path.exists(gz_path):
                url = base_url + filename
                print(f"  Downloading {url} ...")
                urllib.request.urlretrieve(url, gz_path)

            # Decompress .gz to raw IDX
            print(f"  Extracting {filename} ...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(idx_path, 'wb') as f_out:
                    f_out.write(f_in.read())

        # Load with idx2numpy
        loaded[key] = idx2numpy.convert_from_file(idx_path)
        print(f"  Loaded {key}: shape = {loaded[key].shape}, dtype = {loaded[key].dtype}")

    return loaded


print("=" * 60)
print("Task 1: Downloading and Importing Datasets")
print("=" * 60)

print("\n--- MNIST ---")
mnist = download_and_extract(MNIST_BASE, MNIST_FILES, "mnist")
mnist_train_images = mnist["train_images"]
mnist_train_labels = mnist["train_labels"]
mnist_test_images  = mnist["test_images"]
mnist_test_labels  = mnist["test_labels"]

print("\n--- Fashion-MNIST ---")
fashion = download_and_extract(FASHION_BASE, FASHION_FILES, "fashion_mnist")
fashion_train_images = fashion["train_images"]
fashion_train_labels = fashion["train_labels"]
fashion_test_images  = fashion["test_images"]
fashion_test_labels  = fashion["test_labels"]

# Sanity check: plot sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Sample Images (Top: MNIST, Bottom: Fashion-MNIST)", fontsize=14)
for i in range(5):
    axes[0, i].imshow(mnist_train_images[i], cmap='gray')
    axes[0, i].set_title(f"Label: {mnist_train_labels[i]}")
    axes[0, i].axis('off')

    axes[1, i].imshow(fashion_train_images[i], cmap='gray')
    axes[1, i].set_title(f"Label: {fashion_train_labels[i]}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("sample_images.png", dpi=150)
print("\nSample images saved to sample_images.png")

# =============================================================================
# Task 2: Flatten Images to 1-D Arrays
# =============================================================================

print("\n" + "=" * 60)
print("Task 2: Flattening Images (28x28 -> 784)")
print("=" * 60)

# MNIST
mnist_train_flat = mnist_train_images.reshape(mnist_train_images.shape[0], -1)
mnist_test_flat  = mnist_test_images.reshape(mnist_test_images.shape[0], -1)

print(f"\nMNIST train:   {mnist_train_images.shape} -> {mnist_train_flat.shape}")
print(f"MNIST test:    {mnist_test_images.shape}  -> {mnist_test_flat.shape}")

# Fashion-MNIST
fashion_train_flat = fashion_train_images.reshape(fashion_train_images.shape[0], -1)
fashion_test_flat  = fashion_test_images.reshape(fashion_test_images.shape[0], -1)

print(f"Fashion train: {fashion_train_images.shape} -> {fashion_train_flat.shape}")
print(f"Fashion test:  {fashion_test_images.shape}  -> {fashion_test_flat.shape}")

# Save the flattened arrays for use in later tasks
np.savez_compressed(
    os.path.join(DATA_DIR, "processed_data.npz"),
    mnist_train_flat=mnist_train_flat,
    mnist_train_labels=mnist_train_labels,
    mnist_test_flat=mnist_test_flat,
    mnist_test_labels=mnist_test_labels,
    fashion_train_flat=fashion_train_flat,
    fashion_train_labels=fashion_train_labels,
    fashion_test_flat=fashion_test_flat,
    fashion_test_labels=fashion_test_labels,
)
print(f"\nAll flattened data saved to {DATA_DIR}/processed_data.npz")
print("\nTasks 1 & 2 complete!")