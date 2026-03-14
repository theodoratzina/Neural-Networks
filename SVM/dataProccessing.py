import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.feature import hog


# Load CIFAR-10 dataset
def load_cifar10():

    print("Loading CIFAR-10 dataset")
    print("="*70)

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print(f"Original training data shape: {x_train.shape}")
    print(f"Original test data shape: {x_test.shape}")

    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print(f"Flatten training labels shape: {y_train.shape}")
    print(f"Flatten test labels shape: {y_test.shape}")
    print("="*70 + "\n")

    return x_train, y_train, x_test, y_test


# Get CIFAR-10 original class names
def get_class_names():

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("CIFAR-10 Classes")
    print("="*70)
    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")
    print("="*70 + "\n")

    return class_names


# Convert Labels (10 classes -> 2 super-classes)
def convert_labels(y_train, y_test):

    print("Converting CIFAR-10 labels to binary (Animals vs Vehicles)")
    print("="*70)

    # Define positive classes (vehicles)
    positive_classes = {0, 1, 8, 9}      # airplane, automobile, ship, truck

    # Create binary labels
    y_train_binary = [1 if label in positive_classes else 0 for label in y_train]
    y_test_binary = [1 if label in positive_classes else 0 for label in y_test]

    y_train_binary = np.array(y_train_binary)
    y_test_binary  = np.array(y_test_binary)

    print("CIFAR-10 binary super-classes:")
    print("0: Animals (bird, cat, deer, dog, frog, horse)")
    print("1: Vehicles (airplane, automobile, ship, truck)")

    print(f"Sample of converted training labels: {y_train_binary[:10]}")
    print(f"Sample of converted test labels: {y_test_binary[:10]}")
    print("="*70 + "\n")

    return y_train_binary, y_test_binary


# Split CIFAR-10 dataset into training and validation sets
def split_train_validation(x_train, y_train, val_size=0.4):

    print(f"Splitting training set into training ({100 - val_size*100}%) and validation ({val_size*100}%)")
    print("="*70)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=42, stratify=y_train)

    print(f"New training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print("="*70 + "\n")

    return x_train, y_train, x_val, y_val


# Balance CIFAR-10 dataset so both binary classes have the same number of samples
def balance_dataset(x_train, y_train):

    print("Balancing CIFAR-10 dataset by undersampling majority class")
    print("="*70)

    # Separate classes
    class_0 = np.where(y_train == 0)[0]
    class_1 = np.where(y_train == 1)[0]

    print(f"Class 0 samples before balancing: {len(class_0)}")
    print(f"Class 1 samples before balancing: {len(class_1)}")

    # Determine the size of the smaller class
    min_class_size = min(len(class_0), len(class_1))

    # Randomly sample from each class
    np.random.seed(42)
    sampled_class_0 = np.random.choice(class_0, min_class_size, replace=False)
    sampled_class_1 = np.random.choice(class_1, min_class_size, replace=False)

    # Combine and shuffle
    balanced_indices = np.concatenate((sampled_class_0, sampled_class_1))
    np.random.shuffle(balanced_indices)

    # Create balanced dataset
    x_train_balanced = x_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]

    print(f"Balanced dataset size: {x_train_balanced.shape[0]} samples")
    print(f"Class 0 samples: {np.sum(y_train_balanced == 0)}")
    print(f"Class 1 samples: {np.sum(y_train_balanced == 1)}")
    print("="*70 + "\n")

    return x_train_balanced, y_train_balanced


# Flatten and normalize CIFAR-10 images
def flatten_and_normalize(x_train, x_val, x_test):

    print("Flattening and normalizing CIFAR-10 images")
    print("="*70)

    x_train = x_train.reshape((x_train.shape[0], -1))
    x_val = x_val.reshape((x_val.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    print(f"Flattened training data shape: {x_train.shape}")
    print(f"Flattened validation data shape: {x_val.shape}")
    print(f"Flattened test data shape: {x_test.shape}")
    print("="*70 + "\n")

     # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_val = x_val.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, x_val, x_test


# Apply PCA for dimensionality reduction
def apply_pca(x_train, x_val, x_test, n_components):

    print(f"Applying PCA: {x_train.shape[1]} → {n_components} dimensions")
    print("="*70)

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)

    # Calculate explained variance percentage
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance: {explained_var:.2f}%")
    print(f"New training shape: {x_train_pca.shape}")
    print(f"New validation shape: {x_val_pca.shape}")
    print(f"New test shape: {x_test_pca.shape}")
    print("="*70)

    return x_train_pca, x_val_pca, x_test_pca


# HOG Feature Extraction
def extract_hog_features(images):

    print("Extracting HOG features from images")
    print("="*70)

    # Compute HOG features for each image
    hog_features = []
    for img in images:
        feature = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), 
                      orientations=9, block_norm='L2-Hys', visualize=False, channel_axis=-1)
        hog_features.append(feature)

    hog_features = np.array(hog_features)

    print(f"HOG features shape: {hog_features.shape}")
    print("="*70 + "\n")

    return hog_features


# Extract Color Histogram Features
def extract_color_features(images):

    print("Extracting Color histogram features from images")
    print("="*70)

    # Compute color histograms for each image
    color_hist_features = []
    for img in images:
        hist_r, _ = np.histogram(img[:, :, 0], bins=32, range=(0, 256))
        hist_g, _ = np.histogram(img[:, :, 1], bins=32, range=(0, 256))
        hist_b, _ = np.histogram(img[:, :, 2], bins=32, range=(0, 256))
        color_hist = np.concatenate([hist_r, hist_g, hist_b])
        color_hist_features.append(color_hist)

    color_features = np.array(color_hist_features)

    print(f"Color histogram features shape: {color_features.shape}")
    print("="*70 + "\n")

    return color_features