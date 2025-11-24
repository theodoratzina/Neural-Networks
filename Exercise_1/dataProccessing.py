import tensorflow as tf
from sklearn.decomposition import PCA


# Load and normalize CIFAR-10 dataset
def load_cifar10():

    print("Loading and Normalizing CIFAR-10")
    print("="*60)

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print("="*60)

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return x_train, y_train, x_test, y_test


# Flatten CIFAR-10 images
def flatten_cifar10(x_train, x_test):

    print("Flattening CIFAR-10 images")
    print("="*60)

    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    print(f"Flattened training data shape: {x_train.shape}")
    print(f"Flattened test data shape: {x_test.shape}")
    print("="*60)

    return x_train, x_test


# Get CIFAR-10 class Names
def get_class_names():

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("CIFAR-10 Classes")
    print("="*60)
    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")
    print("="*60 + "\n")

    return class_names


# Apply PCA for dimensionality reduction
def apply_pca(x_train, x_test, n_components):

    print(f"Applying PCA: {x_train.shape[1]} → {n_components} dimensions")

    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # Calculate explained variance percentage
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance: {explained_var:.2f}%")
    print(f"New training shape: {x_train_pca.shape}")
    print(f"New test shape: {x_test_pca.shape}")

    return x_train_pca, x_test_pca, pca
