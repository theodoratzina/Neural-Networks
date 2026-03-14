import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
from keras.datasets import mnist
from sklearn.decomposition import PCA 


# Load and normalize MNIST dataset
def load_mnist():

    print("Loading and Normalizing MNIST Dataset")
    print("="*80)

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print("="*80 + "\n")

    # Normalize pixel values to the range [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, y_train, x_test, y_test


# Flatten MNIST images
def flatten_mnist(x_train, x_test):

    print("Flattening MNIST images")
    print("="*80)

    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    print(f"Flattened training data shape: {x_train.shape}")
    print(f"Flattened test data shape: {x_test.shape}")
    print("="*80 + "\n")

    return x_train, x_test


# Get MNIST class Names
def get_class_names():

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    print("MNIST Classes")
    print("="*80)
    for i, class_name in enumerate(class_names):
        print(f"{i}: {class_name}")
    print("="*80 + "\n")

    return class_names


# Apply PCA for reconstruction comparison
def pca_reconstruction(x_train, x_test, n_components):

    # Apply PCA
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    print(f"Applying PCA Reconstruction: {x_train.shape[1]} -> {pca.n_components_} -> {x_train.shape[1]} dimensions")
    print("="*80)

    # Reconstruct to original dimensions
    x_train_reconstructed = pca.inverse_transform(x_train_pca)
    x_test_reconstructed = pca.inverse_transform(x_test_pca)

    # Clip values to [0, 1] range
    x_train_reconstructed = np.clip(x_train_reconstructed, 0, 1)
    x_test_reconstructed = np.clip(x_test_reconstructed, 0, 1)

    # Calculate explained variance percentage
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance: {explained_var:.2f}%")
    print(f"Compressed training shape: {x_train_pca.shape}")
    print(f"Compressed test shape: {x_test_pca.shape}")
    print(f"Reconstructed training shape: {x_train_reconstructed.shape}")
    print(f"Reconstructed test shape: {x_test_reconstructed.shape}")
    print("="*80 + "\n")

    return x_train_reconstructed, x_test_reconstructed, pca.n_components_


# Plot training history
def plot_history(history, model_name="Model"):

    plt.figure(figsize=(6, 4))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Plot original and reconstructed images
def plot_reconstructions(original_img, reconstructed_img, n=10, title="Reconstructed Images"):
    
    fig = plt.figure(figsize=(20, 6))

    for i in range(n):
        # Print original images
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_img[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # Print reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_img[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    fig.text(0.5, 0.90, 'Original Images', ha='center', fontsize=18)
    fig.text(0.5, 0.46, title, ha='center', fontsize=18)
    
    # Adjust layout
    plt.subplots_adjust(left=0.02, right=0.98, top=0.87, 
                        bottom=0.13, wspace=0.05, hspace=0.5)
    
    plt.show()


# Calculate Mean Squared Error
def calculate_mse(original_img, reconstructed_img):

    mse = np.mean((original_img - reconstructed_img) ** 2)
    return mse


# Calculate Peak Signal-to-Noise Ratio
def psnr(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1, 28, 28, 1])
    y_pred = tf.reshape(y_pred, [-1, 28, 28, 1])

    return tf.image.psnr(y_true, y_pred, max_val=1.0)


# Calculate Structural Similarity Index Measure
def ssim(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1, 28, 28, 1])
    y_pred = tf.reshape(y_pred, [-1, 28, 28, 1])

    return tf.image.ssim(y_true, y_pred, max_val=1.0)


# Compare models in a formatted table
def compare_models_table(results):
    
    print("MODEL COMPARISON")
    print("="*100)
    print(f"{'Model':<35} {'MSE':<12} {'Encoding':<12} {'Parameters':<15} {'Training Time':<15}")
    print("-"*100)

    for result in results:
        model_name, mse, encoding_dim, params, train_time = result
        print(f"{model_name:<35} {mse:<12.6f} {encoding_dim:<12} {params:<15,} {train_time:<15.2f}")

    print("="*100 + "\n")


# Get one example of each digit
def get_stratified_samples(y_test, n_per_class=1):

    indices = []

    for digit in range(10):
        # Find all indices of the current digit
        digit_indices = np.where(y_test == digit)[0]

        # Randomly select n_per_class indices
        selected = np.random.choice(digit_indices, size=n_per_class, replace=False)
        indices.extend(selected)

    return np.array(indices)


# Build a CNN digit classifier
def build_digit_classifier():

    # Create the CNN classifier
    cnn_model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    cnn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return cnn_model