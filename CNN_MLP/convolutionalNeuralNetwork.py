import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Create random variations of training images for data augmentation
def data_augmentation():
    
    datagen = ImageDataGenerator(
        rotation_range=15,           # Randomly rotate images by up to 15 degrees
        width_shift_range=0.1,       # Randomly shift images horizontally by up to 10%
        height_shift_range=0.1,      # Randomly shift images vertically by up to 10%
        horizontal_flip=True,        # Randomly flip images horizontally (left-right)
        zoom_range=0.1,              # Randomly zoom in/out by up to 10%
        fill_mode='nearest'          # Fill empty pixels with nearest pixel values
    )
    
    return datagen


# Custom loss function: Sparse Categorical Crossentropy with Label Smoothing
def sparse_cce_label_smoothing(y_true, y_pred):
    
    num_classes = 10           # CIFAR-10 classes
    label_smoothing = 0.1      # Smoothing factor

    # Convert sparse labels to one-hot
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)

    # Categorical crossentropy with label smoothing
    return tf.keras.losses.categorical_crossentropy(
        y_true_oh, y_pred, label_smoothing=label_smoothing)


# Plot learning curves for CNN
def cnn_plots(history):

    # Plot training & validation accuracy values
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot training & validation loss values
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


# Generate classification report and confusion matrix for CNN
def cnn_results(model, x_test, y_test, class_names):

    # Get predictions
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Handle both one-hot and sparse labels
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.flatten()
    
    # Classification Report
    print(f"\n\n=============== Classification Report (CNN) ===============:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix Plot
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap=plt.cm.Blues, 
        normalize='true', values_format='.2f')
    cm_display.ax_.set_title(f'Confusion Matrix (CNN)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    

# Plot per-class accuracy for CNN
def cnn_per_class_accuracy(model, x_test, y_test, class_names):
    
    # Get predictions
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Handle both one-hot and sparse labels
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.flatten()
    
    # Calculate per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Plot
    plt.figure(figsize=(6, 4))
    bars =plt.bar(class_names, per_class_accuracy, color='steelblue', alpha=0.7)
    plt.bar_label(bars, fmt='%.2f', padding=3)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.title('Per-Class Accuracy (CNN)')
    plt.ylim([0, 1])
    plt.axhline(y=per_class_accuracy.mean(), color='red', linestyle='--', 
                label=f'Mean: {per_class_accuracy.mean():.2f}')
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
