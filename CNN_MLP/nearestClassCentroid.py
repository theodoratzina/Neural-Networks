import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# Train and evaluate Nearest Class Centroid classifier
def train_ncc(x_train, y_train, x_test, y_test, metric='euclidean'):
    
    print(f"Nearest Class Centroid (NCC) Classifier for {metric} distance")
    print("="*60)

    # Create and train the classifier
    ncc = NearestCentroid(metric=metric)
    
    start_time = time.time()
    ncc.fit(x_train, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    start_time = time.time()
    y_pred_train = ncc.predict(x_train)
    y_pred_test = ncc.predict(x_test)
    test_time = time.time() - start_time
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train) * 100
    test_accuracy = accuracy_score(y_test, y_pred_test) * 100
    
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Testing time: {test_time:.2f} seconds")
    print(f"Training accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print("="*60)
    
    return ncc, y_pred_test, test_accuracy


# Generate detailed classification report and plots for NCC
def ncc_results(y_test, y_pred, class_names, metric='euclidean'):

    # Classification Report
    print(f"\n\n====== Classification Report (NCC for metric={metric}) ======:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix Plot
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, cmap=plt.cm.Blues, normalize='true', values_format='.2f')
    cm_display.ax_.set_title(f'Confusion Matrix (NCC for metric={metric})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()