import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Create dataset for next digit prediction
def next_digit_dataset(x_train, y_train, x_test, y_test):
    
    # Input: digit N, Output: digit (N+1) % 10
    print("Creating Next Digit Dataset")
    print("="*80)

    # Initialize lists to hold input-output pairs
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []

    # Create training pairs
    print("Creating training pairs:")
    for digit in range(10):
        next_digit = (digit + 1) % 10

        # Get images of current digit
        digit_indices = np.where(y_train == digit)[0]
        # Get images of next digit
        next_digit_indices = np.where(y_train == next_digit)[0]

        # Use minimum count to balance
        min_count = min(len(digit_indices), len(next_digit_indices))

        # Append pairs to training set
        train_inputs.extend(x_train[digit_indices[:min_count]])
        train_outputs.extend(x_train[next_digit_indices[:min_count]])

        print(f"Digit {digit} -> {next_digit}: {min_count} pairs")

    # Create test pairs
    print("\nCreating test pairs:")
    for digit in range(10):
        next_digit = (digit + 1) % 10

        # Get images of current digit
        digit_indices = np.where(y_test == digit)[0]
        # Get images of next digit
        next_digit_indices = np.where(y_test == next_digit)[0]

        # Use minimum count to balance
        min_count = min(len(digit_indices), len(next_digit_indices))

        # Append pairs to test set
        test_inputs.extend(x_test[digit_indices[:min_count]])
        test_outputs.extend(x_test[next_digit_indices[:min_count]])

        print(f"Digit {digit} -> {next_digit}: {min_count} pairs")

    # Convert lists to numpy arrays
    train_inputs = np.array(train_inputs)
    train_outputs = np.array(train_outputs)
    test_inputs = np.array(test_inputs)
    test_outputs = np.array(test_outputs)

    print(f"\nTotal training pairs: {len(train_inputs)}")
    print(f"Total test pairs: {len(test_inputs)}")
    print("="*80 + "\n")

    return train_inputs, train_outputs, test_inputs, test_outputs


# Visualize next digit predictions
def plot_next_digit_predictions(original_img, predicted_img, target_img, n=10):
    
    fig = plt.figure(figsize=(20, 9))

    for i in range(n):
        # Print original images
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original_img[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # Print predicted next digit images
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(predicted_img[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # Print target next digit images
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(target_img[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    fig.text(0.5, 0.93, 'Initial Digits', ha='center', fontsize=18)
    fig.text(0.5, 0.62, 'Predicted Next Digits', ha='center', fontsize=18)
    fig.text(0.5, 0.31, 'Target Next Digits', ha='center', fontsize=18)
    
    # Adjust layout
    plt.subplots_adjust(left=0.02, right=0.98, top=0.91, 
                        bottom=0.09, wspace=0.05, hspace=0.5)
    
    plt.show()


# Get one example of each pair of digits
def get_stratified_next_digit_samples(y_test):

    indices = []
    current_position = 0
    
    # For each starting digit (0→1, 1→2, etc.)
    for digit in range(10):
        next_digit = (digit + 1) % 10
        
        # Count how many pairs exist for this transition
        digit_count = np.sum(y_test == digit)
        next_digit_count = np.sum(y_test == next_digit)
        min_count = min(digit_count, next_digit_count)
        
        # Select one random sample for this transition
        if min_count > 0:
            offset = np.random.randint(0, min_count)
            indices.append(current_position + offset)
            
            # Move to next transition's starting position
            current_position += min_count
    
    return np.array(indices)





def next_digit_quality(model, test_inputs, test_outputs, digit_classifier=None):
    """
    Evaluate next digit predictions
    If digit_classifier is provided, check if predicted digit is correctly classified
    """
    predictions = model.predict(test_inputs)

    # Calculate MSE
    mse = np.mean((test_outputs - predictions) ** 2)
    print(f"Next Digit Prediction MSE: {mse:.6f}")

    # If classifier is provided, check classification accuracy
    if digit_classifier is not None:
        pred_classes = np.argmax(digit_classifier.predict(predictions.reshape(-1, 28, 28, 1)), axis=1)

        # Determine expected next digits
        true_classes = np.argmax(digit_classifier.predict(test_outputs.reshape(-1, 28, 28, 1)), axis=1)

        accuracy = np.mean(pred_classes == true_classes) * 100
        print(f"Classification Accuracy of Predicted Digits: {accuracy:.2f}%")

        return mse, accuracy

    return mse, None
