import numpy as np
from typing import Tuple


# Create a dataset for digit addition using MNIST dataset
def create_digit_adder_dataset(x_data, y_data, seed=42):
    np.random.seed(seed)

    # Group images by digit
    digit_groups = {}
    for digit in range(10):
        digit_groups[digit] = x_data[y_data == digit]

    inputs_list = []
    outputs_list = []
    sum_labels_list = []

    print(f"\nCreating digit addition pairs...")

    # Create all possible pairs
    for digit1 in range(10):
        for digit2 in range(10):
            sum = digit1 + digit2
            tens_digit = sum // 10
            ones_digit = sum % 10

            # Get available images for each digit
            img1 = digit_groups[digit1]
            img2 = digit_groups[digit2]
            tens_img = digit_groups[tens_digit]
            ones_img = digit_groups[ones_digit]

            # Determine how many pairs we can create
            n_pairs = min(len(img1), len(img2), len(tens_img), len(ones_img))
            
            # Limit pairs to avoid memory issues
            n_pairs = min(n_pairs, 100)

            if n_pairs > 0:
                # Select random samples
                idx1 = np.random.choice(len(img1), n_pairs, replace=False)
                idx2 = np.random.choice(len(img2), n_pairs, replace=False)
                idx_tens = np.random.choice(len(tens_img), n_pairs, replace=False)
                idx_ones = np.random.choice(len(ones_img), n_pairs, replace=False)

                # Create input: concatenate two digits side by side
                input_pairs = np.concatenate([img1[idx1], img2[idx2]], axis=2)  # Shape: (n_pairs, 28, 56)

                # Create output: concatenate result digits side by side
                output_pairs = np.concatenate([tens_img[idx_tens], ones_img[idx_ones]], axis=2)  # Shape: (n_pairs, 28, 56)

                inputs_list.append(input_pairs)
                outputs_list.append(output_pairs)
                sum_labels_list.extend([sum] * n_pairs)

                print(f"  {digit1} + {digit2} = {sum} ({tens_digit},{ones_digit}): {n_pairs} pairs")

    # Concatenate all pairs
    inputs = np.concatenate(inputs_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0)
    sum_labels = np.array(sum_labels_list)

    # Shuffle the dataset
    shuffle_idx = np.random.permutation(len(inputs))
    inputs = inputs[shuffle_idx]
    outputs = outputs[shuffle_idx]
    sum_labels = sum_labels[shuffle_idx]

    print(f"\nTotal pairs created: {len(inputs)}")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")

    return inputs, outputs, sum_labels


# Split digit images from concatenated output
def split_digit_images(images):
    
    left_digits = images[:, :, :28]
    right_digits = images[:, :, 28:]
    
    return left_digits, right_digits


def predict_digit_sums(model, test_inputs, digit_classifier=None):
    """
    Predict digit sums using the autoencoder model.
    Optionally use a digit classifier to interpret the output.

    Args:
        model: Trained autoencoder model
        test_inputs: Test input pairs (N, 28, 56, 1) or (N, 3136)
        digit_classifier: Optional CNN classifier to recognize output digits

    Returns:
        predictions: Model output
        recognized_digits: If classifier provided, recognized digit pairs (N, 2)
        predicted_sums: If classifier provided, predicted sum values (N,)
    """
    predictions = model.predict(test_inputs, verbose=0)

    if digit_classifier is not None:
        # Split predictions into left and right digits
        pred_shape = predictions.shape

        if len(pred_shape) == 4:  # Convolutional output (N, 28, 56, 1)
            left_pred, right_pred = split_digit_images(predictions.squeeze())
        else:  # Flat output (N, 3136)
            pred_reshaped = predictions.reshape(-1, 28, 56)
            left_pred, right_pred = split_digit_images(pred_reshaped)

        # Reshape for classifier (N, 28, 28, 1)
        left_pred = left_pred.reshape(-1, 28, 28, 1)
        right_pred = right_pred.reshape(-1, 28, 28, 1)

        # Classify digits
        left_classes = digit_classifier.predict(left_pred, verbose=0).argmax(axis=1)
        right_classes = digit_classifier.predict(right_pred, verbose=0).argmax(axis=1)

        recognized_digits = np.stack([left_classes, right_classes], axis=1)
        predicted_sums = left_classes * 10 + right_classes

        return predictions, recognized_digits, predicted_sums

    return predictions, None, None


def evaluate_addition_accuracy(model, test_inputs, true_sums, digit_classifier):
    """
    Evaluate the accuracy of the digit addition autoencoder.

    Args:
        model: Trained autoencoder
        test_inputs: Test inputs
        true_sums: True sum values
        digit_classifier: Digit recognition model

    Returns:
        accuracy: Percentage of correctly predicted sums
        digit_accuracy: Accuracy per digit position
    """
    _, recognized_digits, predicted_sums = predict_digit_sums(
        model, test_inputs, digit_classifier
    )

    # Calculate sum accuracy
    correct_sums = (predicted_sums == true_sums).sum()
    accuracy = 100 * correct_sums / len(true_sums)

    # Calculate per-digit accuracy
    true_tens = true_sums // 10
    true_ones = true_sums % 10

    tens_correct = (recognized_digits[:, 0] == true_tens).sum()
    ones_correct = (recognized_digits[:, 1] == true_ones).sum()

    tens_accuracy = 100 * tens_correct / len(true_tens)
    ones_accuracy = 100 * ones_correct / len(true_ones)

    return accuracy, (tens_accuracy, ones_accuracy)
