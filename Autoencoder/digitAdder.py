import numpy as np
import matplotlib.pyplot as plt


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
    print("=" * 80)

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

                print(f"{digit1} + {digit2} = {sum} ({tens_digit},{ones_digit}): {n_pairs} pairs")

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
    print("=" * 80)

    return inputs, outputs, sum_labels


# Plot examples of digit addition
def plot_addition_examples(input_img, target_img, predicted_img, n=5):

    fig, axes = plt.subplots(n, 3, figsize=(9, 2*n))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(n):
        # Input (A + B)
        axes[i, 0].imshow(input_img[i].reshape(28, 56), cmap='gray')
        axes[i, 0].set_title('Input: A + B', fontsize=12)
        axes[i, 0].axis('off')
        
        # Target output
        axes[i, 1].imshow(target_img[i].reshape(28, 56), cmap='gray')
        axes[i, 1].set_title('Target Sum', fontsize=12)
        axes[i, 1].axis('off')
        
        # Predicted output
        axes[i, 2].imshow(predicted_img[i].reshape(28, 56), cmap='gray')
        axes[i, 2].set_title('Predicted Sum', fontsize=12)
        axes[i, 2].axis('off')
    
    plt.tight_layout(pad=0.5)
    plt.show()