import numpy as np
import matplotlib.pyplot as plt


def analyze_motion_energy(motion_energy):
    """
    Performs quantitative analysis on the raw motion energy array.

    This function computes basic statistics, plots a histogram of pixel values,
    thresholds the array to identify high-motion regions, and calculates the
    number and percentage of pixels with high motion. It also computes the centroid
    of the high-motion areas if any exist.

    :param motion_energy: A 2D NumPy array of accumulated motion energy.
    :return: A dictionary containing computed metrics.
    """
    # Compute basic statistics.
    mean_val = np.mean(motion_energy)
    median_val = np.median(motion_energy)
    std_val = np.std(motion_energy)
    min_val = np.min(motion_energy)
    max_val = np.max(motion_energy)

    print("Motion Energy Statistics:")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Std: {std_val:.2f}")
    print(f"Min: {min_val:.2f}")
    print(f"Max: {max_val:.2f}")

    # Plot a histogram of the motion energy values.
    plt.figure(figsize=(8, 6))
    plt.hist(motion_energy.ravel(), bins=100, color='blue', alpha=0.7)
    plt.title("Histogram of Motion Energy Values")
    plt.xlabel("Motion Energy Value")
    plt.ylabel("Frequency")
    plt.show()

    # Thresholding: as an example, we use mean + std as the threshold.
    threshold = mean_val + std_val
    print(f"Using threshold: {threshold:.2f}")
    binary_mask = motion_energy > threshold
    high_motion_pixels = np.sum(binary_mask)
    total_pixels = motion_energy.size
    high_motion_percentage = (high_motion_pixels / total_pixels) * 100

    print(f"Number of high-motion pixels: {high_motion_pixels}")
    print(f"Percentage of frame with high motion: {high_motion_percentage:.2f}%")

    # Display the binary mask of high-motion regions.
    plt.figure(figsize=(8, 6))
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Binary Mask of High Motion Energy")
    plt.axis('off')
    plt.show()

    # Calculate the centroid of high-motion regions.
    indices = np.argwhere(binary_mask)
    if indices.size > 0:
        centroid = indices.mean(axis=0)
        print(f"Centroid of high motion region (row, col): {centroid}")
    else:
        centroid = None
        print("No high motion regions found.")

    # Return computed metrics in a dictionary.
    metrics = {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "high_motion_pixel_count": int(high_motion_pixels),
        "high_motion_percentage": high_motion_percentage,
        "centroid": centroid
    }
    return metrics


if __name__ == "__main__":
    # Specify the path to your saved raw motion energy array (.npy file).
    # This file is generated by your earlier motion energy computation code.
    raw_file = "/Users/justinjames/LUPE_Corder-Lab/behavior_snippets/Grooming/motion_energy_with_pose.npy"  # Update with your actual path.

    # Load the raw motion energy array.
    motion_energy = np.load(raw_file)
    print("Loaded motion energy array with shape:", motion_energy.shape)

    # Perform quantitative analysis.
    metrics = analyze_motion_energy(motion_energy)

    print("Computed Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")