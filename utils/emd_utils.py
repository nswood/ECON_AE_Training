import numpy as np
import ot

# Define the active positions in the 8x8 grid (Encoding A)
active_coords = [(i, j) for i in range(8) for j in range(8) if not (i >= 4 and j >= 4)]
active_indices = [i * 8 + j for i, j in active_coords]  # Flattened indices

# Precompute trigger cell positions for Euclidean distance
trigger_cell_positions = np.array([[i, j] for i, j in active_coords])

# Compute the ground distance matrix for EMD (Euclidean)
distance_matrix = np.linalg.norm(
    trigger_cell_positions[:, np.newaxis, :] - trigger_cell_positions[np.newaxis, :, :],
    axis=2
)  # Shape: (48, 48)

def extract_active_cells(image_8x8):
    """Extract the 48 active trigger cell values from an 8x8 image."""
    flat = image_8x8.flatten()
    return flat[active_indices]

def normalize_distribution(vec):
    """Normalize a 1D vector to sum to 1."""
    total = np.sum(vec)
    return vec / total if total > 0 else vec

def compute_emd(input_image, output_image):
    """
    Compute Earth Mover's Distance (EMD) between two 8x8 HGCAL images.

    Args:
        input_image (np.ndarray): Original 8x8 input image
        output_image (np.ndarray): Reconstructed 8x8 output image

    Returns:
        float: EMD value between input and output. Returns 0 if both empty,
               returns 1.0 if one is empty to signal max mismatch.
    """
    x = extract_active_cells(input_image)
    y = extract_active_cells(output_image)

    if np.sum(x) == 0 and np.sum(y) == 0:
        return 0.0
    elif np.sum(x) == 0 or np.sum(y) == 0:
        return 1.0

    x = normalize_distribution(x)
    y = normalize_distribution(y)

    return ot.emd2(x, y, distance_matrix)

def compute_emd_batch(input_images, output_images):
    """
    Compute EMD values for a batch of images.
    
    Args:
        input_images (np.ndarray): Batch of input images, shape (N, 8, 8)
        output_images (np.ndarray): Batch of output images, shape (N, 8, 8)

    Returns:
        list of float: EMD scores for each image pair
    """
    emd_scores = []
    for x_img, y_img in zip(input_images, output_images):
        emd_scores.append(compute_emd(x_img, y_img))
    return emd_scores

