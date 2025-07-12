from __future__ import annotations

import numpy as np


def costes_auto_threshold(
    ch1: np.ndarray,
    ch2: np.ndarray,
    num_thresholds: int = 100,
) -> tuple[float, float, float, float]:
    """
    Implementation of Costes auto-threshold method for colocalization analysis.

    Based on:
    Costes et al. "Automatic and quantitative measurement of protein-protein
    colocalization in live cells" Biophysical Journal 2004
    https://pmc.ncbi.nlm.nih.gov/articles/PMC1304300/

    The method finds thresholds where the Pearson correlation coefficient
    of pixels below the thresholds equals zero, indicating that pixels
    below these thresholds show no statistical correlation.

    This implementation ensures symmetric results regardless of channel order.

    Parameters
    ----------
    ch1: np.ndarray
        First channel image data (2D array).
    ch2: np.ndarray
        Second channel image data (2D array).
    num_thresholds: int
        Number of threshold values to test along the regression line. By default, 100.

    Returns
    -------
    tuple: (threshold_ch1, threshold_ch2, slope, intercept)
        Optimal thresholds for channel 1 and channel 2, slope and intercept of the
        regression line that relates ch2 to ch1 (ch2 = slope * ch1 + intercept).
    """
    # Flatten images for easier processing
    ch1_flat = ch1.ravel()
    ch2_flat = ch2.ravel()

    # If the min value is zero, consider only non-zero pixels
    if np.min(ch1_flat) == 0 or np.min(ch2_flat) == 0:
        mask = (ch1_flat > 0) & (ch2_flat > 0)
        ch1_masked = ch1_flat[mask]
        ch2_masked = ch2_flat[mask]
    else:
        ch1_masked = ch1_flat
        ch2_masked = ch2_flat

    if len(ch1_masked) == 0 or len(ch2_masked) == 0:
        return 0, 0, 0, 0

    # Center the data
    ch1_mean = np.mean(ch1_masked)
    ch2_mean = np.mean(ch2_masked)
    ch1_centered = ch1_masked - ch1_mean
    ch2_centered = ch2_masked - ch2_mean

    # Perform PCA to find the orthogonal regression line
    data = np.vstack([ch1_centered, ch2_centered]).T
    cov_matrix = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # The first principal component is the eigenvector with largest eigenvalue
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]

    # Normalize the principal component
    pc_norm = principal_component / np.linalg.norm(principal_component)

    # Project all points onto the principal component line
    projections = ch1_centered * pc_norm[0] + ch2_centered * pc_norm[1]

    # Generate threshold parameters along the line
    proj_min, proj_max = np.min(projections), np.max(projections)
    t_values = np.linspace(proj_max, proj_min, num_thresholds)

    best_thr_ch1 = best_thr_ch2 = 0
    best_correlation = 1.0

    for t in t_values:
        # Convert parameter t back to (ch1, ch2) coordinates
        thr_ch1 = ch1_mean + t * pc_norm[0]
        thr_ch2 = ch2_mean + t * pc_norm[1]

        # Skip if threshold is outside data range
        if (
            thr_ch1 < np.min(ch1_masked)
            or thr_ch1 > np.max(ch1_masked)
            or thr_ch2 < np.min(ch2_masked)
            or thr_ch2 > np.max(ch2_masked)
        ):
            continue

        # Create mask for pixels below thresholds
        below_mask = (ch1_masked < thr_ch1) & (ch2_masked < thr_ch2)

        if np.sum(below_mask) < 10:  # Need minimum number of pixels
            continue

        # Calculate correlation for pixels below threshold
        ch1_below = ch1_masked[below_mask]
        ch2_below = ch2_masked[below_mask]

        if len(ch1_below) > 1 and np.std(ch1_below) > 0 and np.std(ch2_below) > 0:
            correlation = np.corrcoef(ch1_below.ravel(), ch2_below.ravel())[0, 1]

            # Find threshold where correlation is closest to zero
            if abs(correlation) < abs(best_correlation):
                best_correlation = correlation
                best_thr_ch1 = thr_ch1
                best_thr_ch2 = thr_ch2

    # Calculate slope and intercept for reporting
    slope = pc_norm[1] / pc_norm[0]
    intercept = ch2_mean - slope * ch1_mean

    return best_thr_ch1, best_thr_ch2, slope, intercept
