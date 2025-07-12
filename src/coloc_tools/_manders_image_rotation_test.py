import numpy as np
from _manders_correlation_coefficient import manders_correlation_coefficient


def manders_image_rotation_test(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    threshold_ch1: float = 0.0,
    threshold_ch2: float = 0.0,
) -> tuple[float, float, list[float], list[float], float, float]:
    """
    Perform image rotation randomization test for Manders' coefficients validation.

    This method applies systematic rotations (90°, 180°, 270°) and flips to one channel
    relative to the other, breaking spatial relationships while preserving local
    patterns.

    For non-square images, the function automatically pads them to square dimensions
    with zeros before applying rotations to ensure valid comparisons.

    Parameters
    ----------
    channel_1 : np.ndarray
        First fluorescent channel (kept fixed)
    channel_2 : np.ndarray
        Second fluorescent channel (will be rotated and flipped)
    threshold_ch1 : float, optional
        Intensity threshold for channel 1
    threshold_ch2 : float, optional
        Intensity threshold for channel 2

    Returns
    -------
    Tuple containing:
        - float: Observed M1 coefficient
        - float: Observed M2 coefficient
        - List[float]: M1 coefficients from rotation/flip iterations
        - List[float]: M2 coefficients from rotation/flip iterations
        - float: P-value for M1 (fraction of rotation M1 >= observed M1)
        - float: P-value for M2 (fraction of rotation M2 >= observed M2)
    """

    def _pad_to_square(image: np.ndarray) -> np.ndarray:
        """Pad image to square dimensions with zeros."""
        h, w = image.shape
        max_dim = max(h, w)
        # Calculate padding needed
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        # Pad the image symmetrically
        padded = np.pad(
            image,
            ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w)),
            mode="constant",
            constant_values=0,
        )

        return padded

    def _rotate_and_flip_image(
        image: np.ndarray, rotation: int, flip_type: str
    ) -> np.ndarray:
        """Rotate image by specified angle and optionally flip."""
        # Apply rotation (k=1 means 90°, k=2 means 180°, k=3 means 270°)
        rotated = np.rot90(image, k=rotation)

        # Apply flip if requested
        if flip_type == "horizontal":
            rotated = np.fliplr(rotated)
        elif flip_type == "vertical":
            rotated = np.flipud(rotated)

        return rotated

    # Check if images are square, if not pad them
    if (
        channel_1.shape[0] != channel_1.shape[1]
        or channel_2.shape[0] != channel_2.shape[1]
    ):
        # Pad both channels to square dimensions
        channel_1_padded = _pad_to_square(channel_1)
        channel_2_padded = _pad_to_square(channel_2)
    else:
        channel_1_padded = channel_1
        channel_2_padded = channel_2

    # Calculate observed Manders' coefficients
    observed_m1, observed_m2 = manders_correlation_coefficient(
        channel_1_padded, channel_2_padded, threshold_ch1, threshold_ch2
    )

    # Initialize lists to store rotation/flip coefficients
    rotation_m1_values = []
    rotation_m2_values = []

    # Apply all combinations of rotations and flips
    transformations = [
        (1, None),  # 90° rotation
        (1, "horizontal"),  # 90° rotation + horizontal flip
        (1, "vertical"),  # 90° rotation + vertical flip
        (2, None),  # 180° rotation
        (2, "horizontal"),  # 180° rotation + horizontal flip
        (2, "vertical"),  # 180° rotation + vertical flip
        (3, None),  # 270° rotation
        (3, "horizontal"),  # 270° rotation + horizontal flip
        (3, "vertical"),  # 270° rotation + vertical flip
    ]

    for rotation, flip_type in transformations:
        # Apply rotation and flip to channel 2 (using padded version)
        transformed_ch2 = _rotate_and_flip_image(channel_2_padded, rotation, flip_type)

        # Calculate Manders' coefficients with transformed channel 2
        rotation_m1, rotation_m2 = manders_correlation_coefficient(
            channel_1_padded, transformed_ch2, threshold_ch1, threshold_ch2
        )
        rotation_m1_values.append(rotation_m1)
        rotation_m2_values.append(rotation_m2)

    # Calculate p-values
    n_transformations = len(rotation_m1_values)
    p_value_m1 = np.sum(np.array(rotation_m1_values) >= observed_m1) / n_transformations
    p_value_m2 = np.sum(np.array(rotation_m2_values) >= observed_m2) / n_transformations

    return (
        float(observed_m1),
        float(observed_m2),
        rotation_m1_values,
        rotation_m2_values,
        float(p_value_m1),
        float(p_value_m2),
    )
