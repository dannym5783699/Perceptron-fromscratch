from typing import Callable, Tuple

import numpy as np


def generate_binary_data(
    condition_function: Callable,
    n_samples: int = 100,
    n_features: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate binary data for testing the perceptron.

    Args:
        condition_function (Callable): Function that classifies the samples into two classes. Should
            accept a 2D array of shape (n_samples, n_features) and return a boolean array of shape (n_samples).
        n_samples (int): Number of samples to generate per class.
        n_features (int): Number of features to generate.
        seed (int, optional): Optional seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X, y where X is the generated data and y are the labels.
    """

    # Generate a large batch of samples,
    # then filter the samples based on the classifying function
    # Operates under the assumption that this will generate enough samples
    # ...it should probably work for most simple cases
    samples_per_batch = 10 * n_samples
    rng = np.random.default_rng(seed)

    while True:
        # Generate a large batch of samples
        x = rng.uniform(-10, 10, (samples_per_batch, n_features))

        # Filter the samples based on the classifying function
        try:
            class1 = x[condition_function(x) > 0]
            class2 = x[condition_function(x) < 0]
        except IndexError as e:
            if "too many indices" in str(e):
                raise ValueError(
                    "Condition function must return a 1D array of the same length as the number of samples."
                )
            elif "out of bounds" in str(e):
                raise ValueError(
                    f"""Condition function must accept a 2D array of shape ({n_samples}, {n_features})"""
                )
            else:
                raise e

        # Verify we have enough samples, otherwise try again
        if len(class1) >= n_samples and len(class2) >= n_samples:
            break

    # Select the first n_samples from each class
    class1 = class1[:n_samples]
    class2 = class2[:n_samples]

    # Concatenate the two classes
    x = np.vstack([class1, class2])

    # Create the labels 1 and -1
    y = np.hstack([np.ones(n_samples), -np.ones(n_samples)])

    # Shuffle the data
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    x = x[idx]
    y = y[idx]

    return x, y
