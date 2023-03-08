import numpy as np
from scipy.optimize import curve_fit


def get_curve_minimums(arr):
    """
    Compute the weighted average of "x" values for an array of data points.

    Parameters:
        arr (numpy.ndarray): The array of data points to process.

    Returns:
        float: The weighted average of the "x" values.

    """
    # Compute the number of pairs of adjacent points in the array
    n = len(arr) - 1
    # Compute the weights for each pair of adjacent points
    # The weights increase quadratically from 1 to n^2
    weights = np.arange(1, n + 1) ** 2
    # Compute the "x" value for each pair of adjacent points
    x_values = [arr[i] + (arr[i+1] - arr[i]) * 2 for i in range(n)]
    # Compute the weighted average of the "x" values
    average_x = np.sum(weights * x_values) / np.sum(weights)
    # Return the weighted average of the "x" values
    return average_x
