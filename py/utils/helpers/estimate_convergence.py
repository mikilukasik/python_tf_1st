import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def estimate_convergence(data):
    """
    Given a list of data points representing the loss after each epoch, estimate
    the value it will settle on and how many more epochs are needed for convergence.
    """
    # Fit a linear curve to the data
    X = np.arange(1, len(data) + 1).reshape((-1, 1))
    y = np.array(data).reshape((-1, 1))
    polynomial_features = PolynomialFeatures(degree=1, include_bias=False)
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Estimate the settling value and epochs remaining
    settling_value = model.predict(
        np.array([[len(data) + 1]]))[0][0]

    # Check if the settling value is the same as the last data point
    if settling_value == data[-1]:
        return settling_value, 0

    # Calculate the derivative of the polynomial at the last data point
    dydx = model.coef_[0]

    # Estimate the epochs remaining
    epochs_remaining = abs(int((settling_value - data[-1]) / dydx))

    # Return the estimated settling value and epochs remaining
    return settling_value, epochs_remaining
