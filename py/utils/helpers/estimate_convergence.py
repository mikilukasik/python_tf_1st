import math


def estimate_convergence(loss_values):
    """
    Calculates the learning curve of the given loss values and estimates the
    final loss value the model will settle on.

    Args:
        loss_values (list[float]): List of float values representing the loss
            values of a Keras model after each epoch.

    Returns:
        A tuple (final_loss, remaining_epochs) where final_loss is the estimated
        final loss value the model will settle on, and remaining_epochs is the
        number of epochs remaining to reach that value.
    """
    if not loss_values:
        return None, None

    n = len(loss_values)
    if n == 1:
        return loss_values[0], None

    # Calculate the slope of the learning curve
    m = (loss_values[-1] - loss_values[0]) / (n - 1)

    # Calculate the y-intercept of the learning curve
    b = loss_values[0]

    # Calculate the estimated final loss value using the formula for a line
    final_loss = b + m * n

    # Calculate the number of epochs remaining to reach the final loss value
    remaining_epochs = math.ceil((final_loss - loss_values[-1]) / m)

    return final_loss, remaining_epochs
