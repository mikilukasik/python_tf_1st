import numpy as np


def get_ideal_lr(initial_lr):
    lr_history = [initial_lr]
    loss_history = []
    best_loss = np.inf
    best_lr = initial_lr

    def ideal_lr(loss):
        nonlocal best_loss, best_lr
        loss_history.append(loss)

        # Keep history lengths consistent
        lr_len = len(lr_history)
        loss_len = len(loss_history)

        if lr_len < loss_len:
            loss_history = loss_history[-lr_len:]
        elif lr_len > loss_len:
            lr_history = lr_history[-loss_len:]

        # Calculate the second derivative of the loss history
        loss_diff = np.diff(loss_history)
        loss_diff_diff = np.diff(loss_diff)

        # Find the index of the inflection point in the loss history
        inflection_point_index = np.argmax(loss_diff_diff) + 1

        # Calculate the ideal learning rate based on the inflection point
        ideal_lr = lr_history[-1] * (2.0 ** (inflection_point_index - lr_len))

        # Update the best learning rate and loss
        if loss < best_loss:
            best_loss = loss
            best_lr = lr_history[-1]

        # Calculate the ideal learning rate based on SGD
        ideal_lr_sgd = best_lr * np.sqrt(initial_lr / lr_history[-1])

        # Choose the smaller of the two ideal learning rates
        ideal_lr = min(ideal_lr, ideal_lr_sgd)

        # Add the current learning rate to the history
        lr_history.append(ideal_lr)

        return ideal_lr

    return ideal_lr
