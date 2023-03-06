import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)


class TrainingStats:
    def __init__(self, initial_lr, initial_batch_size, loss_history=None, lr_history=None, time_history=None, sample_size_history=None, batch_size_history=None):
        self.lr_history = [] if lr_history is None else lr_history
        self.loss_history = [] if loss_history is None else loss_history
        self.time_history = [] if time_history is None else time_history
        self.sample_size_history = [] if sample_size_history is None else sample_size_history
        self.batch_size_history = [] if batch_size_history is None else batch_size_history
        self.best_loss = np.inf
        self.lr = initial_lr
        self.batch_size = initial_batch_size

        if len(self.lr_history) > 0 and self.lr_history[-1] != initial_lr:
            self.lr_history.append(initial_lr)
            self.batch_size_history.append(initial_batch_size)

    def add_to_stats(self, loss, lr, time, sample_size, batch_size):
        self.loss_history.append(loss)
        self.lr_history.append(lr)
        self.time_history.append(time)
        self.sample_size_history.append(sample_size)
        self.batch_size_history.append(batch_size)

        self.lr = lr
        self.batch_size = batch_size

        if loss < self.best_loss:
            self.best_loss = loss
            self.lr = lr
            self.batch_size = batch_size

    def get_next_lr(self):
        return self.lr, 'Learning rate adjustment disabled'

        lr_len = len(self.lr_history)
        loss_len = len(self.loss_history)
        history_length = min(lr_len, loss_len, len(self.time_history), len(
            self.sample_size_history), len(self.batch_size_history))

        if history_length < 2:
            next_lr = self.lr
            log = f'Using the initial learning rate: {next_lr:.6f}'
            return next_lr, log

        loss_diff = np.diff(self.loss_history[-history_length:])
        if len(loss_diff) < 2:
            next_lr = self.lr
            log = f'Using the current learning rate: {next_lr:.6f}'
            return next_lr, log

        loss_diff_diff = np.diff(loss_diff)

        if len(loss_diff_diff) > 0:
            inflection_point_index = np.argmax(loss_diff_diff) + 1
        else:
            inflection_point_index = 0

        ideal_lr = self.lr * (2.0 ** (inflection_point_index - lr_len))

        if self.best_loss == np.inf:
            ideal_lr_sgd = self.lr
        else:
            ideal_lr_sgd = self.best_loss * np.sqrt(self.lr / self.best_loss)

        if ideal_lr < self.lr:
            next_lr = ideal_lr
            reason = 'ideal_lr'
        elif ideal_lr_sgd < self.lr:
            next_lr = ideal_lr_sgd
            reason = 'ideal_lr_sgd'
        else:
            next_lr = self.lr
            reason = 'previous'

        if next_lr != self.lr:
            improvement = (
                (self.loss_history[-1] - self.loss_history[-2]) / self.loss_history[-2]) * 100.0
            log = f'Changing the learning rate from {self.lr:.6f} to {next_lr:.6f} ({reason}), expected improvement: {improvement:.6f}%'
        else:
            log = f'Using the initial learning rate: {next_lr:.6f}'

        return next_lr, log

    def get_batch_size(self, samples_per_epoch):
        if not self.batch_size_history:
            next_batch_size = self.batch_size
            return next_batch_size, f'Using the default batch size: {next_batch_size}\n'

        lr_len = len(self.lr_history)
        loss_len = len(self.loss_history)
        batch_size_len = len(self.batch_size_history)
        history_length = min(lr_len, loss_len, len(self.time_history), len(
            self.sample_size_history), batch_size_len)

        if history_length < 2:
            next_batch_size = self.batch_size_history[-1]
            return next_batch_size, f'Using the previous batch size: {next_batch_size}\n'

        loss_diff = np.diff(self.loss_history[-history_length:])
        if len(loss_diff) < 2:
            next_batch_size = self.batch_size_history[-1]
            return next_batch_size, f'Using the previous batch size: {next_batch_size}\n'

        loss_diff_diff = np.diff(loss_diff)
        inflection_point_index = np.argmax(loss_diff_diff) + 1

        ideal_batch_size_up = int(
            self.batch_size_history[-1] * (2.0 ** (inflection_point_index - batch_size_len)))
        ideal_batch_size_down = int(
            self.batch_size_history[-1] / (2.0 ** (inflection_point_index - batch_size_len)))

        batch_size_diff_up = ideal_batch_size_up - self.batch_size_history[-1]
        batch_size_diff_down = self.batch_size_history[-1] - \
            ideal_batch_size_down

        if batch_size_diff_up > 0 and batch_size_diff_up < self.batch_size_history[-1]:
            expected_improvement = (batch_size_diff_up / self.batch_size_history[-1]) * (
                self.loss_history[-1] / samples_per_epoch)
            if expected_improvement >= self.min_expected_improvement:
                next_batch_size = ideal_batch_size_up
                reason = 'Increasing the batch size to optimize for faster convergence'
                message = f'{reason}\nBatch size changing from {self.batch_size_history[-1]} to {next_batch_size}, a {((next_batch_size - self.batch_size_history[-1]) / self.batch_size_history[-1]) * 100:.2f}% increase\nExpected improvement: {expected_improvement:.6f}\n'
            else:
                next_batch_size = self.batch_size_history[-1]
                message = f'Using the previous batch size: {next_batch_size}\n'
        elif batch_size_diff_down > 0:
            expected_improvement = (batch_size_diff_down / self.batch_size_history[-1]) * (
                self.loss_history[-1] / samples_per_epoch)
            if expected_improvement >= self.min_expected_improvement:
                next_batch_size = ideal_batch_size_down
                reason = 'Decreasing the batch size to optimize for better generalization'
                message = f'{reason}\nBatch size changing from {self.batch_size_history[-1]} to {next_batch_size}, a {((next_batch_size - self.batch_size_history[-1]) / self.batch_size_history[-1]) * 100:.2f}% decrease\nExpected improvement: {expected_improvement:.6f}\n'
            else:
                next_batch_size = self.batch_size_history[-1]
                message = f'Using the previous batch size: {next_batch_size}\n'
        else:
            next_batch_size = self.batch_size_history[-1]
            message = f'Using the previous batch size: {next_batch_size}\n'

        return next_batch_size, message

    def save_stats(self, folder):
        if len(self.loss_history) == 0:
            return

        filename = f'{folder}/training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

        with PdfPages(filename) as pdf:
            # Plot the loss history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.plot(self.loss_history)
            pdf.savefig(fig)
            plt.close()

            # Plot the learning rate history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning rate')
            ax.plot(self.lr_history[1:])
            pdf.savefig(fig)
            plt.close()

            # Plot the time history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.plot(self.time_history)
            pdf.savefig(fig)
            plt.close()

            # Plot the sample size history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Sample size')
            ax.plot(self.sample_size_history)
            pdf.savefig(fig)
            plt.close()

            # Plot the batch size history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Batch size')
            ax.plot(self.batch_size_history)
            pdf.savefig(fig)
            plt.close()

    def save_stats2(self, folder):
        if len(self.loss_history) == 0:
            return

        filename = f'{folder}/training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

        with PdfPages(filename) as pdf:
            # Plot the loss history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.plot(self.loss_history, 'r')
            ax.set_title('Loss history')
            ax.text(0.95, 0.95, f'Best loss: {self.best_loss:.6f}', ha='right',
                    va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
            pdf.savefig(fig)
            plt.close()

            # Plot the learning rate history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning rate')
            ax.plot(self.lr_history[1:], 'g')
            ax.set_title('Learning rate history')
            ax.text(0.95, 0.95, f'Best learning rate: {self.lr:.6f}', ha='right',
                    va='top', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
            pdf.savefig(fig)
            plt.close()

            # Plot the time history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Time (seconds)')
            ax.plot(self.time_history, 'b')
            ax.set_title('Time history')
            pdf.savefig(fig)
            plt.close()

            # Plot the sample size history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Sample size')
            ax.plot(self.sample_size_history, 'm')
            ax.set_title('Sample size history')
            pdf.savefig(fig)
            plt.close()

            # Plot the batch size history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Batch size')
            ax.plot(self.batch_size_history, 'c')
            ax.set_title('Batch size history')
            pdf.savefig(fig)
            plt.close()

            # Plot a 3D graph of learning rate and loss history
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning rate')
            ax.set_zlabel('Loss')
            ax.plot(self.lr_history[1:], self.loss_history,
                    range(1, len(self.loss_history)+1), 'g')
            ax.set_title('Learning rate and loss history')
            pdf.savefig(fig)
            plt.close()

    def get_estimated_eta(self):
        logging.info(f'Length of lr_history: {len(self.lr_history)}')
        logging.info(f'Length of loss_history: {len(self.loss_history)}')
        logging.info(f'Length of time_history: {len(self.time_history)}')
        logging.info(
            f'Length of sample_size_history: {len(self.sample_size_history)}')
        logging.info(
            f'Length of batch_size_history: {len(self.batch_size_history)}')

        history_length = len(self.lr_history)
        if history_length < 2:
            return float('inf')

        # Estimate the time remaining based on the loss slope
        loss_diff = np.diff(self.loss_history[-history_length:])
        if len(loss_diff) < 2:
            return float('inf')
        loss_diff_diff = np.diff(loss_diff)
        inflection_point_index = np.argmax(loss_diff_diff) + 1
        loss_slope = loss_diff_diff[inflection_point_index -
                                    1] if inflection_point_index > 0 else np.mean(loss_diff_diff)

        # Estimate the time remaining based on the learning rate slope
        lr_diff = np.diff(self.lr_history[-history_length:])
        lr_diff_diff = np.diff(lr_diff)
        inflection_point_index = np.argmax(lr_diff_diff) + 1
        lr_slope = lr_diff_diff[inflection_point_index -
                                1] if inflection_point_index > 0 else np.mean(lr_diff_diff)

        # Estimate the time remaining based on the recent epoch durations
        time_diff = np.diff(self.time_history[-history_length:])
        recent_time = np.mean(
            self.time_history[-min(5, len(self.time_history)):])
        time_slope = np.mean(time_diff[-min(5, len(time_diff)):]) / recent_time

        # Estimate the total remaining time
        remaining_epochs = 1000 - history_length
        total_time = (remaining_epochs / time_slope) * \
            (loss_slope / lr_slope) if lr_slope > 0 else float('inf')
        return total_time

    def print_stats(self):
        if not self.lr_history and not self.loss_history and not self.time_history and not self.sample_size_history and not self.batch_size_history:
            print('No history available yet.')
            return

        if self.loss_history:
            current_loss = self.loss_history[-1]
            print(f'  Loss: {current_loss:.6f}')

        if self.lr_history:
            current_lr = self.lr_history[-1]
            print(f'  Learning rate: {current_lr:.6f}')

        if self.time_history:
            current_time = self.time_history[-1]
            print(f'  Time: {current_time:.2f} seconds')

        if self.sample_size_history:
            current_sample_size = self.sample_size_history[-1]
            print(f'  Samples: {current_sample_size}')

        if self.batch_size_history:
            current_batch_size = self.batch_size_history[-1]
            print(f'  Batch size: {current_batch_size}')

        if self.time_history:
            elapsed_time_seconds = sum(self.time_history)
            remaining_time_seconds = self.get_estimated_eta()

            elapsed_hours, _ = divmod(elapsed_time_seconds, 60 * 60)
            elapsed_days, elapsed_hours = divmod(elapsed_hours, 24)
            remaining_hours, _ = divmod(remaining_time_seconds, 60 * 60)
            remaining_days, remaining_hours = divmod(remaining_hours, 24)

            print(
                f'  Elapsed time: {elapsed_days:.0f} days, {elapsed_hours:.0f} hours')

            if remaining_time_seconds >= 0:
                print(
                    f'  Remaining time: {remaining_days:.0f} days, {remaining_hours:.0f} hours')
            else:
                print('  Remaining time: N/A')

        print('')

    def calculate_ideal_lr_and_batch_size(self):
        lr_len = len(self.lr_history)
        loss_len = len(self.loss_history)
        batch_size_len = len(self.batch_size_history)
        history_length = min(lr_len, loss_len, len(self.time_history), len(
            self.sample_size_history), batch_size_len)

        if history_length < 2:
            next_lr = self.lr_history[-1]
            next_batch_size = self.batch_size_history[-1]
            print(
                f'Using the previous learning rate: {next_lr:.6f} and batch size: {next_batch_size:.6f}')
            return next_lr, next_batch_size

        loss_diff = np.diff(self.loss_history[-history_length:])
        loss_diff_diff = np.diff(loss_diff)

        inflection_point_index = np.argmax(loss_diff_diff) + 1

        ideal_lr = self.lr_history[-1] * \
            (2.0 ** (inflection_point_index - lr_len))
        ideal_lr_sgd = self.lr * \
            np.sqrt(self.lr_history[-1] / self.lr)
        if ideal_lr < self.lr_history[-1]:
            next_lr = ideal_lr
            reason = 'ideal_lr'
        elif ideal_lr_sgd < self.lr_history[-1]:
            next_lr = ideal_lr_sgd
            reason = 'ideal_lr_sgd'
        else:
            next_lr = self.lr_history[-1]
            reason = 'previous'

        ideal_batch_size = self.batch_size_history[-1] * \
            (2.0 ** (inflection_point_index - batch_size_len))
        if ideal_batch_size < self.batch_size_history[-1]:
            next_batch_size = ideal_batch_size
            reason = 'ideal_batch_size'
        else:
            next_batch_size = self.batch_size_history[-1]
            reason = 'previous'

        if next_lr != self.lr_history[-1] or next_batch_size != self.batch_size_history[-1]:
            print(
                f'Changing the learning rate from {self.lr_history[-1]:.6f} to {next_lr:.6f} ({reason})')
            print(
                f'Changing the batch size from {self.batch_size_history[-1]:.6f} to {next_batch_size:.6f} ({reason})')
        else:
            print(
                f'Using the previous learning rate: {next_lr:.6f} and batch size: {next_batch_size:.6f}')

        return next_lr, next_batch_size

    def get_history(self):
        return {
            'lr_history': self.lr_history,
            'loss_history': self.loss_history,
            'time_history': self.time_history,
            'sample_size_history': self.sample_size_history,
            'batch_size_history': self.batch_size_history,
            'best_loss': self.best_loss,
            'lr': self.lr,
            'batch_size': self.batch_size
        }
