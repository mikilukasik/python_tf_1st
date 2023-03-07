import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import logging
from datetime import datetime
import tensorflow as tf
from .estimate_convergence import estimate_convergence
import shutil
import os
import time

logging.basicConfig(level=logging.INFO)


class TrainingManager:
    def __init__(self, initial_lr, initial_batch_size, force_lr, model_meta=None):
        if model_meta is None:
            raise ValueError("model_meta cannot be None")

        if not model_meta.get('lr_history'):
            model_meta['lr_history'] = [
                {'active': True, 'lr': initial_lr, 'epoch_history': [], 'started': datetime.now().isoformat(), 'avg_epoch_time': None}]
        elif force_lr and model_meta['lr_history'][-1]['lr'] != initial_lr:
            model_meta['lr_history'][-1]['active'] = False
            model_meta['lr_history'].append({'active': True, 'lr': initial_lr, 'epoch_history': [
            ], 'started': datetime.now().isoformat(), 'avg_epoch_time': None})

        self.model_meta = model_meta
        self.lr_meta = model_meta['lr_history'][-1]
        self.initial_batch_size = initial_batch_size

    def get_next_lr(self, lr):
        return self.lr_meta['lr']

    def get_optimizer(self):
        optimizer = tf.keras.optimizers.legacy.Adam(self.get_next_lr(None))
        return optimizer

    def add_to_stats(self, loss, lr, time, sample_size, batch_size, gpu):
        if not np.isclose(lr, self.lr_meta['lr'], rtol=1e-7, atol=1e-7):
            print(lr, self.lr_meta['lr'])
            raise ValueError("learning rate mismatch")

        self.lr_meta['epoch_history'].append(
            {'loss': loss, 'time': time, 'sample_size': sample_size, 'batch_size': batch_size, 'gpu': gpu})
        self.lr_meta['avg_epoch_time'] = sum(
            [x['time'] for x in self.lr_meta['epoch_history']]) / len(self.lr_meta['epoch_history'])

        if len(self.lr_meta['epoch_history']) >= 100:
            last_50_losses = [x['loss']
                              for x in self.lr_meta['epoch_history'][-50:]]
            prev_50_losses = [x['loss']
                              for x in self.lr_meta['epoch_history'][-100:-50]]
            last_50_loss_avg = sum(last_50_losses) / len(last_50_losses)
            prev_50_loss_avg = sum(prev_50_losses) / len(prev_50_losses)
            loss_diff = last_50_loss_avg - prev_50_loss_avg

            print("Loss difference between last 50 and previous 50 epochs:", loss_diff)

            if (loss_diff * -1) < self.lr_meta['lr']*100:
                self.model_meta['lr'] = self.lr_meta['lr']/2
                print("New learning rate will be:", self.model_meta['lr'])

                self.lr_meta['finished'] = datetime.now().isoformat()
                self.lr_meta['active'] = False

                self.lr_meta = {'active': True, 'lr': self.model_meta['lr'], 'epoch_history': [
                ], 'started': datetime.now().isoformat(), 'avg_epoch_time': None}
                self.model_meta['lr_history'].append(self.lr_meta)

    def print_stats(self):
        if not self.lr_meta['epoch_history']:
            print('No history available yet.')
            return

        last_epoch = self.lr_meta['epoch_history'][-1]

        print(f'  Loss: {last_epoch["loss"]:.6f}')
        print(
            f'  Learning rate: {self.model_meta["lr_history"][-1]["lr"]:.6f}')
        print(f'  Time: {last_epoch["time"]:.2f} seconds')

        elapsed_time_seconds = sum(epoch['time']
                                   for lr_meta in self.model_meta['lr_history']
                                   for epoch in lr_meta['epoch_history'])

        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
        elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
        elapsed_days, elapsed_hours = divmod(elapsed_hours, 24)

        print(
            f'  Elapsed time: {elapsed_days:.0f} days, {elapsed_hours:.0f} hours, {elapsed_minutes:.0f} minutes, {elapsed_seconds:.0f} seconds')

        convergence_value, epochs_remaining = estimate_convergence(
            [epoch['loss'] for epoch in self.lr_meta['epoch_history']])
        print(f'  Estimated convergence value: {convergence_value:.6f}')
        print(
            f'  Estimated epochs remaining until convergence: {epochs_remaining}')

        avg_epoch_time = self.lr_meta['avg_epoch_time']
        if avg_epoch_time is not None:
            remaining_time_seconds = avg_epoch_time * epochs_remaining
            remaining_minutes, remaining_seconds = divmod(
                remaining_time_seconds, 60)
            remaining_hours, remaining_minutes = divmod(remaining_minutes, 60)
            remaining_days, remaining_hours = divmod(remaining_hours, 24)

            print(
                f'  Estimated remaining time: {remaining_days:.0f} days, {remaining_hours:.0f} hours, {remaining_minutes:.0f} minutes, {remaining_seconds:.0f} seconds')

        print('')

    def save_stats(self, folder, save_copy=False):
        if len(self.model_meta['lr_history'][0]['epoch_history']) == 0:
            return

        filename = f'{folder}/training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

        # Create the PdfPages object
        pdf = PdfPages(filename)

        # Plot the loss history and learning rate
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Samples learned')
        ax1.set_ylabel('Loss', color='tab:red')
        loss_history = []
        for lr_meta in self.model_meta['lr_history']:
            for epoch in lr_meta['epoch_history']:
                loss_history.append(epoch['loss'])
            ax1.plot(loss_history, color='tab:red', linewidth=0.5)
            ax1.tick_params(axis='y', labelcolor='tab:red')
            ax1.set_title('Loss and learning rate history')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning rate', color='tab:blue')
        lr_history = []
        for lr_meta in self.model_meta['lr_history']:
            for epoch in lr_meta['epoch_history']:
                lr_history.append(lr_meta['lr'])
        ax2.plot(lr_history, color='tab:blue', linewidth=1)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Plot the batch size timeline
        batch_size_history = []
        for lr_meta in self.model_meta['lr_history']:
            for epoch in lr_meta['epoch_history']:
                batch_size_history.append(epoch['batch_size'])
        ax3 = ax1.twiny()
        ax3.set_xticks(np.arange(len(batch_size_history)))
        ax3.set_xticklabels(
            [str(batch_size_history[i])
             if i == 0 or batch_size_history[i] != batch_size_history[i-1]
             else '' for i in range(len(batch_size_history))],
            rotation=90, fontsize=4)
        ax3.set_xlabel('Batch size', fontsize=6)
        ax3.tick_params(axis='x', labelsize=6)

        plt.tight_layout()
        # Remove the green bars in the background of the graph
        ax1.set_facecolor('none')
        pdf.savefig(fig)
        plt.close()

        # Close the PdfPages object
        pdf.close()

        # Save a copy to the parent directory of the folder if save_copy is True
        if save_copy:
            parent_dir = os.path.abspath(os.path.join(folder, os.pardir))
            filename_root = f'training_stats.pdf'
            filename_root_path = f'{parent_dir}/{filename_root}'
            if os.path.isfile(filename_root_path):
                os.remove(filename_root_path)  # remove existing file
            shutil.copy(filename, filename_root_path)
