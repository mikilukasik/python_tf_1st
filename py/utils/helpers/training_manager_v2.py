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


def get_existing_epochs(model_meta):
    existing_epochs = []

    if 'lr_history' not in model_meta:
        return []

    for lr_meta in model_meta['lr_history']:
        for epoch in lr_meta['epoch_history']:
            existing_epochs.append({'l': epoch['loss'], 't': epoch['time'],
                                   's': epoch['sample_size'], 'b': epoch['batch_size'], 'g': epoch['gpu'], 'lr': lr_meta['lr']})

    return existing_epochs


def transtorm_old_model_meta(model_meta):
    if (model_meta.get('training_stats')):
        return

    existing_epochs = get_existing_epochs(model_meta)

    model_meta['training_stats'] = {'epochs': existing_epochs}


def get_random_lr():
    # Define the range of values
    min_val = 0.0000001
    max_val = 0.001

    # Generate a random value on a log scale
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    log_random_val = np.random.uniform(log_min, log_max)

    rnd_val = 10 ** log_random_val

    print('rnd_val', rnd_val)

    # Transform the log value back to the original scale
    return rnd_val


class TrainingManagerV2:
    def __init__(self, model_meta=None, forced_lr=None, lr_multiplier=1, batch_size=64):
        if model_meta is None:
            raise ValueError("model_meta cannot be None")

        transtorm_old_model_meta(model_meta)

        self.model_meta = model_meta

        self.forced_lr = forced_lr
        self.lr_multiplier = lr_multiplier
        self.batch_size = batch_size

    def get_next_lr(self, lr):
        return get_random_lr() * self.lr_multiplier
        return self.lr_meta['lr']

    def get_optimizer(self):
        optimizer = tf.keras.optimizers.legacy.Adam(self.get_next_lr(None))
        return optimizer

    def add_to_stats(self, loss, lr, time, sample_size, batch_size, gpu):
        # if not np.isclose(lr, self.lr_meta['lr'], rtol=1e-7, atol=1e-7):
        #     print(lr, self.lr_meta['lr'])
        #     raise ValueError("learning rate mismatch")
        self.model_meta['training_stats']['epochs'].append(
            {'l': loss, 't': time,
             's': sample_size, 'b': batch_size, 'lr': lr, 'g': gpu}

        )

        # self.lr_meta['epoch_history'].append(
        #     {'loss': loss, 'time': time, 'sample_size': sample_size, 'batch_size': batch_size, 'gpu': gpu})
        # self.lr_meta['avg_epoch_time'] = sum(
        #     [x['time'] for x in self.lr_meta['epoch_history']]) / len(self.lr_meta['epoch_history'])

        # if len(self.lr_meta['epoch_history']) >= 100:
        #     last_50_losses = [x['loss']
        #                       for x in self.lr_meta['epoch_history'][-50:]]
        #     prev_50_losses = [x['loss']
        #                       for x in self.lr_meta['epoch_history'][-100:-50]]
        #     last_50_loss_avg = sum(last_50_losses) / len(last_50_losses)
        #     prev_50_loss_avg = sum(prev_50_losses) / len(prev_50_losses)
        #     loss_diff = last_50_loss_avg - prev_50_loss_avg

        #     print("Loss difference between last 50 and previous 50 epochs:", loss_diff)

        #     if (loss_diff * -1) < self.lr_meta['lr']*100:
        #         self.model_meta['lr'] = self.lr_meta['lr']/2
        #         print("New learning rate will be:", self.model_meta['lr'])

        #         self.lr_meta['finished'] = datetime.now().isoformat()
        #         self.lr_meta['active'] = False

        #         self.lr_meta = {'active': True, 'lr': self.model_meta['lr'], 'epoch_history': [
        #         ], 'started': datetime.now().isoformat(), 'avg_epoch_time': None}
        #         self.model_meta['lr_history'].append(self.lr_meta)

    def print_stats(self):
        if not self.model_meta['training_stats']['epochs'] or len(self.model_meta['training_stats']['epochs']) < 1:
            print('No history available yet.')
            return

        last_epoch = self.model_meta['training_stats']['epochs'][-1]

        print('Last epoch stats:')

        print(f'  Loss: {last_epoch["l"]:.6f}')
        print(
            f'  Learning rate: {last_epoch["lr"]:.6f}')
        print(f'  Time: {last_epoch["t"]:.2f} seconds')

        elapsed_time_seconds = sum(epoch['t']
                                   for epoch in self.model_meta['training_stats']['epochs'])

        elapsed_minutes, elapsed_seconds = divmod(elapsed_time_seconds, 60)
        elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
        elapsed_days, elapsed_hours = divmod(elapsed_hours, 24)

        print(
            f'  Elapsed time: {elapsed_days:.0f} days, {elapsed_hours:.0f} hours, {elapsed_minutes:.0f} minutes, {elapsed_seconds:.0f} seconds')

        # convergence_value, epochs_remaining = estimate_convergence(
        #     [epoch['loss'] for epoch in self.lr_meta['epoch_history']])
        # print(f'  Estimated convergence value: {convergence_value:.6f}')
        # print(
        #     f'  Estimated epochs remaining until convergence: {epochs_remaining}')

        # avg_epoch_time = self.lr_meta['avg_epoch_time']
        # if avg_epoch_time is not None:
        #     remaining_time_seconds = avg_epoch_time * epochs_remaining
        #     remaining_minutes, remaining_seconds = divmod(
        #         remaining_time_seconds, 60)
        #     remaining_hours, remaining_minutes = divmod(remaining_minutes, 60)
        #     remaining_days, remaining_hours = divmod(remaining_hours, 24)

        #     print(
        #         f'  Estimated remaining time: {remaining_days:.0f} days, {remaining_hours:.0f} hours, {remaining_minutes:.0f} minutes, {remaining_seconds:.0f} seconds')

        print('')
