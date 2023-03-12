import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from datetime import datetime
import logging
# from datetime import datetime
import tensorflow as tf
# from .estimate_convergence import estimate_convergence
from .get_random_multiplier import get_random_multiplier
from ..print_large import print_large
# from .predict_next_lr import predict_next_lr
# import shutil
# import os
# import time
import math

logging.basicConfig(level=logging.INFO)


def get_existing_epochs(model_meta):
    existing_epochs = []

    if 'lr_history' not in model_meta:
        return []

    for lr_meta in model_meta['lr_history']:
        for epoch in lr_meta['epoch_history']:
            existing_epochs.append({'l': epoch['loss'], 't': epoch['time'],
                                   's': epoch['sample_size'], 'b': epoch['batch_size'], 'g': epoch.get('gpu', True), 'lr': lr_meta['lr']})

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
        self.epochs_since_lr_multiplier_adjusted = 0

        print_large('training_manger initialized, epochs in history:',
                    len(self.model_meta['training_stats']['epochs']))

    def get_next_lr(self, lr):
        if len(self.model_meta['training_stats']['epochs']) == 0:
            return 0.0001

        next_lr = 0.002000458 + (-0.000001246426 - 0.002000458) / (1 + math.pow(
            self.model_meta['training_stats']['epochs'][-1]['l'] / 3.323395, 10.29872))
        random_multiplier = get_random_multiplier(1.25)
        result = next_lr * self.lr_multiplier * random_multiplier

        print('next_lr from formula', next_lr)
        print('lr_multiplier', self.lr_multiplier)
        print('random_multiplier', random_multiplier)
        print('next_lr', result)
        print('')

        return result

    def get_optimizer(self):
        optimizer = tf.keras.optimizers.legacy.Adam(self.get_next_lr(None))
        return optimizer

    def add_to_stats(self, loss, lr, time, sample_size, batch_size, gpu):
        self.epochs_since_lr_multiplier_adjusted += 1

        self.model_meta['training_stats']['epochs'].append(
            {'l': loss, 't': time,
             's': sample_size, 'b': batch_size, 'lr': lr, 'g': gpu})

        if len(self.model_meta['training_stats']['epochs']) >= 200:
            last_100_loss = [
                epoch['l'] for epoch in self.model_meta['training_stats']['epochs'][-100:]]
            prev_100_loss = [
                epoch['l'] for epoch in self.model_meta['training_stats']['epochs'][-200:-100]]
            loss_diff_in_past_200 = sum(
                last_100_loss)/len(last_100_loss) - sum(prev_100_loss)/len(prev_100_loss)
            print('loss diff in past 200 epochs:', loss_diff_in_past_200)

            if self.epochs_since_lr_multiplier_adjusted > 100 and loss_diff_in_past_200 > 0:
                self.lr_multiplier /= 3
                self.epochs_since_lr_multiplier_adjusted = 0
                print_large('New lr multiplier:', self.lr_multiplier)

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

        print('')
