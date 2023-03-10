import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def get_existing_epochs(model_meta):
    existing_epochs = []

    for lr_meta in model_meta['lr_history']:
        for epoch in lr_meta['epoch_history']:
            existing_epochs.append({'l': epoch['loss'], 't': epoch['time'],
                                   's': epoch['sample_size'], 'b': epoch['batch_size'], 'lr': lr_meta['lr']})

    return existing_epochs


def transtorm_old_model_meta(model_meta, initial_lr):
    if (model_meta.get('training_stats')):
        return

    existing_epochs = get_existing_epochs(model_meta)

    model_meta['training_stats'] = {'epochs': existing_epochs}


class ModelMetaManagerV2:
    def __init__(self, initial_lr, model_meta=None):
        if model_meta is None:
            raise ValueError("model_meta cannot be None")

        transtorm_old_model_meta(model_meta, initial_lr)

        if not model_meta.get('lr_history'):
            model_meta['lr_history'] = [
                {'active': True, 'lr': initial_lr, 'epoch_history': [], 'started': datetime.now().isoformat(), 'avg_epoch_time': None}]

        self.model_meta = model_meta
        self.lr_meta = model_meta['lr_history'][-1]

    def add_to_stats(self, loss, lr, time, sample_size, batch_size):
        self.lr_meta['epoch_history'].append(
            {'loss': loss, 'time': time, 'sample_size': sample_size, 'batch_size': batch_size})
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

    def get_next_lr(self):
        return self.lr_meta['lr']
