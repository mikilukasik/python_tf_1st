import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)


class ModelMetaManager:
    def __init__(self, initial_lr, model_meta=None):
        if model_meta is None:
            raise ValueError("model_meta cannot be None")

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

    def print_stats(self):
        if not self.lr_meta['epoch_history']:
            print('No history available yet.')
            return

        last_epoch = self.lr_meta['epoch_history'][-1]

        print(f'  Loss: {last_epoch["loss"]:.6f}')
        print(f'  Learning rate: {self.lr_meta["lr"]:.6f}')
        print(f'  Time: {last_epoch["time"]:.2f} seconds')
        # print(f'  Samples: {last_epoch["sample_size"]}')
        # print(f'  Batch size: {last_epoch["batch_size"]}')

        elapsed_time_seconds = sum(x['time']
                                   for x in self.lr_meta['epoch_history'])
        # remaining_time_seconds = self.get_estimated_eta()

        elapsed_hours, _ = divmod(elapsed_time_seconds, 60 * 60)
        elapsed_days, elapsed_hours = divmod(elapsed_hours, 24)
        # remaining_hours, _ = divmod(remaining_time_seconds, 60 * 60)
        # remaining_days, remaining_hours = divmod(remaining_hours, 24)

        print(
            f'  Elapsed time: {elapsed_days:.0f} days, {elapsed_hours:.0f} hours')

        # if remaining_time_seconds >= 0:
        #     print(f'  Remaining time: {remaining_days:.0f} days, {remaining_hours:.0f} hours')
        # else:
        #     print('  Remaining time: N/A')

        print('')

    def save_stats(self, folder):
        if len(self.model_meta['lr_history'][0]['epoch_history']) == 0:
            return

        filename = f'{folder}/training_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

        with PdfPages(filename) as pdf:
            # Plot the loss history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            loss_history = []
            for lr_meta in self.model_meta['lr_history']:
                for epoch in lr_meta['epoch_history']:
                    loss_history.append(epoch['loss'])
                    if not lr_meta['active']:
                        ax.axvline(x=len(loss_history)-1,
                                   color='g', linestyle='--')
            ax.plot(loss_history, 'r')
            ax.set_title('Loss history')
            pdf.savefig(fig)
            plt.close()

            # Plot the learning rate history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning rate')
            lr_history = [x['lr'] for x in self.model_meta['lr_history']]
            ax.plot(lr_history[1:], 'g')
            ax.set_title('Learning rate history')
            pdf.savefig(fig)
            plt.close()

            # Plot the loss history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            loss_history = [x['loss']
                            for x in self.model_meta['lr_history'][0]['epoch_history']]
            ax.plot(loss_history, 'r')
            ax.set_title('Loss history')
            pdf.savefig(fig)
            plt.close()

            # Plot the learning rate history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning rate')
            lr_history = [x['lr'] for x in self.model_meta['lr_history']]
            ax.plot(lr_history[1:], 'g')
            ax.set_title('Learning rate history')
            pdf.savefig(fig)
            plt.close()

            # Plot the epoch time history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Epoch time (seconds)')
            epoch_time_history = [x['avg_epoch_time']
                                  for x in self.model_meta['lr_history']]
            ax.plot(epoch_time_history[1:], 'b')
            ax.set_title('Epoch time history')
            pdf.savefig(fig)
            plt.close()

            # Plot the sample size history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Sample size')
            sample_size_history = [x['sample_size']
                                   for x in self.model_meta['lr_history'][0]['epoch_history']]
            ax.plot(sample_size_history, 'm')
            ax.set_title('Sample size history')
            pdf.savefig(fig)
            plt.close()

            # Plot the batch size history
            fig, ax = plt.subplots()
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Batch size')
            batch_size_history = [x['batch_size']
                                  for x in self.model_meta['lr_history'][0]['epoch_history']]
            ax.plot(batch_size_history, 'c')
            ax.set_title('Batch size history')
            pdf.savefig(fig)
            plt.close()

            # # Plot a 3D graph of learning rate and loss history
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.set_xlabel('Epoch')
            # ax.set_ylabel('Learning rate')
            # ax.set_zlabel('Loss')
            # lr_history = [x['lr'] for x in self.model_meta['lr_history']]
            # loss_history = [x['loss']
            #                 for x in self.model_meta['lr_history'][0]['epoch_history']]
            # ax.plot(lr_history[1:], loss_history,
            #         range(1, len(loss_history)+1), 'g')
            # ax.set_title('Learning rate and loss history')
            # pdf.savefig(fig)
            # plt.close()
