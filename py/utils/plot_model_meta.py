from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from .get_training_forecast import get_training_forecast_ai


def plot_model_meta(model_meta, filename, plot_forecast=False, title='Loss and learning rate history'):
    if len(model_meta['training_stats']['epochs']) == 0:
        return

    pdf = PdfPages(filename)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Samples learned')
    ax1.set_ylabel('Loss', color='tab:red')
    loss_history = []
    for epoch in model_meta['training_stats']['epochs']:
        loss_history.append(epoch['l'])
    ax1.plot(loss_history, color='tab:red', linewidth=0.5)
    ax1.set_title(title, fontdict={'fontsize': 8})
    ax1.set_ylim(1.45, 1.65)    # ax1.set_ylim(1.7, 1.8)


    if bool(plot_forecast):
        forecast_data = get_training_forecast_ai(model_meta)

        ax3 = ax1.twinx()
        ax3.set_ylabel('Forecast', color='tab:green')

        ax3.plot(forecast_data, color='tab:green', linewidth=1)
        ax3.tick_params(axis='y', labelcolor='tab:green')
        ax1.set_ylim(ax3.get_ylim())

        print(ax3.get_ylim())

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning rate', color='tab:blue')
    lr_history = []
    # for lr_meta in model_meta['lr_history']:
    for epoch in model_meta['training_stats']['epochs']:
        lr_history.append(epoch['lr'])
    ax2.plot(lr_history, color='tab:blue', linewidth=1)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Plot the batch size timeline
    batch_size_history = []
    # for lr_meta in model_meta['lr_history']:
    for epoch in model_meta['training_stats']['epochs']:
        batch_size_history.append(epoch['b'])
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
    ax1.set_facecolor('none')
    pdf.savefig(fig)
    plt.close()

    pdf.close()
