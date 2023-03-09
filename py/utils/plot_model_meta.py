from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from .get_training_forecast import get_training_forecast


def plot_model_meta(model_meta, filename, plot_forecast=False):
    if len(model_meta['lr_history'][0]['epoch_history']) == 0:
        return

    # Create the PdfPages object
    pdf = PdfPages(filename)

    # Plot the loss history and learning rate

    # if plot_forecast is true, use our new get_training_forecast method to generate one more set of data and plot it on the same graph here

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Samples learned')
    ax1.set_ylabel('Loss', color='tab:red')
    loss_history = []
    for lr_meta in model_meta['lr_history']:
        for epoch in lr_meta['epoch_history']:
            loss_history.append(epoch['loss'])
        ax1.plot(loss_history, color='tab:red', linewidth=0.5)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_title('Loss and learning rate history')

    if plot_forecast:
        forecast_data = get_training_forecast(model_meta)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning rate', color='tab:green')

        ax2.plot(forecast_data, color='tab:green', linewidth=1)
        ax2.tick_params(axis='y', labelcolor='tab:green')
    else:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning rate', color='tab:blue')
        lr_history = []
        for lr_meta in model_meta['lr_history']:
            for epoch in lr_meta['epoch_history']:
                lr_history.append(lr_meta['lr'])
        ax2.plot(lr_history, color='tab:blue', linewidth=1)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Plot the batch size timeline
    batch_size_history = []
    for lr_meta in model_meta['lr_history']:
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

    # if plot_forecast:
    #     # Generate a new set of loss data using get_training_forecast
    #     forecast_data = get_training_forecast(model_meta)

    #     print(forecast_data)

    #     forecast_samples = np.arange(
    #         len(loss_history), len(loss_history) + len(forecast_data))
    #     ax1.plot(forecast_samples, forecast_data,
    #              color='tab:green', linewidth=0.5)

    plt.tight_layout()
    # Remove the green bars in the background of the graph
    ax1.set_facecolor('none')
    pdf.savefig(fig)
    plt.close()

    # Close the PdfPages object
    pdf.close()
