from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_rate_stats(model_meta, filename, plot_forecast=False):
    if len(model_meta['lr_history'][0]['epoch_history']) == 0:
        return

    pdf = PdfPages(filename)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Loss')
    ax1.set_ylabel('Learning rate', color='tab:blue')
    ax1.set_xlim(right=2)
    ax1.set_ylim(top=0.0001)
    ax1.set_ylim(bottom=-0.00001)
    # ax1.set_xscale('log')  # Set the x-scale to logarithmic

    lr_history = []
    for lr_meta in model_meta['lr_history']:
        for epoch in lr_meta['epoch_history']:
            lr_history.append(lr_meta['lr'])

    loss_history = []
    for lr_meta in model_meta['lr_history']:
        for epoch in lr_meta['epoch_history']:
            loss_history.append(epoch['loss'])
    ax1.plot(loss_history, lr_history, '.', color='tab:blue', markersize=1)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Add a point to the beginning of the data to help with the curve fit
    loss_history.insert(0, 7.5)
    lr_history.insert(0, 0.003)
    loss_history.insert(0, 30)
    lr_history.insert(0, 0.003)

    loss_history.insert(0, 0)
    lr_history.insert(0, 0.000001)
    loss_history.insert(0, 1.5)
    lr_history.insert(0, 0.000001)

    ax1.set_xlim(left=min(loss_history))

    # Fit a curve to the data using polynomial regression
    z = np.polyfit(loss_history, lr_history, 3)
    p = np.poly1d(z)
    x_fit = np.linspace(min(loss_history), max(loss_history), 100)
    y_fit = p(x_fit)
    ax1.plot(x_fit, y_fit, '--', color='black', linewidth=1)

    # Add text annotation with function used to generate the curve
    ax1.annotate('Curve: {}'.format(p), xy=(0.5, 0.95), xycoords='axes fraction',
                 fontsize=8, ha='center', va='top')

    # Generate equivalent Python function
    function_str = 'def learning_rate_from_loss(loss):\n'
    function_str += '\tz = {}\n'.format(
        repr(z).replace('[', '').replace(']', ''))
    function_str += '\treturn np.polyval(z, loss)\n'
    ax1.annotate('Function: {}'.format(function_str), xy=(0.5, 0.85), xycoords='axes fraction',
                 fontsize=8, ha='center', va='top')

    ax1.set_title('Loss and learning rate history')
    plt.tight_layout()
    ax1.set_facecolor('none')
    pdf.savefig(fig)
    plt.close()

    pdf.close()
