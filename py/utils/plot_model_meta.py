import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
from .get_training_forecast import get_training_forecast_ai

# max_plotted_val = 1.55
# min_plotted_val = 1.5


def plot_model_meta(model_meta, filename, plot_forecast=False, title='Loss and learning rate history'):
    if len(model_meta['training_stats']['epochs']) == 0:
        return

    # Create subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Loss History
    loss_history = [epoch['l']
                    for epoch in model_meta['training_stats']['epochs']]
    fig.add_trace(go.Scatter(x=list(range(len(loss_history))), y=loss_history,
                             mode='lines', name='Loss', line=dict(color='red', width=1)), secondary_y=False)

    # Learning Rate History
    lr_history = [epoch['lr']
                  for epoch in model_meta['training_stats']['epochs']]
    fig.add_trace(go.Scatter(x=list(range(len(lr_history))), y=lr_history,
                             mode='lines', name='Learning Rate', line=dict(color='blue', width=1)), secondary_y=True)

    # Forecast Data
    if plot_forecast:
        forecast_data = get_training_forecast_ai(model_meta)
        fig.add_trace(go.Scatter(x=list(range(len(forecast_data))), y=forecast_data,
                                 mode='lines', name='Forecast', line=dict(color='green', width=1)), secondary_y=True)

    # Layout settings
    fig.update_layout(
        title=title,
        xaxis_title='Epochs',
        yaxis_title='Loss',
        xaxis=dict(rangeslider=dict(visible=True)),
        yaxis=dict(fixedrange=False),
        yaxis2=dict(fixedrange=False)
    )

    fig.update_yaxes(title_text="Learning Rate", secondary_y=True)
    fig.update_yaxes(
        # range=[min_plotted_val, max_plotted_val],
        secondary_y=False)

    # Export to HTML
    if filename:
        pio.write_html(fig, file=filename)

    return pio.to_html(fig, full_html=False)
