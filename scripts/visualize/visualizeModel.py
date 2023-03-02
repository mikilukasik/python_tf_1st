import tensorflow as tf
from keras.utils.vis_utils import plot_model
# import visualkeras

model = tf.keras.models.load_model('./models/test/250_1.8948385057449342')
model.summary()

plot_model(model, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True, show_layer_activations=True)

# visualkeras.layered_view(model)
