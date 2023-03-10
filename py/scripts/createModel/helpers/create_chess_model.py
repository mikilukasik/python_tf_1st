from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Activation, Add, Multiply, LeakyReLU
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Softmax, Permute, Lambda
from .attention_block import attention_block


def create_chess_model(input_shape=(8, 8, 14), conv_filters=[],
                       residual_filters=[128, 128, 128, 128], use_attention=False,
                       inter_channel=64, num_dense_layers=2, dense_units=[1024, 512],
                       dropout_rate=0, num_output_units=1837, use_batchnorm=True, activation='relu'):
    '''
    Create a Keras model for predicting the next move in a chess game using a dataset of chess positions.

    Parameters:
    input_shape (tuple): The shape of the input chess position. Default is (8, 8, 14).
    conv_filters (list): The number of filters for each convolutional layer. Default is [32, 64, 128].
    residual_filters (list): The number of filters for each residual block. Default is [128, 128, 128].
    use_attention (bool): Whether to use attention blocks in the residual blocks. Default is True.
    inter_channel (int): The number of filters for the attention block. Default is 64.
    num_dense_layers (int): The number of dense layers to be used. Default is 2.
    dense_units (list): The number of units for each dense layer. Default is [512, 256].
    dropout_rate (float): The dropout rate to be used in the dense layers. Default is 0.2.
    num_output_units (int): The number of output units for the softmax classification layer. Default is 1837.
    use_batchnorm (bool): Whether to use BatchNormalization layers. Default is True.
    activation (str): The activation function to use for all layers except the output layer. Default is 'relu'.

    Returns:
    A Keras model for predicting the next move in a chess game.
    '''

    # Infer the number of residual blocks from the length of residual_filters
    num_residual_blocks = len(residual_filters)

    # Define input layer
    inputs = Input(shape=input_shape)

    # Define convolutional layers
    x = inputs
    for i in range(len(conv_filters)):
        x = Conv2D(filters=conv_filters[i], kernel_size=(
            8, 8), activation=activation, padding='same')(x)

    # Define residual blocks with attention
    for i in range(num_residual_blocks):
        residual = x
        x = Conv2D(filters=residual_filters[i], kernel_size=(
            3, 3), activation=activation, padding='same')(x)
        x = Conv2D(filters=residual_filters[i], kernel_size=(
            3, 3), activation=activation, padding='same')(x)
        if use_attention:
            x = attention_block(residual, x, inter_channel,
                                activation=activation)
        else:
            residual = Conv2D(filters=residual_filters[i], kernel_size=(
                1, 1), padding='same')(residual)
        x = Add()([x, residual])
        x = Activation(activation)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)

    # Define dense layers
    x = GlobalAveragePooling2D()(x)
    for i in range(num_dense_layers):
        x = Dense(units=dense_units[i], activation=activation)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    # Define output layer
    outputs = Dense(units=num_output_units, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model
