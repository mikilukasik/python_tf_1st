from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import BatchNormalization, Activation
from keras.layers import Add, GlobalAveragePooling2D

from createFragments.residual_block import residual_block
from createFragments.attention_block import attention_block


def create_chess_model(input_shape=(8, 8, 14), use_attention=False, use_batch_norm=False, use_global_avg_pool=False,
                       conv_layers=4, filters=[64, 64, 128, 128], residual_blocks=2, dense_layers=2, dense_units=[1024, 512],
                       loss='mse', activation='elu'):
    """
    A function that creates a convolutional neural network (CNN) model for chess move prediction.

    Args:
        input_shape: Tuple representing the shape of the input image.
        use_attention: Whether to use the attention block in the model.
        use_batch_norm: Whether to use batch normalization after each convolutional layer.
        use_global_avg_pool: Whether to use global average pooling before the final fully connected layer.
        conv_layers: Number of convolutional layers in the model.
        filters: List of integers representing the number of filters in each convolutional layer.
        residual_blocks: Number of residual blocks in the model.
        dense_layers: Number of dense layers in the model.
        dense_units: List of integers representing the number of units in each dense layer.
        loss: String representing the loss function to use in the model.
        activation: String representing the activation function to use in the model.

    Returns:
        The compiled model.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Convolutional layers
    for i in range(conv_layers):
        x = Conv2D(filters[i], (3, 3), padding='same')(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # Residual blocks
    for i in range(residual_blocks):
        x = residual_block(x, filters[-1], use_batch_norm=use_batch_norm)
    if use_attention:
        x = attention_block(x)

    # Global average pooling, if specified
    if use_global_avg_pool:
        x = GlobalAveragePooling2D()(x)
    else:
        # Flatten the output of the final Conv2D layer
        x = Flatten()(x)

    # Dense layers with specified number of units and activation function
    for i in range(dense_layers):
        x = Dense(dense_units[i])(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(0.5)(x)

    # Output layer with a softmax activation
    outputs = Dense(1837, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with Adam optimizer and specified loss function
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss=loss, optimizer=optimizer)

    return model


# This function, create_chess_model(), creates a convolutional neural network (CNN) model for chess move prediction. It takes in several parameters such as the shape of the input image, whether to use attention and batch normalization layers, and the number of convolutional layers, residual blocks, dense layers, and units.

# Here are some use examples of this function:

# python
# Copy code
# # Create a basic chess model with default settings
# model1 = create_chess_model()

# # Create a chess model with attention
# model2 = create_chess_model(use_attention=True)

# # Create a chess model with batch normalization
# model3 = create_chess_model(use_batch_norm=True)

# # Create a chess model with global average pooling
# model4 = create_chess_model(use_global_avg_pool=True)

# # Create a chess model with custom number of convolutional layers and filters
# model5 = create_chess_model(conv_layers=6, filters=[64, 64, 128, 128, 256, 256])

# # Create a chess model with custom number of residual blocks
# model6 = create_chess_model(residual_blocks=3)

# # Create a chess model with custom number of dense layers and units
# model7 = create_chess_model(dense_layers=3, dense_units=[1024, 512, 256])

# # Create a chess model with attention, batch normalization, and global average pooling
# model8 = create_chess_model(use_attention=True, use_batch_norm=True, use_global_avg_pool=True)
# In these examples, we create different versions of the chess model with various combinations of settings, such as using attention, batch normalization, and global average pooling layers, or customizing the number of layers and filters.
