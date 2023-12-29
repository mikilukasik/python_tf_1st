from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout
from keras.models import Model


def bottleneck_block(input_tensor, filters, kernel_size=3, stride=1, increase_dim=False):
    """Create a bottleneck block."""
    expansion_factor = 4
    reduced_filters = filters // expansion_factor

    # Contraction
    x = Conv2D(reduced_filters, (1, 1), strides=(
        stride if increase_dim else 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Processing
    x = Conv2D(reduced_filters, (kernel_size, kernel_size), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Expansion
    x = Conv2D(filters, (1, 1))(x)
    x = BatchNormalization()(x)

    # Shortcut connection
    shortcut = input_tensor
    if increase_dim or filters != input_tensor.shape[-1]:
        shortcut = Conv2D(filters, (1, 1), strides=(
            stride if increase_dim else 1))(input_tensor)
        shortcut = BatchNormalization()(shortcut)

    # Add shortcut to output
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def create_resnet(input_shape, num_classes, filters_sequence, dense_units=1024, dropout_rate=0.5):
    """Create a ResNet model with custom filters sequence and additional dense layers."""
    inputs = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(filters_sequence[0], (3, 3), strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Bottleneck Blocks with custom filter sequence
    for filters in filters_sequence:
        x = bottleneck_block(x, filters)

    # Flatten before Dense Layers
    x = Flatten()(x)

    # # Dense Layers with Dropout
    # x = Dense(dense_units, activation='relu')(x)
    # x = Dropout(dropout_rate)(x)
    # x = Dense(dense_units, activation='relu')(x)
    # x = Dropout(dropout_rate)(x)

    # Output Layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
