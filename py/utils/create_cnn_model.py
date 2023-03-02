from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import BatchNormalization, Activation
from keras.layers import Add
from keras.layers import GlobalAveragePooling2D

from createFragments.residual_block import residual_block
from createFragments.attention_block import attention_block


def create_cnn_model(input_shape=(8, 8, 14), num_classes=1837, use_attention=False, use_batchnorm=True):
    """
    A function that creates a convolutional neural network (CNN) model for image classification.

    Args:
        input_shape: Tuple representing the shape of the input image.
        num_classes: Number of classes to classify the image into.
        use_attention: Whether to use the attention block in the model.
        use_batchnorm: Whether to use batch normalization in the model.

    Returns:
        The compiled model.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    # Two Conv2D layers with 64 filters each and ReLU activation
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)

    # Residual blocks with two Conv2D layers with 128 filters each and ReLU activation
    x = residual_block(x, 128, use_batchnorm)
    x = residual_block(x, 128, use_batchnorm)

    # Residual blocks with two Conv2D layers with 256 filters each and ReLU activation
    x = residual_block(x, 256, use_batchnorm)
    x = residual_block(x, 256, use_batchnorm)
    if use_attention:
        x = attention_block(x)

    # Residual blocks with two Conv2D layers with 512 filters each and ReLU activation
    x = residual_block(x, 512, use_batchnorm)
    x = residual_block(x, 512, use_batchnorm)
    if use_attention:
        x = attention_block(x)

    # MaxPooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output of the final Conv2D layer
    x = GlobalAveragePooling2D()(x)

    # Dense layers with 1024 and 512 units and ELU activation
    x = Dense(1024, activation='elu')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='elu')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Output layer with a softmax activation
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with Adam optimizer
    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model
