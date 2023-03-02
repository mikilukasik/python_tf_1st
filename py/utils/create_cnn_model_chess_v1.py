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


# It's difficult to say what the best model for learning chess would be, as it depends on the specific problem you are trying to solve. However, there are some general considerations you can keep in mind to build a model that is suitable for chess:

# Input representation: Chess is played on an 8x8 board with 12 unique piece types. One-hot encoding is a common input representation for this problem, where each square on the board is represented as a vector of length 12 (one for each piece type), with a value of 1 in the corresponding position if the square contains that piece and 0 otherwise.

# Model architecture: Convolutional neural networks (CNNs) are a popular choice for image-based problems like chess, as they can effectively learn features from the board representation. In addition, recurrent neural networks (RNNs) can be used to model the temporal aspect of chess, as each move is dependent on the previous state of the board.

# Output representation: The output of the model should be a probability distribution over possible moves, rather than a classification into a fixed number of classes.

# Here's an example of how you could modify the create_cnn_model function to incorporate these considerations:

# python
# Copy code
# def create_chess_model(input_shape=(8, 8, 12), use_attention=False):
#     """
#     A function that creates a convolutional neural network (CNN) model for chess move prediction.

#     Args:
#         input_shape: Tuple representing the shape of the input image.
#         use_attention: Whether to use the attention block in the model.

#     Returns:
#         The compiled model.
#     """
#     inputs = Input(shape=input_shape)
#     x = inputs

#     # Convolutional layers
#     x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
#     x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
#     x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
#     x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

#     # Residual blocks
#     x = residual_block(x, 256)
#     x = residual_block(x, 256)
#     if use_attention:
#         x = attention_block(x)

#     # Flatten the output of the final Conv2D layer
#     x = Flatten()(x)

#     # Dense layers with 1024 and 512 units and ELU activation
#     x = Dense(1024, activation='elu')(x)
#     x = Dropout(0.5)(x)
#     x = Dense(512, activation='elu')(x)
#     x = Dropout(0.5)(x)

#     # Output layer with a softmax activation
#     outputs = Dense(8*8*73, activation='softmax')(x)

#     # Reshape output to (8, 8, 73) to represent move probabilities for each square and piece
#     outputs = Reshape((8, 8, 73))(outputs)

#     # Create the model
#     model = Model(inputs=inputs, outputs=outputs)

#     # Compile the model with Adam optimizer and mean squared error loss
#     optimizer = Adam(lr=0.001)
#     model.compile(loss='mse', optimizer=optimizer)

#     return model


# In this modified function, we've changed the input shape to be compatible with one-hot encoding of the chess board. We've also added additional convolutional and residual blocks to allow the model to learn more complex features from the board representation. The output layer has been changed to produce a probability distribution over all possible moves, which


# Miki
# explain in detail why this model is good for chess
# Unfortunately, as I don't have any information on the specific model that you are referring to, I cannot provide you with a detailed explanation of why it would be good for chess.

# However, in general, a good model for chess would need to be able to take in the current state of the chess board as input and output a probability distribution over the possible moves that could be made. The model would need to be able to learn the rules of chess and be able to accurately predict which moves are likely to lead to a good outcome.

# In addition to accuracy, there are other important factors to consider when designing a model for chess. The model should be efficient, able to make predictions quickly so that it can be used in real-time applications. The model should also be able to generalize well to new positions, so that it can be applied in a wide range of scenarios.

# There are many different approaches that can be taken when designing a model for chess, and the best approach may depend on the specific problem that you are trying to solve. Some popular approaches include neural networks, decision trees, and reinforcement learning algorithms.
