from keras.layers import Conv2D, Activation, Add, Multiply


def attention_block(x, g, inter_channel):
    # Define theta_x (query)
    theta_x = Conv2D(inter_channel, kernel_size=1,
                     strides=1, padding='same')(x)

    # Define phi_g (key)
    phi_g = Conv2D(inter_channel, kernel_size=1, strides=1, padding='same')(g)

    # Define f (attention map)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    f = Conv2D(1, kernel_size=1, strides=1, padding='same')(f)
    f = Activation('sigmoid')(f)

    # Define h (weighted feature map)
    h = Multiply()([x, f])

    # Define y (output)
    y = Add()([h, x])

    return y
