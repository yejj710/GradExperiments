from typing import cast
from keras.applications.resnet import ResNet50


def create_model(input_shape, num_classes, name='lenet', dataset='mnist'):
    from tensorflow import keras
    # from tensorflow.keras import layers
    from keras.api._v2.keras import layers
    def lenet_model():
        # Create model
        if dataset == "mnist":
            model = keras.Sequential(
                [
                    keras.Input(shape=input_shape),
                    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Flatten(),
                    layers.Dropout(0.3),
                    layers.Dense(num_classes, activation="softmax"),
                ]
            )
        elif dataset == 'cifar':
            model = keras.Sequential([
                # keras.Input(shape=input_shape),
                layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation="relu", input_shape=input_shape),
                layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation="relu",
                                    input_shape=(32, 32, 3)),
                layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
                layers.Flatten(),
                layers.Dropout(0.3),
                layers.Dense(units=120, activation="relu"),
                layers.Dense(units=84, activation="relu"),
                layers.Dense(units=num_classes, activation="softmax"),
            ])

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy"])
        
        return model

    def resnet_model():
        # raise ValueError('do not support resnet model')
        if dataset == "mnist":
            # model = resnet_50(28, 28, 1, 10)
            model = ResNet50(input_shape=(28, 28, 1), num_classes=10)
        elif dataset == "cifar":
            model = resnet_50(32, 32, 3, 10)
        
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=["accuracy"])
        return model

    if name == 'lenet':
        return lenet_model
    elif name == 'resnet':
        return resnet_model
    

def resnet_50(width, height, channel, classes):
    from keras.models import Model
    from keras.layers import Input, Dense, add, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate, \
        Flatten, ZeroPadding2D
    
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = BatchNormalization(axis=3, name=bn_name)(x)
        return x
    
    def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
        k1,k2,k3=nb_filters
        x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
        x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
        x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
        if with_conv_shortcut:
            shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
            x = add([x, shortcut])
            return x
        else:
            x = add([x, inpt])
            return x
    
    inpt = Input(shape=(width, height, channel))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    #conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model



def unused_tcp_port() -> int:
    """Return an unused port"""
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return cast(int, sock.getsockname()[1])