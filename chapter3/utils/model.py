from typing import cast


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
        pass

    if name == 'lenet':
        return lenet_model
    elif name == 'resnet':
        return resnet_model
    

def unused_tcp_port() -> int:
    """Return an unused port"""
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return cast(int, sock.getsockname()[1])