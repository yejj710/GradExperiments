from typing import cast


def create_model(input_shape, num_classes, name='lenet'):
    from tensorflow import keras
    # from tensorflow.keras import layers
    from keras.api._v2.keras import layers
    def lenet_model():
        # Create model
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