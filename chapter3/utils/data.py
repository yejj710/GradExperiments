import os
import pickle
import numpy as np
from secretflow.utils.simulation.data.ndarray import create_ndarray


def get_cifar10_data_and_label(type='train', root="/home/yejj/tmp/cifar-10-batches-py"):
    def load_file(filename):
        with open(filename, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data
    
    if type == 'train':
        data_batch_1 = load_file(os.path.join(root, 'data_batch_1'))
        data_batch_2 = load_file(os.path.join(root, 'data_batch_2'))
        data_batch_3 = load_file(os.path.join(root, 'data_batch_3'))
        data_batch_4 = load_file(os.path.join(root, 'data_batch_4'))
        data_batch_5 = load_file(os.path.join(root, 'data_batch_5'))
        original_data = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]
    else:
        data_batch_1 = load_file(os.path.join(root, 'test_batch'))
        original_data = [data_batch_1]
    dataset = []
    labelset = []
    for data in original_data:
        img_data = (data["data"])
        img_label = (data["labels"])
        dataset.append(img_data)
        labelset.append(img_label)
    dataset = np.concatenate(dataset)
    labelset = np.concatenate(labelset)
    return dataset, labelset


def get_CIFAR10_dataset():
    train_dataset, label_dataset = get_cifar10_data_and_label()
    test_dataset, test_label_dataset = get_cifar10_data_and_label(type='test')
    return train_dataset, label_dataset, test_dataset, test_label_dataset


def load_cifar_data(parts):
    """Loads the CIFAR10 dataset.

    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    ```
    """
    from keras.datasets.cifar import load_batch

    x_train = np.empty((50000, 3, 32, 32), dtype="uint8")
    y_train = np.empty((50000,), dtype="uint8")
    # x_train, y_train, x_test, y_test = get_CIFAR10_dataset()
    path = "/home/yejj/tmp/cifar-10-batches-py"
    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        (
            x_train[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse=False)


    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)    

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    # x_test = x_test.astype(x_train.dtype)
    # y_test = y_test.astype(y_train.dtype)
    return (
        (
            create_ndarray(x_train, parts=parts, axis=0, is_torch=False),
            create_ndarray(y_train, parts=parts, axis=0),
        ),
        (
            create_ndarray(x_test, parts=parts, axis=0, is_torch=False),
            create_ndarray(y_test, parts=parts, axis=0),
        ),
    )
