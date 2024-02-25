import os
import pickle
import numpy as np


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


def load_cifar_data():
    """Loads the CIFAR10 dataset.

    This is a dataset of 50,000 32x32 color training images and 10,000 test
    images, labeled over 10 categories. See more info at the
    [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).
    Example:

    ```python
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    ```
    """

    x_train, y_train, x_test, y_test = get_CIFAR10_dataset()

    # y_train = np.reshape(y_train, (len(y_train), 1))
    # y_test = np.reshape(y_test, (len(y_test), 1))
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.fit_transform(y_test.reshape(-1, 1))

    # if backend.image_data_format() == "channels_last":
    #     x_train = x_train.transpose(0, 2, 3, 1)
    #     x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)
