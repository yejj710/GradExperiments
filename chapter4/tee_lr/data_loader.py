from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def _load_breast_cancer():
    X, y = load_breast_cancer(return_X_y=True)
    # 归一化
    mm = MinMaxScaler()
    X = mm.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return x_train, y_train, x_test, y_test


def _load_numeric_only_dataset(path, row_count, column_count, sep='\t'):
    # - can't use `pandas.read_csv` because it may result in 5x overhead
    # - can't use `numpy.loadtxt` because it may result in 3x overhead
    # And both mentioned above solutions are very slow compared to the one implemented below.
    dataset = np.zeros((row_count, column_count, ), dtype=np.float32, order='F')
    with open(path, 'rb') as f:
        for line_idx, line in enumerate(f):
            # `str.split()` is too slow, use `numpy.fromstring()`
            print(line)
            row = np.fromstring(line, dtype=np.float32, sep=sep)
            assert row.size == column_count, 'got too many columns at line %d (expected %d columns, got %d)' % (line_idx + 1, column_count, row.size)
            # doing `dataset[line_idx][:]` instead of `dataset[line_idx]` is here on purpose,
            # otherwise we may reallocate memory, while here we just copy
            dataset[line_idx][:] = row

    assert line_idx + 1 == row_count, 'got too many lines (expected %d lines, got %d)' % (row_count, line_idx + 1)

    return pd.DataFrame(dataset)


def _load_epsilon():
    from catboost.datasets import epsilon
    test_path = "/home/yejj/GradExperiments/chapter4/tee_lr/epsilon_normalized.t"
    return _load_numeric_only_dataset(test_path, 1000, 2001, sep='\t')

    epsilon_train, epsilon_test = epsilon()
    return epsilon_test



if __name__ == "__main__":
    # X, y, x_test, y_tset = _load_breast_cancer()
    X = _load_epsilon()
    # print(X)

    num_nan = np.sum(np.isnan(X))
    num_zeros = np.sum(X == 0)
    print(num_nan, num_zeros)
    print(X.shape)
    # print(y, type(y), y.shape)