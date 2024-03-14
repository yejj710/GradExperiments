from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import numpy as np
import csv


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


def read_libsvm(input_file, max_feature, max_sample=None, random_sample=None, choose_feature=None, **kwargs):
    """
    读取libsvm格式的数据集
    :param input_file: 数据集文件名
    :param max_feature: 数据集最大维
    :return: SVM的训练参数x,y
    """
    count = 0 #样本个数
    y = []
    x = []
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=" ")
        # 每一行
        for line in reader:
            # print(line)
            if(line[-1] == ''):
                line.pop(-1)#去掉末尾""空字符
            # 获取y
            y_line = line.pop(0)
            y.append(int(y_line))
            temp = 0
            temp_x = []
            # 获取x
            try:
                for str in line:
                    index = int(str.split(":")[0])
                    value = float(str.split(":")[1])
 
                    # 没列出的列全部为0
                    for j in range(temp, index):
                        if j < (index-1):
                            temp_x.append(0)
                        else:
                            temp_x.append(value)
                            temp=index
                            break
                x.append(temp_x)
                count += 1
                if count % 1000 == 0:
                    print(f'finish read {count} lines.')
            except KeyError:
                continue
 
    # 补全每一行最后一维到最大维度的0值
    for i in range(count):
        for j in range(len(x[i]), max_feature):
            x[i].append(0)
 
    y = np.array(y)
    x = np.array(x).reshape(count, max_feature)

    # 随机采样
    if random_sample:
        sample_num = int(random_sample * len(y))

        sample_list = [i for i in range(len(y))]
        sample_list = random.sample(sample_list, sample_num) 

        x = x[sample_list,:] 
        y = y[sample_list] 
    print(f'finish random sampling, current feature shape is {x.shape}')
    return x, y


def transform_libsvm(inputfile, max_feature):
    """
    将libsvm格式转换成csv格式, 保留原数据集
    :param inputfile: libsvm格式的数据集
    :param max_feature: 要转换的数据集的最大维
    :return: 输出csv格式的数据集
    """
    path = "../input/"
    x, y = read_libsvm(path+inputfile, max_feature)
    rows = x.shape[0]
    columns = x.shape[1]
    file=open("../output/"+inputfile+".csv", "w+")
    for r in range(rows):
        r_line = ""
        for c in range(columns):
            temp=str(x[r, c]) + ","
            r_line += temp
        r_lin += str(y[r])
        file.write(r_line + "\n")
    file.close()


def _read_epsilon():

    test_path = "/home/yejj/GradExperiments/chapter4/tee_lr/epsilon/epsilon_normalized.t"
    train_path = "/home/yejj/GradExperiments/chapter4/tee_lr/epsilon/epsilon_normalized"

    # from catboost.datasets import epsilon
    # epsilon_train, epsilon_test = epsilon()  catboost

    need_col = np.load('./epsilon/col.npy')
    

    # x_test, y_test = read_libsvm(test_path, 2000, random_sample=0.1)
    # # 取500列
    # x_test = x_test[:,need_col]
    # print(f'finish feature sampling, current feature shape is {x_test.shape}')
    x_train, y_train = read_libsvm(train_path, 2000, random_sample=0.1)
    x_train = x_train[:,need_col]
    print(f'finish feature sampling, current feature shape is {x_train.shape}')

    # np.save('./epsilon/x_test.npy', x_test)
    # np.save('./epsilon/y_test.npy', y_test)

    np.save('./epsilon/x_train.npy', x_train)
    np.save('./epsilon/y_train.npy', y_train)


def load_epsilon():
    x_train = np.load("./epsilon/x_train.npy")
    x_test = np.load("./epsilon/x_test.npy")
    y_train = np.load("./epsilon/y_train.npy")
    y_test = np.load("./epsilon/y_test.npy")
    return x_train, y_train, x_test, y_test 

if __name__ == "__main__":
    # from func import generate_unique_random_numbers
    # need_col = generate_unique_random_numbers(500)
    # np.save("./epsilon/col.npy", np.array(need_col))

    # _load_epsilon()
    x_train = np.load("./epsilon/x_train.npy")
    x_test = np.load("./epsilon/x_test.npy")
    print(x_train.shape, x_test.shape)
    print(x_train[0])

    # num_nan = np.sum(np.isnan(x_train))
    # num_nan += np.sum(np.isnan(x_test))

    # num_zeros = np.sum(x_train  == 0)
    # num_zeros += np.sum(x_test  == 0)
    # print(num_nan, num_zeros)
    # print(X.shape)

    # print(y, type(y), y.shape)
    # random_sample = 0.25
    # data = np.array([[ 0,  1,  2,  3, 6],
    #              [ 4,  5,  6,  7, 6],
    #              [ 8,  9, 10, 11, 6],
    #              [12, 13, 14, 0, 6]])
    # print(data, data.shape)
    # num_nan = np.sum(np.isnan(data))
    # data = data[:,[0,3]]
    # print(data, data.shape)