import numpy as np 
import pandas as pd
import pickle
import csv
import os
from sklearn.model_selection import train_test_split


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
        import random
        sample_num = int(random_sample * len(y))

        sample_list = [i for i in range(len(y))]
        sample_list = random.sample(sample_list, sample_num) 

        x = x[sample_list,:] 
        y = y[sample_list] 
        print(f'finish random sampling, current feature shape is {x.shape}')
    return x, y


def _read_a9a():

    test_path = "/home/yejj/GradExperiments/chapter4/tee_xgb/a9a_data/a9a.t"
    train_path = "/home/yejj/GradExperiments/chapter4/tee_xgb/a9a_data/a9a"
    
    x_test, y_test = read_libsvm(test_path, 123)
    print(f'finish read test data, shape of test_x is {x_test.shape}')

    x_train, y_train = read_libsvm(train_path, 123)
    print(f'finish read train data, shape of train_x is {x_train.shape}')

    np.save('./a9a_data/x_test.npy', x_test)
    np.save('./a9a_data/y_test.npy', y_test)

    np.save('./a9a_data/x_train.npy', x_train)
    np.save('./a9a_data/y_train.npy', y_train)


def load_a9a():
    x_train = np.load("../a9a_data/x_train.npy")
    x_test = np.load("../a9a_data/x_test.npy")
    y_train = np.load("../a9a_data/y_train.npy")
    y_test = np.load("../a9a_data/y_test.npy")
    return x_train, y_train, x_test, y_test 


def load_weather_aus(first_load=True):
    if first_load:
        data = pd.read_csv('weatherAUS.csv')
        # 删除标签列中包含Nan的样本    
        data = data.dropna(subset=['RainTomorrow'])

        data = data.fillna(-1)
        numerical_features = [x for x in data.columns if data[x].dtype == np.float]
        category_features = [x for x in data.columns if data[x].dtype != np.float and x != 'RainTomorrow']
        # print(category_features)
        # 离散变量编码
        def get_mapfunction(x):
            mapp = dict(zip(x.unique().tolist(), range(len(x.unique().tolist()))))
            def mapfunction(y):
                if y in mapp:
                    return mapp[y]
                else:
                    return -1
            return mapfunction
        for i in category_features:
            data[i] = data[i].apply(get_mapfunction(data[i]))

        data_target_part = data['RainTomorrow']
        data_features_part = data[[x for x in data.columns if x != 'RainTomorrow']]

        ## 测试集大小为20%， 80%/20%分
        x_train, x_test, y_train, y_test = train_test_split(data_features_part, data_target_part, test_size = 0.2, random_state = 314)

        y_train = y_train.replace({'Yes': 1, 'No': 0})
        y_test  = y_test.replace({'Yes': 1, 'No': 0})

        save_directory = os.getcwd()
        file_path = os.path.join(save_directory, f'./weather_aus.pkl')

        with open(file_path, 'wb') as file:
            pickle.dump((x_train, x_test, y_train, y_test), file)
    else:
        save_directory = os.getcwd()
        file_path = os.path.join(save_directory, f'../weather_aus.pkl')
        with open(file_path, 'rb') as file:
           tmp = pickle.load(file)
           x_train, x_test, y_train, y_test = tmp[0], tmp[1], tmp[2], tmp[3]
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    # data = pd.read_csv('weatherAUS.csv')

    x_train, x_test, y_train, y_test = load_weather_aus(False)
    print(x_train.info)
    print(x_test.info)
    # print(y_train[0:10])