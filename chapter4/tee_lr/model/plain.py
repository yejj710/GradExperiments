import secretflow as sf
import sys
import time
import math
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import spu
from numpy.random import RandomState
from secretflow.data.ndarray import FedNdarray
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import proxy
from secretflow.device.device.pyu import PYU, PYUObject
from secretflow.device.driver import reveal
from secretflow.security.aggregation.aggregator import Aggregator
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def breast_cancer(party_id=None, train: bool = True) -> (np.ndarray, np.ndarray):
    sys.path.append("/home/yejj/GradExperiments/chapter4/tee_lr")
    from utils.data_loader import _load_breast_cancer
    x_train, y_train, x_test, y_test = _load_breast_cancer(315)
    if train:
        if party_id:
            if party_id == 1:
                return x_train[:, :15], y_train
            else:
                return x_train[:, 15:], np.ndarray([])
        else:
            return x_train, y_train
    else:
        return x_test, y_test


def _epsilon(party_id=None, train: bool = True) -> (np.ndarray, np.ndarray):
    x_train = np.load("../epsilon/x_train.npy")
    x_test = np.load("../epsilon/x_test.npy")
    y_train = np.load("../epsilon/y_train.npy")
    y_test = np.load("../epsilon/y_test.npy")
    # x_train shape (40000, 500)
    y_train = list(map(lambda x: 0 if x == -1 else 1, y_train))
    y_test = list(map(lambda x: 0 if x == -1 else 1, y_test))

    if train:
        if party_id == 1:
            return x_train[:, :100], np.array(y_train)
        elif party_id == 2:
            return x_train[:, 100:], np.ndarray([])
        else:
            raise ValueError('party_id params only support 1 and 2')
    else:
        return x_test, np.array(y_test)


def _init_weight(feature_num, party_id=None) -> (np.ndarray, np.float64):
    if party_id == 1:
        return np.zeros((feature_num,)), 0.0
    else:
        return np.zeros((feature_num,)), 0.0


def compute_wx(weight, inputs, party_id=None):
    if party_id == 1:
        w, b = weight[0], weight[1]
        wx = np.dot(inputs, w) + b
    else:
        w = weight
        wx = np.dot(inputs, w)
    return wx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(W, b, inputs):
    return sigmoid(np.dot(inputs, W) + b)


def merge_wx(wx1, wx2):
    val1 = reveal(wx1)
    val2 = reveal(wx2)
    return sigmoid(val1+val2)


def loss(target, pred):
    label_probs = pred * target + (1 - pred) * (1 - target)
    return -np.mean(np.log(label_probs))


def gradient_descent(weights, x, a, label, learning_rate, party_id=None)-> (np.ndarray, np.float64):
    if party_id == 1:
        # print('label len', len(label))
        w, b = weights[0], weights[1]
        # print(np.dot(reveal(x).T, (a-label)))
        new_w = w - learning_rate*(1/len(label))*np.dot(reveal(x).T, (a-label))
        b = b - learning_rate*np.mean(a-label)
        return new_w, b
    else:
        w = weights
        new_w = w - learning_rate*(1/len(label))*np.dot(reveal(x).T, (a-label))
        return new_w, 0.0
    

def fit():
    pass

def validate_model(W, b, X_test, y_test):
    y_pred_score = predict(W, b, X_test)
    y_pred = list(map(lambda x: 0 if x <0.5 else 1, y_pred_score))
    # print(y_pred)
    return roc_auc_score(y_test, y_pred_score), f1_score(y_test, y_pred, average='binary')


if __name__ == "__main__":
    # sf.shutdown()
    sf.init(['alice','bob'], address="local")
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')

    x1, y = alice(breast_cancer)(party_id=1)
    x2, _ = bob(breast_cancer)(party_id=2)

    # print(reveal(x1).shape[1], reveal(x2).shape[1])
    X_test, y_test = breast_cancer(train=False)


    w1, bias = alice(_init_weight)(reveal(x1).shape[1], party_id=1)
    w2, _ = bob(_init_weight)(reveal(x2).shape[1], party_id=2)
    epochs = 25
    start_time = time.time()
    for i in range(epochs):

        wx1 = alice(compute_wx)((w1, bias), x1, party_id=1)
        wx2 = bob(compute_wx)(w2, x2, party_id=2)

        pred_score = merge_wx(wx1, wx2)

        # update weight
        w1, bias = alice(gradient_descent)(w1, reveal(x1), pred_score, reveal(y), 0.01, party_id=1)
        w2, _ = bob(gradient_descent)(w2, reveal(x2), pred_score, reveal(y), 0.01, party_id=2)

        # print("new w1", reveal(w1))
        # print("new bias", reveal(bias))
    
    W_ = np.concatenate([reveal(w1), reveal(w2)], axis=0)
    # .extend(list(reveal(w2)))
    # print(reveal(bias))
    print(W_)
    auc, f1 = validate_model(np.array(W_), reveal(bias), X_test, y_test)
    print(f"train time :{time.time()-start_time}")
    print(auc, f1)
    sf.shutdown()
