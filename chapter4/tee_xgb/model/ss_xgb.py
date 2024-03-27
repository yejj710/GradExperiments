import sys
import time
import logging

import secretflow as sf
from secretflow.ml.boost.ss_xgb_v import Xgb
from secretflow.device.driver import wait, reveal
from secretflow.data import FedNdarray, PartitionWay
from secretflow.data.split import train_test_split
import numpy as np
sys.path.append("/home/yejj/GradExperiments/chapter4/tee_xgb")
from data_loader import load_a9a
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, classification_report

# refer to https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.3.0b0/user_guide/mpc_ml/decision_tree

# init log
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_weather(start, end, is_train=True, need_label=False):
    sys.path.append("/home/yejj/GradExperiments/chapter4/tee_xgb")
    from data_loader import load_weather_aus
    x_train, x_test, y_train, y_test = load_weather_aus(first_load=False)

    if need_label:
        if is_train:
            return y_train.to_numpy()
        else:
            return y_test.to_numpy()

    if is_train:
        
        x_train = x_train.to_numpy()
        return x_train[:, start:end]
    else:
        x_test = x_test.to_numpy()
        return x_test[:, start:end]
    

def load_a9a(start, end, is_train=True, need_label=False):
    sys.path.append("/home/yejj/GradExperiments/chapter4/tee_xgb")
    from data_loader import load_a9a
    x_train, y_train, x_test, y_test = load_a9a()

    if need_label:
        le = LabelEncoder()
        if is_train:
            y_train = le.fit_transform(y_train)
            return y_train
        else:
            y_test = le.fit_transform(y_test)
            return y_test

    if is_train:
        # x_train = x_train.to_numpy()
        return x_train[:, start:end]
    else:
        # x_test = x_test.to_numpy()
        return x_test[:, start:end]


def load_fed_a9a():
    v_train_data = FedNdarray(
        partitions={
            alice: alice(load_a9a)(0, 61, True, False),
            bob: bob(load_a9a)(61, 123, True, False),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    v_test_data = FedNdarray(
        partitions={
            alice: alice(load_a9a)(0, 61, False, False),
            bob: bob(load_a9a)(61, 123, False, False),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    v_train_label = FedNdarray(
        partitions={alice: alice(load_a9a)(1, 1, True, True)},
        partition_way=PartitionWay.VERTICAL,
    )
    v_test_label = FedNdarray(
        partitions={alice: alice(load_a9a)(1, 1, False, True)},
        partition_way=PartitionWay.VERTICAL,
    )
    return v_train_data, v_test_data, v_train_label, v_test_label


def load_fed_weather():
    v_train_data = FedNdarray(
        partitions={
            alice: alice(load_weather)(0, 11, True, False),
            bob: bob(load_weather)(11, 22, True, False),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    v_test_data = FedNdarray(
        partitions={
            alice: alice(load_weather)(0, 11, False, False),
            bob: bob(load_weather)(11, 22, False, False),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    v_train_label = FedNdarray(
        partitions={alice: alice(load_weather)(1, 1, True, True)},
        partition_way=PartitionWay.VERTICAL,
    )
    v_test_label = FedNdarray(
        partitions={alice: alice(load_weather)(1, 1, False, True)},
        partition_way=PartitionWay.VERTICAL,
    )
    return v_train_data, v_test_data, v_train_label, v_test_label


if __name__ == "__main__":
    sf.shutdown()
    # init all nodes in local Standalone Mode.
    sf.init(['alice', 'bob'], address='local')

    # init PYU, the Python Processing Unit, process plaintext in each node.
    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    # carol = sf.PYU('carol')

    # init SPU, the Secure Processing Unit,
    #           process ciphertext under the protection of a multi-party secure computing protocol
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    v_train_data, v_test_data, v_train_label, v_test_label = load_fed_weather()

    # v_train_data, v_test_data, v_train_label, v_test_label = load_fed_a9a()

    # wait IO finished
    wait([p.data for p in v_train_data.partitions.values()])
    wait([p.data for p in v_test_data.partitions.values()])
    wait([p.data for p in v_train_label.partitions.values()])
    wait([p.data for p in v_test_label.partitions.values()])

    # run SS-XGB
    xgb = Xgb(spu)
    start = time.time()
    params = {
        # for more detail, see Xgb API doc
        'num_boost_round': 20,
        'max_depth': 5,
        'learning_rate': 0.2,
        'objective': 'logistic',
        'reg_lambda': 0.1,
        'subsample': 1,
        'colsample_by_tree': 1,
    }
    model = xgb.train(params, v_train_data, v_train_label)
    logging.info(f"train time: {time.time() - start}")

    # Do predict
    start = time.time()
    # Now the result is saved in the spu by ciphertext
    spu_yhat = model.predict(v_test_data)
    # reveal for auc, acc and classification report test.
    yhat = reveal(spu_yhat)
    logging.info(f"predict time: {time.time() - start}")
    
    y = reveal(v_test_label.partitions[alice])
    # get the area under curve(auc) score of classification
    logging.info(f"auc: {roc_auc_score(y, yhat)}")
    binary_class_results = np.where(yhat<=0.5, 0, 1)
    # get the accuracy score of classification
    # logging.info(f"acc: {accuracy_score(y, binary_class_results)}")
    logging.info(f"f1: {f1_score(y, binary_class_results)}")
    
    # get the report of classification
    print("classification report:")
    print(classification_report(y, binary_class_results))