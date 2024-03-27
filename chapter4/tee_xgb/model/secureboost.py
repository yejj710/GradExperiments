import numpy as np
import sys
import time
from sklearn.metrics import roc_auc_score

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ml.boost.sgb_v import (
    Sgb,
    get_classic_XGB_params,
    get_classic_lightGBM_params,
)
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from secretflow.ml.boost.sgb_v.model import load_model
from conf import heu_config, cluster_def

# refer to https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.3.0b0/tutorial/SecureBoost

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
    le = LabelEncoder()
    if need_label:
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
    alice_ip = '127.0.0.1'
    bob_ip = '127.0.0.1'
    ip_party_map = {bob_ip: 'bob', alice_ip: 'alice'}

    _system_config = {'lineage_pinning_enabled': False}
    sf.shutdown()
    # init cluster
    sf.init(
        ['alice', 'bob'],
        address='local',
        _system_config=_system_config,
        object_store_memory=5 * 1024 * 1024 * 1024,
    )

    alice = sf.PYU('alice')
    bob = sf.PYU('bob')
    heu = sf.HEU(heu_config, cluster_def['runtime_config']['field'])

    v_train_data, v_test_data, v_train_label, v_test_label = load_fed_weather()

    # v_train_data, v_test_data, v_train_label, v_test_label = load_fed_a9a()

    # wait IO finished
    wait([p.data for p in v_train_data.partitions.values()])
    wait([p.data for p in v_test_data.partitions.values()])
    wait([p.data for p in v_train_label.partitions.values()])
    wait([p.data for p in v_test_label.partitions.values()])

    params = {
        # for more detail, see Xgb API doc
        'num_boost_round': 5,
        'max_depth': 5,
        'learning_rate': 0.2,
        'objective': 'logistic',
        'reg_lambda': 0.1,
        'subsample': 1,
        'colsample_by_tree': 1,
    }
    # params = get_classic_XGB_params()
    # params['num_boost_round'] = 3
    # params['max_depth'] = 5
    # print(params)

    sgb = Sgb(heu)
    start = time.time()
    model = sgb.train(params, v_train_data, v_train_label)
    print(f"train time: {time.time() - start}")

    heu_yhat = model.predict(v_test_data)

    yhat = reveal(heu_yhat)
    y = reveal(v_test_label.partitions[alice])
    # get the area under curve(auc) score of classification
    print(f"auc: {roc_auc_score(y, yhat)}")
    binary_class_results = np.where(yhat<=0.5, 0, 1)
    # get the accuracy score of classification
    # print(f"acc: {accuracy_score(y, binary_class_results)}")
    print(f"f1: {f1_score(y, binary_class_results)}")
    
    # get the report of classification
    print("classification report:")
    print(classification_report(y, binary_class_results))
    