import sys
import secretflow as sf
sys.path.append("/home/yejj/GradExperiments/chapter3")
from utils.model import create_model
from utils.data import load_cifar_data
from secretflow.ml.nn import FLModel
from secretflow.ml.nn.callbacks.early_stopping import EarlyStoppingEpoch
import numpy as np
# from copy_ndarray import create_ndarray
from secretflow.device import reveal
from secretflow.utils.simulation.datasets import load_mnist
from sklearn.metrics import f1_score, accuracy_score
import pickle
import os



def _init_aggreator(type, **kwargs):
    from secretflow.security.aggregation import PlainAggregator, SPUAggregator
    from secretflow.security.aggregation.experiment import LDPAggregator
    from secretflow.security.aggregation.experiment.ppb_aggregator import PPBAggregator
    if type == "plain":
        agg = PlainAggregator(kwargs.get("agg_server"))
    elif type == "ldp":
        agg = LDPAggregator(kwargs.get("agg_server"))
    elif type == "phe":
        # (agg_server, [devices[i] for i in range(10)])
        devices = kwargs.get("devices")
        # todo device name 随机应变
        agg = PPBAggregator(kwargs.get("agg_server"), [devices[i] for i in range(10)])
    elif type == "ss":
        import spu
        DEFAULT_SEMI2K_RUNTIME_CONFIG = {
            'protocol': spu.spu_pb2.SEMI2K,
            'field': spu.spu_pb2.FM64,
        }

        cdef = sf.utils.testing.cluster_def([f"party{i}" for i in range(10)], DEFAULT_SEMI2K_RUNTIME_CONFIG)
        spu_device = sf.SPU(cdef)

        agg = SPUAggregator(spu_device)
    else:
        raise ValueError(f"current {type} does not support")
    return agg


def _train_steps(type, devices, server, x_train, y_train, x_test, y_test, epochs=10, callback=None):
    # num_classes = 10
    input_shape = (32, 32, 3)
    model = create_model(input_shape, 10, "lenet", "cifar")

    aggregator = _init_aggreator(type=type, agg_server=server, devices=devices)

    fed_model = FLModel(server=server,
                        device_list=[devices[i] for i in range(10)],
                        model=model,
                        aggregator=aggregator,
                        strategy="fed_avg_w",
                        backend = "tensorflow")
    
    history = fed_model.fit(x_train,
                y_train,
                validation_data=(x_test, y_test),
                epochs=epochs,
                sampler_method="batch",
                batch_size=128,
                aggregate_freq=1,
                callbacks=callback)

    pred_fed_y = fed_model.predict(x_test, batch_size=128)

    original_y = np.array([])
    pred_y = np.array([])

    for device in [devices[i] for i in range(10)]:
        temp = y_test.partitions[device]
        original_y = np.concatenate((original_y, np.argmax(np.array(reveal(temp)), axis=1)))

        temp1 = pred_fed_y[device]
        pred_y = np.concatenate((pred_y, np.argmax(np.array(reveal(temp1)), axis=1)))
    
    f1_macro = f1_score(original_y, pred_y, average='macro')
    accuracy = accuracy_score(original_y, pred_y)

    # 6. dump evaluate metrics
    dump_result = {"history":history, 'f1':f1_macro, 'acc':accuracy}
    # print(dump_result)
    save_directory = os.getcwd()

    file_path = os.path.join(save_directory, f'../evaluate_metrics/cifar_{type}_225.pkl')

    with open(file_path, 'wb') as file:
        pickle.dump(dump_result, file)
    

if __name__ == "__main__":
    # 1. init env
    sf.shutdown()

    nodes = [f"party{i}" for i in range(10)]
    nodes.append('server')

    sf.init(nodes, address="local")  # init nodes
    
    devices = {}    # init PYU device
    for i in range(10):
        devices[i] = sf.PYU(f"party{i}")
    agg_server = sf.PYU("server")

    # 2. load data & global config
    # (x_train, y_train), (x_test, y_test) = load_mnist(parts=[devices[i] for i in range(10)], normalized_x=True, categorical_y=True)
    (x_train, y_train), (x_test, y_test) = load_cifar_data([devices[i] for i in range(10)])

    earlystop_callback = EarlyStoppingEpoch(
		monitor='val_loss', min_delta=0.005, mode='min',
        patience=10)

    # 3. train and evaluate model steps
    # ['plain', 'ldp', 'phe', 'ss']
    for agg_type in ['plain', 'ldp', 'phe', 'ss']:
        _train_steps(type=agg_type,
                     devices=devices,
                     server=agg_server,
                     x_train=x_train,
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test,
                     epochs=25,
                     callback=earlystop_callback)

    