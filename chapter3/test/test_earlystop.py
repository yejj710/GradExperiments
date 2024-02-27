import sys

import secretflow as sf

sys.path.append("/home/yejj/GradExperiments/chapter3")
from utils.model import create_model

from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import PlainAggregator

sf.shutdown()
#init 10 nodes and 1 server
# nodes = [f"party{i}" for i in range(10)]
# nodes.append('server')

# init nodes
# sf.init(nodes, address="local")
sf.init(['alice', 'bob','server'], address="local")
# init PYU device
# devices = {}
# for i in range(10):
#     devices[i] = sf.PYU(f"party{i}")
# agg_server = sf.PYU("server")
alice, bob, agg_server = sf.PYU("alice"), sf.PYU("bob"), sf.PYU("server")


from secretflow.utils.simulation.datasets import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(parts=[alice, bob], normalized_x=True, categorical_y=True)
print(x_train.partition_shape())

num_classes = 10
input_shape = (28, 28, 1)
model = create_model(input_shape, num_classes, "lenet")

plain_aggregator = PlainAggregator(agg_server)

fed_plain_model = FLModel(server=agg_server,
                    device_list=[alice, bob],
                    model=model,
                    aggregator=plain_aggregator,
                    strategy="fed_avg_w",
                    backend = "tensorflow")

from secretflow.ml.nn.callbacks.early_stopping import EarlyStoppingEpoch

earlystop_callback = EarlyStoppingEpoch(
		monitor='val_loss', min_delta=0.005, mode='min',
        patience=1)

import time

start_time = time.time()
history = fed_plain_model.fit(x_train,
                    y_train,
                    validation_data=(x_test, y_test),
                    epochs=1,
                    sampler_method="batch",
                    batch_size=128,
                    aggregate_freq=1,
                    callbacks=earlystop_callback)
plain_time = time.time() - start_time
# print("cost time : ", time.time()-start_time)
print(history)

pred_y = fed_plain_model.predict(x_test, batch_size=128)
print(pred_y)
