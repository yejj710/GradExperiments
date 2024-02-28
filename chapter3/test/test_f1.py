import sys
import secretflow as sf
sys.path.append("/home/yejj/GradExperiments/chapter3")
from utils.model import create_model
from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import PlainAggregator
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from test.copy_ndarray import create_ndarray
from secretflow.device import reveal


if __name__ == "__main__":
    sf.shutdown()
    # sf.init(nodes, address="local")
    sf.init(['a', 'b'], address="local")
    # init PYU device
    alice, bob = sf.PYU("a"), sf.PYU("b")

    filepath = "/home/yejj/.secretflow/datasets/mnist.npz"
    with np.load(filepath) as f:
        yt = f['y_test']
    print(yt[0:10], yt[5000:5015])
    encoder = OneHotEncoder(sparse=False)
    yt = encoder.fit_transform(yt.reshape(-1, 1))
    # print(yt[0:5])
    h_arr = create_ndarray(yt, parts=[alice, bob], axis=0)
    # print(h_arr)
    result = np.array([])
    print("result", result)
    for device in [alice, bob]:
        temp = h_arr.partitions[device]
        # print(temp)
        y_pred = np.argmax(np.array(reveal(temp)[0:5]), axis=1)
        result = np.concatenate((result, y_pred))
        # print(type(y_pred))
        # print(y_pred)
        # print(reveal(temp)[0:5])
        # [7 2 1 0 4]  [3 9 9 8 4]
    print(result, type(result))
    from sklearn.metrics import precision_score, recall_score, f1_score
    y = [7,2,1,0,4,3,9,9,8,4]
    f1_micro = f1_score(y, result, average='micro')
    print(f1_micro)

    

