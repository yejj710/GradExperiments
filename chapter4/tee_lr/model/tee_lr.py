# occlum refer to https://occlum.readthedocs.io/en/latest/quickstart.html#


import numpy as np
import secretflow as sf
from secretflow.device import TEEU


cluster_config = {
    'parties': {
        'alice': {
            'address': '192.168.88.66:20001',
            'listen_address': '0.0.0.0:20001'
    },
        'bob': {
            'address': '192.168.88.66:20002',
            'listen_address': '0.0.0.0:20002'
        }
    },
    'self_party': 'Alice'
}


auth_manager_config = {
    'host': 'host of AuthManager',
    'ca_cert': 'path_of_ca_certificate_of_AuthManager',
    'mr_enclave': 'mrenclave of AuthManager',
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(W, b, inputs):
    return sigmoid(np.dot(inputs, W) + b)


def loss(W, b, inputs, targets):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -np.mean(np.log(label_probs))


def average(data):
    return np.average(data, axis=1)


a = alice(lambda: np.random.rand(4, 3))()
b = bob(lambda: np.random.rand(4, 3))()

# Transfer data to teeu.
a_teeu = a.to(teeu, allow_funcs=average)
b_teeu = b.to(teeu, allow_funcs=average)

# TEEU runs average.
avg_val = teeu(average)([a_teeu, b_teeu])
print(sf.reveal(avg_val))

a = sf.reveal(a)
b = sf.reveal(b)
np.testing.assert_equal(avg_val, average([a, b]) )


def alice_part():
    # Alice starts a local ray inside tee.
    global cluster_config, auth_manager_config

    sf.init(
        address='local', 
        cluster_config=cluster_config, 
        auth_manager_config=auth_manager_config,
        _temp_dir="/host/tmp/ray",
        _plasma_directory="/tmp",
    )
    alice = TEEU('alice', mr_enclave='')
    bob = TEEU('bob', mr_enclave='') # add bob mr_enclave

    # allow funcs


def bob_part():
    pass

if __name__ == "__main__":
    pass