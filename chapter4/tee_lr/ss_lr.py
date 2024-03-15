from jax import grad
import jax.numpy as jnp
import secretflow as sf
import time
import math
from data_loader import _load_breast_cancer
import numpy as np
from sklearn.metrics import roc_auc_score


def breast_cancer(party_id=None, train: bool = True) -> (np.ndarray, np.ndarray):
    # x, y = load_breast_cancer(return_X_y=True)
    # x = (x - np.min(x)) / (np.max(x) - np.min(x))
    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=42
    # )
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


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


# Training loss is the negative log-likelihood of the training examples.
def loss(W, b, inputs, targets):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.mean(jnp.log(label_probs))


def train_step(W, b, x1, x2, y, learning_rate, batch_size):
    x = jnp.concatenate([x1, x2], axis=1)
# TODO https://github.com/secretflow/spu/blob/main/examples/python/ml/jax_lr/jax_lr.py
    xs = jnp.array_split(x, batch_size, axis=0)

    ys = jnp.array_split(y, batch_size, axis=0)

    def train_batch(w, b):
        for x, y in zip(xs, ys):
            Wb_grad = grad(loss, (0, 1))(W, b, x, y)
            W -= learning_rate * Wb_grad[0]
            b -= learning_rate * Wb_grad[1]
        return w, b
    for x, y in zip(xs, ys):
        Wb_grad = grad(loss, (0, 1))(W, b, x, y)
        W -= learning_rate * Wb_grad[0]
        b -= learning_rate * Wb_grad[1]
    # Wb_grad = grad(loss, (0, 1))(W, b, x, y)
    # W -= learning_rate * Wb_grad[0]
    # b -= learning_rate * Wb_grad[1]
    return W, b


def fit(W, b, x1, x2, y, epochs=1, learning_rate=0.01, batch_size=64):
    # sample_size = y.shape[0]
    # xs = jnp.array_split(x, len(x) / batch_size, axis=0)
    # ys = jnp.array_split(y, len(y) / batch_size, axis=0)
    for _ in range(epochs):
        # 
        W, b = train_step(W, b, x1, x2, y, learning_rate=learning_rate, batch_size=batch_size)
            # W, b = train_step(W, b, batch_x1, batch_x2, batch_y, learning_rate=learning_rate)
    return W, b


def validate_model(W, b, X_test, y_test):
    y_pred = predict(W, b, X_test)
    return roc_auc_score(y_test, y_pred)


if __name__ == "__main__":
    sf.shutdown()

    sf.init(['alice', 'bob'], address='local')

    alice, bob = sf.PYU('alice'), sf.PYU('bob')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    x1, y = alice(breast_cancer)(party_id=1)
    x2, _ = bob(breast_cancer)(party_id=2)

    X_test, y_test = breast_cancer(train=False)

    # x1, x2, y
    W = jnp.zeros((30,))
    b = 0.0

    W_, b_, x1_, x2_, y_ = (
        sf.to(alice, W).to(spu),
        sf.to(alice, b).to(spu),
        x1.to(spu),
        x2.to(spu),
        y.to(spu),
    )

    W_, b_ = spu(
        fit,
        static_argnames=['epochs'],
        num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
        user_specified_num_returns=2,
    )(W_, b_, x1_, x2_, y_, epochs=10, learning_rate=0.01, batch_size=5)

    auc = validate_model(sf.reveal(W_), sf.reveal(b_), X_test, y_test)

    print(f'auc={auc}')