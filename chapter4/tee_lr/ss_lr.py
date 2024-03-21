from jax import grad
import jax.numpy as jnp
import secretflow as sf
import time
import jax
from data_loader import _load_breast_cancer
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


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


def _epsilon(party_id=None, train: bool = True) -> (np.ndarray, np.ndarray):
    x_train = np.load("./epsilon/x_train.npy")
    x_test = np.load("./epsilon/x_test.npy")
    y_train = np.load("./epsilon/y_train.npy")
    y_test = np.load("./epsilon/y_test.npy")
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


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


def loss(W, b, inputs, targets):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.mean(jnp.log(label_probs))


def train_step(W, b, x1, x2, y, learning_rate, batch_num):
    x = jnp.concatenate([x1, x2], axis=1)
# refer to https://github.com/secretflow/spu/blob/main/examples/python/ml/jax_lr/jax_lr.py
    xs = jnp.array_split(x, batch_num, axis=0)
    ys = jnp.array_split(y, batch_num, axis=0)

    def body_fun(_, loop_carry):
        W, b = loop_carry
        for x, y in zip(xs, ys):
            g = grad(loss, argnums=(0, 1))(W, b, x, y)
            W -= g[0] * learning_rate
            b -= g[1] * learning_rate

        return W, b

    return jax.lax.fori_loop(0, batch_num, body_fun, (W, b))
    # Wb_grad = grad(loss, (0, 1))(W, b, x, y)
    # W -= learning_rate * Wb_grad[0]
    # b -= learning_rate * Wb_grad[1]
    return W, b



def fit(W, b, x1, x2, y, x_test, y_test, epochs=1, learning_rate=0.01, batch_num=10):
    loss_history = list()
    x = jnp.concatenate([x1, x2], axis=1)
    for _ in range(epochs):

        W, b = train_step(W, b, x1, x2, y, learning_rate=learning_rate, batch_num=batch_num)
            # W, b = train_step(W, b, batch_x1, batch_x2, batch_y, learning_rate=learning_rate)
        pred = predict(W, b, x)
        loss = jnp.log(pred) * y+ jnp.log(1 - pred) * (1 - y)
        loss_history.append(-jnp.mean(loss))

    return W, b, loss_history


def validate_model(W, b, X_test, y_test):
    y_pred_score = predict(W, b, X_test)
    y_pred = list(map(lambda x: 0 if x <0.5 else 1, y_pred_score))

    return roc_auc_score(y_test, y_pred_score), f1_score(y_test, y_pred, average='binary')


if __name__ == "__main__":
    sf.shutdown()

    sf.init(['alice', 'bob'], address='local')

    alice, bob = sf.PYU('alice'), sf.PYU('bob')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    x1, y = alice(breast_cancer)(party_id=1)
    x2, _ = bob(breast_cancer)(party_id=2)

    X_test, y_test = breast_cancer(train=False)
    # x1, y = alice(_epsilon)(party_id=1)
    # x2, _ = bob(_epsilon)(party_id=2)

    # X_test, y_test = _epsilon(train=False) 
    # print(y_test[:10], type(y_test))
    # x1, x2, y
    W = jnp.zeros((30,))
    # W = jnp.zeros((500,))
    b = 0.0

    W_, b_, x1_, x2_, y_ = (
        sf.to(alice, W).to(spu),
        sf.to(alice, b).to(spu),
        x1.to(spu),
        x2.to(spu),
        y.to(spu),
    )
    start_time = time.time()
    W_, b_, loss_history = spu(
        fit,
        static_argnames=['epochs', 'learning_rate', 'batch_num'],
        num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
        user_specified_num_returns=3,
    )(W_, b_, x1_, x2_, y_, X_test, y_test, epochs=5, learning_rate=0.05, batch_num=1)

    auc, f1 = validate_model(sf.reveal(W_), sf.reveal(b_), X_test, y_test)
    print(f"train time :{time.time()-start_time}")

    print(f'auc={auc}, f1={f1}')
    print(sf.reveal(loss_history))