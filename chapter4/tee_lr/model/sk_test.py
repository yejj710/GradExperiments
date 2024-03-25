import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

sys.path.append("/home/yejj/GradExperiments/chapter4/tee_lr")
from utils.data_loader import _load_breast_cancer, load_epsilon
# x_train, y_train, x_test, y_test = _load_breast_cancer(315)
x_train, y_train, x_test, y_test = load_epsilon()


model = linear_model.LogisticRegression(max_iter=10)

model.fit(x_train, y_train)

y_hat = model.predict(x_test)

print("auc=", roc_auc_score(y_test, y_hat))
y_pred = list(map(lambda x: 0 if x <0.5 else 1, y_hat))

print("f1=", f1_score(y_test, y_pred, average='binary'))