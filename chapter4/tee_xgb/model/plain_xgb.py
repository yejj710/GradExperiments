import sys
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
sys.path.append("/home/yejj/GradExperiments/chapter4/tee_xgb")
from data_loader import load_weather_aus, load_a9a
import time
from sklearn.preprocessing import LabelEncoder


def test_aga():
    x_train, y_train, x_test, y_test = load_a9a()
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    # clf = XGBClassifier(
    #     use_label_encoder=False,
    #     max_depth=5,
    #     n_estimators=20,
    #     learning_rate=0.2,
    #     reg_lambda=0.1,
    #     colsample_bytree=1)
    clf = XGBClassifier(n_estimators=20)
    
    start = time.time()
    clf.fit(x_train, y_train)
    print(f"train time: {(time.time() - start)}")

    # train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)

    # print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_train, train_predict))
    print('The AUC is:',metrics.roc_auc_score(y_test, test_predict))
    y_pred = list(map(lambda x: 0 if x <=0.5 else 1, test_predict))
    print('The f1 is:',metrics.f1_score(y_test, y_pred))
    # print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test, test_predict))

    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
    print('The confusion matrix result:\n', confusion_matrix_result)


def test_weather():
    x_train, x_test, y_train, y_test = load_weather_aus(first_load=False)

    params = {
        # for more detail, see Xgb API doc
        'num_boost_round': 2,
        'max_depth': 5,
        'learning_rate': 0.2,
        'sketch_eps': 0.08,
        'objective': 'logistic',
        'reg_lambda': 0.1,
        'colsample_by_tree': 1,
        'base_score': 0.5,
    }
    clf = XGBClassifier()
    # clf = XGBClassifier(
    #     use_label_encoder=False,
    #     max_depth=5,
    #     n_estimators=20,
    #     learning_rate=0.2,
    #     reg_lambda=0.1,
    #     colsample_bytree=1)

    start = time.time()
    clf.fit(x_train, y_train)
    print(f"train time: {(time.time() - start)}")
    # train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)

    # print('The accuracy of the train set is:',metrics.accuracy_score(y_train, train_predict))
    # print('The accuracy of the test set is:',metrics.accuracy_score(y_test, test_predict))
    print('The AUC is:',metrics.roc_auc_score(y_test, test_predict))
    y_pred = list(map(lambda x: 0 if x <0.5 else 1, test_predict))
    print('The f1 is:',metrics.f1_score(y_test, y_pred, average='binary'))


if __name__ == "__main__":
    # labels =    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    # predicts =  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # print(metrics.f1_score(labels, predicts))
    # test_aga()
    # test_weather()

    x_train, x_test, y_train, y_test = load_weather_aus(first_load=False)
    print(y_train.value_counts())
    print(y_test.value_counts())
