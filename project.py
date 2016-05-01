import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
import os.path
from sklearn import preprocessing
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.grid_search import GridSearchCV

class XGBoostClassifier():
    def __init__(self, **params):
        self.clf = None

        self.params = params

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        pred = self.predict(X)
        return roc_auc_score(y, pred)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')

        self.params.update(params)
        return self

if __name__ == "__main__":
    if not os.path.isfile('data/processed_train.csv') or not os.path.isfile('data/processed_test.csv'):
        print("Processed data does not exist")
        train = pd.read_csv('data/train.csv')
        test = pd.read_csv('data/test.csv')

        # Add ZeroSum feature, 371 + 1, this implies the credibility of the sample
        print("Add ZeroSum feature")
        feature_names = train.columns[1:-1]
        train.insert(1, 'ZeroSum', (train[feature_names] == 0).astype(int).sum(axis=1))
        test.insert(1, 'ZeroSum', (test[feature_names] == 0).astype(int).sum(axis=1))

        # Remove constant features
        print("Removing constant features")
        remove = []
        for col in train.columns:
            if train[col].std() == 0:
                remove.append(col)

        train.drop(remove, axis=1, inplace=True)
        test.drop(remove, axis=1, inplace=True)

        # Remove identical features
        print("Removing identical features")
        remove = []
        c = train.columns[1:-1]
        for i in range(len(c)):
            v = train[c[i]].values
            for j in range(i + 1, len(c)):
                if np.array_equal(v, train[c[j]].values):
                    remove.append(c[j])

        train.drop(remove, axis=1, inplace=True)
        test.drop(remove, axis=1, inplace=True)

        print("Remaining features %d" % len(train.columns))
        # TODO: Min Max cut in test data ??

        # TODO: Feature Analysis??? PCC or MI

        # Normalization
        scaler = preprocessing.StandardScaler(copy = False).fit(train[train.columns[1:-1]])
        scaler.transform(train[train.columns[1:-1]])
        scaler.transform(test[test.columns[1:]])

        # Save the processed data
        train.to_csv("data/processed_train.csv", index=False)
        test.to_csv("data/processed_test.csv", index=False)
    else:
        print("Processed data exists")
        train = pd.read_csv('data/processed_train.csv')
        test = pd.read_csv('data/processed_test.csv')

    train_x = train[train.columns[1:-1]]  # remove ID column and TARGET column
    train_y = train[train.columns[-1]]

    # Cross validation
    classifier = XGBoostClassifier(eval_metric="auc", booster="gbtree", objective="binary:logistic", silent = 1)

    # Parameters to be searched
    # tuning_parameters = {
    #     'num_boost_round': [560, 100],
    #     'eta': [0.0202048, 0.05],
    #     'max_depth': [5, 6],
    #     'subsample': [0.6815, 0.9],
    #     'colsample_bytree': [0.701, 0.9],
    # }
    tuning_parameters = {
        'num_boost_round': [1],
        'eta': [0.0202048],
        'max_depth': [5],
        'subsample': [0.6815],
        'colsample_bytree': [0.701],
    }
    classifiers = GridSearchCV(classifier, tuning_parameters, n_jobs=1, cv=3) # bug if n_jobs > 1 ???

    print("Start Training")
    classifiers.fit(train_x, train_y)

    best_parameters, score, _ = max(classifiers.grid_scores_, key=lambda x: x[1])
    print('Best Score:', score)
    print('Best Parameters:')
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    # Save the processed data
    train.to_csv("data/processed_train_%f.csv" % score, index=False)
    test.to_csv("data/processed_test_%f.csv" % score, index=False)

    test_x = test[test.columns[1:]]  # exclude ID column
    test_y = classifiers.predict(test_x)

    submission = pd.DataFrame({"ID": test.ID, "TARGET": test_y})
    submission.to_csv("test_pred_%f.csv" % score, index=False)