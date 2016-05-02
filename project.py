import pandas as pd
import xgboost as xgb
import numpy as np
import os.path
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.grid_search import GridSearchCV

CROSS_VALIDATION = True

class XGBoostClassifier():
    def __init__(self, **params):
        self.clf = None

        self.params = params

    def fit(self, X, y, num_boost_round=None):
        dtrain = xgb.DMatrix(csr_matrix(X), y)
        if num_boost_round:
            self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
        else:
            self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.num_boost_round)

    def predict(self, X):
        dtest = xgb.DMatrix(csr_matrix(X))
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

        # Min Max cut in test data
        print("Min Max cut")
        for col in train.columns[1:-1]:
            test[col].clip(train[col].min, train[col].max)

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

    # Feature Analysis, pcc TODO: choose the first n features?
    print("Calculating PCC")
    c = train.columns[1:-1]
    pcc = []
    for i in range(len(c)):
        p = abs(np.corrcoef(train[c[i]].values, train[train.columns[-1]].values)[0, 1])
        pcc.append(p)

    print("features %d" % len(pcc))

    classifier = XGBoostClassifier(eval_metric="auc", booster="gbtree", objective="binary:logistic", eta=0.02, max_depth=5, subsample = 0.6, colsample_bytree = 0.7, silent = 1)
    test_x = test[test.columns[1:]]  # exclude ID column

    # Cross validation
    if CROSS_VALIDATION:
        best_score = -1
        for num_features in [100, 150, 200, 250, 300, len(pcc)]:
            sorted_pcc_indices = np.add(np.array(pcc).argsort()[::-1][:num_features], 1)

            train_x = train[train.columns[sorted_pcc_indices]]  # remove ID column and TARGET column
            train_y = train[train.columns[-1]]
            print("Remaining features %d" % len(sorted_pcc_indices))
            print ("Cross Validation")
            # Parameters to be searched
            tuning_parameters = {
                'num_boost_round': [500],
                'eta': [0.02],
                'max_depth': [5],
                'subsample': [0.6],
                'colsample_bytree': [0.7],
            }
            classifiers = GridSearchCV(classifier, tuning_parameters, n_jobs=1, cv=3) # bug if n_jobs > 1 ???

            print("Start Training")
            classifiers.fit(train_x, train_y)

            best_parameters, score, _ = max(classifiers.grid_scores_, key=lambda x: x[1])
            print('Score:', score)
            if score > best_score:
                print('Best Score:', score)
                print('Best Parameters:')
                for param_name in sorted(best_parameters.keys()):
                    print("%s: %r" % (param_name, best_parameters[param_name]))
                best_score = score
                test_y = classifiers.predict(test_x)
    else:
        print ("No Cross Validation")
        train_x = train[train.columns[1:-1]]  # remove ID column and TARGET column
        train_y = train[train.columns[-1]]

        classifier.fit(train_x, train_y, num_boost_round = 500)
        score = roc_auc_score(train_y, classifier.predict(train_x))
        test_y = classifier.predict(test_x)
        print('Score:', score)

    # Save the processed data
    train.to_csv("data/processed_train_%f.csv" % best_score, index=False)
    test.to_csv("data/processed_test_%f.csv" % best_score, index=False)

    # Manual labeling TODO: add more from https://www.kaggle.com/zfturbo/santander-customer-satisfaction/to-the-top-v3/code
    for i in range(test.shape[0]):
        row = test.irow(i)
        nv = row['num_var33'] + row['saldo_medio_var33_ult3'] + row['saldo_medio_var44_hace2'] + row['saldo_medio_var44_hace3'] + row['saldo_medio_var33_ult1'] + row['saldo_medio_var44_ult1']
        if nv > 0 or \
            row['var15'] < 23 or \
            row['saldo_medio_var5_hace2'] > 160000 or \
            row['saldo_var33'] > 0 or \
            row['var38'] > 3988596 or \
            row['var21'] > 7500 or \
            row['num_var30'] > 9 or \
            row['num_var13_0'] > 6 or \
            row['num_var33_0'] > 0 or \
            row['imp_ent_var16_ult1'] > 51003 or \
            row['imp_op_var39_comer_ult3'] > 13184 or \
            row['saldo_medio_var5_ult3'] > 108251:

            test_y[i] = 0

    submission = pd.DataFrame({"ID": test.ID, "TARGET": test_y})
    submission.to_csv("test_pred_%f.csv" % score, index=False)