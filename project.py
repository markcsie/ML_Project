import pandas as pd
import xgboost as xgb
import numpy as np
import pickle
import os.path
from sklearn import preprocessing
from sklearn.metrics import log_loss, roc_auc_score

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

        # TODO: Cross validation?????

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

    if not os.path.isfile("classifier.p"):
        print("Classifier does not exist")
        train_x = train[train.columns[1:-1]]  # remove ID column and TARGET column
        train_y = train[train.columns[-1]]

        # TODO: Parameter explaination?????
        params = {}
        params["objective"] = "binary:logistic"
        params["booster"] = "gbtree"
        params["eval_metric"] = "auc"
        params["eta"] = 0.0202048
        params["max_depth"] = 5
        params["subsample"] = 0.6815
        params["colsample_bytree"] = 0.701

        train_data = xgb.DMatrix(train_x, train_y)
        print("Start Training")
        classifier = xgb.train(params = params, dtrain = train_data, num_boost_round = 560, verbose_eval = False)

        train_pred = classifier.predict(train_data)
        train_log_loss = log_loss(train_y, train_pred)
        train_roc = roc_auc_score(train_y, train_pred)

        print(train_log_loss)
        print(train_roc)

        pickle.dump(classifier, open("classifier_%f_%f.p" % (train_log_loss, train_roc), "wb"))
        pickle.dump(classifier, open("classifier.p", "wb"))

        # Save the processed data
        train.to_csv("data/processed_train_%f_%f.csv" % (train_log_loss, train_roc), index=False)
        test.to_csv("data/processed_test_%f_%f.csv" % (train_log_loss, train_roc), index=False)
    else:
        print("Classifier does exists")
        classifier = pickle.load(open("classifier.p", "rb" ))

    test_x = test[test.columns[1:]]  # exclude ID column
    test_data = xgb.DMatrix(test_x)
    test_y = classifier.predict(test_data)

    submission = pd.DataFrame({"ID": test.ID, "TARGET": test_y})
    submission.to_csv("test_pred_%f.csv" % train_roc, index=False)