import pandas as pd
import xgboost as xgb
import pickle
import os.path
from sklearn.metrics import log_loss, roc_auc_score

if __name__ == "__main__":
    # TODO: Add ZeroSum feature

    # TODO: Remove constant features

    # TODO: Remove identical features

    # TODO: Min Max cut in test data ??

    # TODO: Cross validation?????

    if not os.path.isfile("classifier.p"):
        train = pd.read_csv('data/train.csv')
        train_x = train[train.columns[1:-1]]  # remove ID column and TARGET column, 371 - 1 - 1
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
    else:
        classifier = pickle.load(open("classifier.p", "rb" ))

    test = pd.read_csv('data/test.csv')
    test_x = test[test.columns[1:]]  # remove ID column
    test_data = xgb.DMatrix(test_x)
    test_y = classifier.predict(test_data)

    submission = pd.DataFrame({"ID": test.ID, "TARGET": test_y})
    submission.to_csv("test_pred.csv", index=False)