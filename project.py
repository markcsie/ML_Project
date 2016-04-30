import pandas as pd
import xgboost as xgb
import pickle
import os.path

if __name__ == "__main__":
    if not os.path.isfile("classifier.p"):
        train_x = train[train.columns[1:-1]]  # remove ID column and TARGET column, 371 - 1 - 1
        train_y = train[train.columns[-1]]
        print("Start Training")
        train = pd.read_csv('data/train.csv')

        # TODO:Remove constant features

        params = {}
        params["objective"] = "binary:logistic"
        params["booster"] = "gbtree"
        params["eval_metric"] = "auc"
        params["eta"] = 0.0202048
        params["max_depth"] = 5
        params["subsample"] = 0.6815
        params["colsample_bytree"] = 0.701

        train_data = xgb.DMatrix(train_x, train_y)
        watchlist = [(train_data, 'train')]
        classifier = xgb.train(params = params, dtrain = train_data, num_boost_round = 560, evals = watchlist, verbose_eval=False)
        pickle.dump(classifier, open("classifier.p", "wb"))
    else:
        classifier = pickle.load(open("classifier.p", "rb" ))

    test = pd.read_csv('data/test.csv')
    test_x = test[test.columns[1:]]  # remove ID column
    test_data = xgb.DMatrix(test_x)
    test_y = classifier.predict(test_data)

    submission = pd.DataFrame({"ID": test.ID, "TARGET": test_y})
    submission.to_csv("test_pred.csv", index=False)
