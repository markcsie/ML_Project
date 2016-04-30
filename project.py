import pandas as pd

if __name__ == "__main__":
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # print (train)
    x = train.columns[1:-1] # remove ID column // 371 - 1