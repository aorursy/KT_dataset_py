import pandas as pd
import numpy as np


def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('ID')
    return output


def prepare_dataset():
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    features = train.columns.values
    
    norm_f = []
    for f in features:
        u = train[f].unique()
        if len(u) != 1:
            norm_f.append(f)


    remove = []
    for i in range(len(norm_f)):
        v1 = train[norm_f[i]].values
        for j in range(i+1, len(norm_f)):
            v2 = train[norm_f[j]].values
            if np.array_equal(v1, v2):
                remove.append(norm_f[j])
    
    for r in remove:
        norm_f.remove(r)

    train = train[norm_f]
    norm_f.remove('TARGET')
    test = test[norm_f]
    features = get_features(train, test)
    return train, test, features


def find_min_max_features(df, f):
    return df[f].min(), df[f].max()


def analayze_data(train, test):
    print('Length of train: ', len(train.index))
    train_zero = train[train['TARGET'] == 0]
    print('Length of train [TARGET = 0]: ', len(train_zero.index))
    train_one = train[train['TARGET'] == 1]
    print('Length of train [TARGET = 1]: ', len(train_one.index))
    # train_one.to_csv("debug.csv", index=False)
    one_range = dict()
    for f in train.columns:
        mn0, mx0 = find_min_max_features(train_zero, f)
        mn1, mx1 = find_min_max_features(train_one, f)
        mnt = 'N/A'
        mxt = 'N/A'
        if f in test.columns:
            mnt, mxt = find_min_max_features(test, f)
        one_range[f] = (mn1, mx1)
        if mn0 != mn1 or mn1 != mnt or mx0 != mx1 or mx1 != mxt:
            print("\nFeature {}".format(f))
            print("Range target=0  ({} - {})".format(mn0, mx0))
            print("Range target=1  ({} - {})".format(mn1, mx1))
            print("Range in test   ({} - {})".format(mnt, mxt))


train, test, features = prepare_dataset()
analayze_data(train, test)