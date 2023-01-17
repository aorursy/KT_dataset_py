from sklearn.cross_validation import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings 
warnings.filterwarnings("ignore")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

def load_data(pickle_file):
    load_file = open(pickle_file,'rb')
    data = pickle.load(load_file)
    return data


def pickle_data(path, data):
    file = path
    save_file = open(file, 'wb')
    pickle.dump(data, save_file, -1)
    save_file.close()
def split_data(data, nrounds=5, train_size=0.8, stratified=True):
    # initialise random state
    rnd_state = np.random.RandomState(1234)
    split_results = []
    # label for stratified split, assump to be the last column of the input data
    y = data.iloc[:, data.shape[1]-1].as_matrix()
    # perform splitting runs
    for run in range(0, nrounds):
        if stratified:
            train_ix, val_ix = train_test_split(np.arange(data.shape[0]), train_size=train_size,                                                
                                                stratify=y, random_state=rnd_state)
        else:
            train_ix, val_ix = train_test_split(np.arange(data.shape[0]), train_size=train_size,
                                                random_state=rnd_state)
        data_train = data.ix[train_ix]
        data_val = data.ix[val_ix]
        train_zeros=data_train["TARGET"].value_counts()[0]
        train_ones=data_train["TARGET"].value_counts()[1]
        val_zeros=data_val["TARGET"].value_counts()[0]
        val_ones=data_val["TARGET"].value_counts()[1]

        print ("Run %d, zero in data_train: %d, one in data_train: %d" %(run, train_zeros, train_ones))
        print ("Run %d, zero in data_val: %d, one in data_val: %d" %(run, val_zeros, val_ones))

    return 0
print("Performing stratified data split")
stratified_splits = split_data(train, nrounds=5)
print("Performing non-stratified data split")
non_stratified_splits=split_data(train, nrounds=20, stratified=False)