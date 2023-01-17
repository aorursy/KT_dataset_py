import numpy
from keras.utils import get_file
from keras.utils.np_utils import to_categorical
from zipfile import ZipFile
import implicit
from implicit.als import AlternatingLeastSquares
def load_data():
    path = get_file('ml-100k.zip', origin='http://files.grouplens.org/datasets/movielens/ml-100k.zip')
    with ZipFile(path, 'r') as ml_zip:
        max_item_id  = -1
        train_history = {}
        with ml_zip.open('ml-100k/ua.base', 'r') as file:
            for line in file:
                user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('\t')
                if int(user_id) not in train_history:
                    train_history[int(user_id)] = [int(item_id)]
                else:
                    train_history[int(user_id)].append(int(item_id))

                if max_item_id < int(item_id):
                    max_item_id = int(item_id)

        test_history = {}
        with ml_zip.open('ml-100k/ua.test', 'r') as file:
            for line in file:
                user_id, item_id, rating, timestamp = line.decode('utf-8').rstrip().split('\t')
                if int(user_id) not in test_history:
                    test_history[int(user_id)] = [int(item_id)]
                else:
                    test_history[int(user_id)].append(int(item_id))
        
    max_item_id += 1 # actual item_id starts from 1
    train_users = list(train_history.keys())
    # convert items as sparse matrix for each user
    # sparse matrix: items which user liked are 1
    train_x = numpy.zeros((len(train_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(train_history.values()):
        mat = to_categorical(hist, max_item_id)
        train_x[i] = numpy.sum(mat, axis=0)

    test_users = list(test_history.keys())
    test_x = numpy.zeros((len(test_users), max_item_id), dtype=numpy.int32)
    for i, hist in enumerate(test_history.values()):
        mat = to_categorical(hist, max_item_id)
        test_x[i] = numpy.sum(mat, axis=0)

    return train_users, train_x, test_users, test_x
train_users, train_x, test_users, test_x= load_data()
train_users[0:3]
train_x[0:3]
# 943 users, 1683 items
train_x.shape
import numpy as np
from scipy.sparse import csr_matrix, random
# csr_matrix is a class
user_items = csr_matrix(train_x, dtype=np.float64)
user_items
# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(user_items.T)

item_factors, user_factors = model.item_factors, model.user_factors
# item-factors matrix
item_factors.shape
item_factors[0]
# user-factors matrix
user_factors.shape
user_factors[0]
# recommend five items for user (index=0)
recs = model.recommend(0, user_items, N=5)
# (itemid, score)
recs
# recommend five items for user (index=100)
recs = model.recommend(100, user_items, N=5)
# (itemid, score)
recs