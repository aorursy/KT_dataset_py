from sklearn.model_selection import KFold

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

kf = KFold(n_splits=3, shuffle=True)

for train_index, test_index in kf.split(X):

     print("%s %s" % (train_index, test_index))
from sklearn.model_selection import StratifiedKFold

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

skf = StratifiedKFold(n_splits=4, shuffle=True)

for train_index, test_index in skf.split(X, y):

    print("%s %s" % (train_index, test_index))
from sklearn.model_selection import GroupKFold

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

groups = ['a','a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd']

gkf = GroupKFold(n_splits=4)

for train_index, test_index in gkf.split(X, y, groups=groups):

     print("%s %s" % (train_index, test_index))