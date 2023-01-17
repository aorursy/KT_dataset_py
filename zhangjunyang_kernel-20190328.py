# 注意,Iris数据集给出的三种花是按照顺序来的,前50个是第0类,51-100是第1类,101~150是第2类,分训练集和测试集时要把顺序打乱,

# 这里引入一个两类shuffle()函数,它接收两个参数,分别是x和y,然后把x,y绑在一起shuffle.



from sklearn.datasets import load_iris

import numpy

import pandas

import warnings

warnings.filterwarnings('ignore') 



# iris = pd.read_csv("../input/Iris.csv");

iris = load_iris()



def shuffle_in_unison(a, b):

    assert len(a) == len(b)

    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)

    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)

    permutation = numpy.random.permutation(len(a))

    for old_index, new_index in enumerate(permutation):

        shuffled_a[new_index] = a[old_index]

        shuffled_b[new_index] = b[old_index]

    return shuffled_a, shuffled_b



def load_data():

    # import random

    # iris.data = random.shuffle(iris.data)

    # iris.target = random.shuffle(iris.target)

    iris.data, iris.target = shuffle_in_unison(iris.data, iris.target)

    x_train ,x_test = iris.data[:100],iris.data[100:]

    y_train, y_test = iris.target[:100].reshape(-1,1),iris.target[100:].reshape(-1,1)

    return x_train, y_train, x_test, y_test



from sklearn import tree, svm, naive_bayes,neighbors

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

import tensorflow as tf

# from tf.contrib import learn



x_train, y_train, x_test, y_test = load_data()



clfs = {

        # 'DNN': tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10,20,10], n_classes=3,

        #                                         optimizer=tf.train.AdamOptimizer(learning_rate=0.01), model_dir=”/logs”), \

        'svm': svm.SVC(),\

        'decision_tree': tree.DecisionTreeClassifier(), \

        'naive_gaussian': naive_bayes.GaussianNB(), \

        'naive_mul': naive_bayes.MultinomialNB(),\

        'K_neighbor' : neighbors.KNeighborsClassifier(),\

        'bagging_knn' : BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5), \

        'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),

        'random_forest' : RandomForestClassifier(n_estimators=50),\

        'adaboost':AdaBoostClassifier(n_estimators=50),\

        'gradient_boost' : GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)

        }

# def try_DNN_method(clf):

#     clf.fit(x_train, y_train, step=100)

#     score = clf.score(x_test,y_test.ravel())

#     print('the score is :', score)



def try_different_method(clf):

    clf.fit(x_train,y_train.ravel())

    score = clf.score(x_test,y_test.ravel())

    print('the score is :', score)



for clf_key in clfs.keys():

    print('the classifier is :',clf_key)

#     if clf_key == 'DNN':

#         clf = clfs[clf_key]

#         try_DNN_method(clf)

#     else:

#         clf = clfs[clf_key]

#         try_different_method(clf)



    clf = clfs[clf_key]

    try_different_method(clf)