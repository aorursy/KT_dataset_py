import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.model_selection import GridSearchCV

import os

print(os.listdir("../input"))
def show_accuracy(a, b, tip):

    acc = a.ravel() == b.ravel()

    print(tip + '正确率：%.2f%%' % (100*np.mean(acc)))
if __name__ == '__main__':



    data = pd.read_csv('../input/train.csv')

    label = data.label

    data=data.drop('label',axis=1)

    train, test,train_labels, test_labels = train_test_split(data, label, train_size=0.8, random_state=42)

    print('Load Data OK...')
    clf = svm.SVC(C=1, kernel='rbf', gamma=0.001) 

    print('Start Learning...')

    clf.fit(train[:-1], train_labels[:-1].values.ravel())

    print('Learning is OK...')

    train_label_hat = clf.predict(train)

    show_accuracy(train_labels, train_label_hat, '训练集')

    train_label_hat = clf.predict(test)

    print(train_label_hat)

    print(test_labels)

    show_accuracy(test_labels, test_labels, '测试集')