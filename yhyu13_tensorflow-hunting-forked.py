import numpy as np

import pandas as pd 

import seaborn as sns

import tensorflow as tf



import tflearn

from matplotlib import pyplot as plt

from sklearn.metrics import classification_report
%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



submission = pd.read_csv("../input/sample_submission.csv")

submission["type"] = "Unknown"
#sns.pairplot(train.drop("id",axis=1), hue="type", diag_kind="kde")
import itertools
comb = list(itertools.combinations(train.drop(["id", "color", "type"], axis=1).columns, 2))
try_comb = pd.DataFrame()

for c in comb:

    try_comb[c[0]+"_x_"+c[1]] = train[c[0]].values *train[c[1]].values



try_comb["type"] = train.type
#sns.pairplot(try_comb, hue="type", diag_kind="kde")
try_comb = None
for i in [1,2,-1]:

    train[comb[i][0]+"_x_"+comb[i][1]] = train[comb[i][0]].values * train[comb[i][1]].values

    test[comb[i][0]+"_x_"+comb[i][1]] = test[comb[i][0]].values * test[comb[i][1]].values
from sklearn.decomposition import KernelPCA
kPCA = KernelPCA(n_components=2, kernel="rbf", gamma=1)

transf = kPCA.fit_transform(train.drop(["id", "color", "type"], axis=1))



plt.figure(figsize=(10,8))



for label,marker,color in zip(["Ghost", "Ghoul", "Goblin"],('x', 'o', '^'),('blue', 'red', 'green')):



    plt.scatter(x=transf[:,0][(train.type == label).values],

                y=transf[:,1][(train.type == label).values],

                marker=marker,

                color=color,

                alpha=0.7,

                label='class {}'.format(label)

                )



plt.legend()

plt.title('KernelPCA projection')



plt.show()
train["kPCA_0"] = transf[:,0]

train["kPCA_1"] = transf[:,1]
transf_test = kPCA.transform(test.drop(["id", "color"], axis=1).values)



test["kPCA_0"] = transf_test[:,0]

test["kPCA_1"] = transf_test[:,1]
#sns.pairplot(train.drop(["id", "color"],axis=1), hue="type", diag_kind="kde")
X_train = train.drop(["id", "color", "type"], axis=1)

y_train = pd.get_dummies(train["type"]).values



X_test = test.drop(["id", "color"], axis=1)
run = True
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
scores = []

predictions = np.zeros((X_test.values.shape[0], 3))

for train_idx, val_idx in kf.split(X_train):

    with tf.Graph().as_default():



        net = tflearn.input_data(shape=[None, 9])



        net = tflearn.fully_connected(net, 1024,

                                      activation='relu',

                                      weights_init='xavier',

                                      regularizer='L2')

        net = tflearn.dropout(net, 0.5)



        net = tflearn.fully_connected(net, 3, activation='softmax')



        net = tflearn.regression(net)



        model = tflearn.DNN(net, tensorboard_verbose=0)



        model.fit(X_train.values[train_idx],

                  y_train[train_idx],

                  n_epoch=150)



        score = model.evaluate(X_train.values[val_idx], y_train[val_idx])



        scores.append(score[0])



        print("\n", "SCORE:", score[0], "\n\n")



        prediction = np.array(model.predict(X_test))



        predictions += prediction * score[0]
scores
test_pred = np.argmax(predictions, axis=1).astype(str)

test_pred[test_pred=="0"] = "Ghost"

test_pred[test_pred=="1"] = "Ghoul"

test_pred[test_pred=="2"] = "Goblin"

test_pred
submission["type"] = test_pred
submission.to_csv("NN.csv", index=False)