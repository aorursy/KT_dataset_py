# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # machine learning, deep learning



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

sess.run(hello)
a = tf.constant(10)

b = tf.constant(32)

sess.run(a+b)
# Read in data with Pandas

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
x = train.values[:,1:]

print(x.shape)

y = train['label'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



#xt = test.values[:,:]

#yt = test['label'].values
def input_fn(df):

    feature_columns = {k: tf.constant(df[k].values)

                      for k in df.columns if k != 'label'}

    labels = tf.constant(df['label'].values)

    return feature_columns, labels



def train_input_fn():

    return input_fn(train)



def eval_input_fn():

    return input_fn(test)
# Specify that all features have real-value data

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=784)]
# Build 3 layer DNN with 10, 20, 10 units respectively

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,

                                           hidden_units=[10, 20, 10],

                                           n_classes=10)
classifier.fit(x=x, y=y, steps=200)
#yp = list(classifier.predict(xt, as_iterable=True))
#yp[0:10]
#df = pd.DataFrame()

#ids = [x+1 for x in range(len(yp))]

#df["ImageId"] = ids

#df["Label"] = yp
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))