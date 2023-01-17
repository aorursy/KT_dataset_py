import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/HR_comma_sep.csv')

y = df['left']

df.drop('left',axis=1,inplace=True)

df = pd.get_dummies(df)

trainX, testX, trainY, testY = train_test_split(df, y, test_size=0.33, random_state=42)
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

clf = MLPClassifier(solver='lbfgs',activation='logistic',verbose=10,

                    alpha=1e-5,hidden_layer_sizes=(28, 2), random_state=1)

clf.fit(trainX, trainY)

print(accuracy_score(clf.predict(testX), testY))
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

import numpy

from keras.wrappers.scikit_learn import KerasClassifier



# create model

model = Sequential()

model.add(Dense(20, input_dim=20, init='uniform', activation='relu'))

model.add(Dense(8, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.array(trainX),np.array(trainY),nb_epoch=17,batch_size=16)

predictions = model.predict(np.array(testX))
print(accuracy_score(predictions.round(), np.array(testY)))
import tensorflow as tf

CONTINUOUS_COLUMNS = ["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours"]

LABEL_COLUMN = "label"

trainX[LABEL_COLUMN] = 0

testX[LABEL_COLUMN] = 0



def input_fn(df):

    # Creates a dictionary mapping from each continuous feature column name (k) to

    # the values of that column stored in a constant Tensor.

    continuous_cols = {k: tf.constant(df[k].values)

        for k in CONTINUOUS_COLUMNS}

    # Creates a dictionary mapping from each categorical feature column name (k)

    # to the values of that column stored in a tf.SparseTensor.

    #categorical_cols = {k: tf.SparseTensor(

    #    indices=[[i, 0] for i in range(df[k].size)],

    #    values=df[k].values,

    #    shape=[df[k].size, 1])

    #for k in CATEGORICAL_COLUMNS}

    # Merges the two dictionaries into one.

    #feature_cols = dict(continuous_cols.items() + categorical_cols.items())

    feature_cols = dict(continuous_cols.items())

    # Converts the label column into a constant Tensor.

    label = tf.constant(df[LABEL_COLUMN].values)

    #Returns the feature columns and the label.

    return feature_cols, label



def train_input_fn():

    return input_fn(trainX)



def eval_input_fn():

    return input_fn(testX)
print(df.columns.values)
#satisfaction_level = tf.contrib.layers.real_valued_column("satisfaction_level")

#last_evaluation = tf.contrib.layers.real_valued_column("last_evaluation")

#number_project = tf.contrib.layers.real_valued_column("number_project")

#average_montly_hours = tf.contrib.layers.real_valued_column("average_montly_hours")
#model_dir = trainX.mkdtemp()

#m = tf.contrib.learn.LinearClassifier(feature_columns=[satisfaction_level,last_evaluation,number_project,average_montly_hours])
#m.fit(input_fn=train_input_fn, steps=10)
#results = m.evaluate(input_fn=eval_input_fn, steps=1)
#for key in sorted(results):

#    print(key, results[key])