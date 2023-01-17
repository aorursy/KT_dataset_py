# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
LEARNING_RATE = 0.03
MAX_ITERATION = 5000
BATCH_SIZE = 500
def read_data(file, scaler):
    df = pd.read_csv(file)[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
    df = df.fillna(df.mean())
    df = pd.concat([df, pd.get_dummies(df['Sex'])], axis=1).drop('Sex', axis=1)
    df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='Pclass')], axis=1).drop('Pclass', axis=1)
    x = df[['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age', 'SibSp', 'Parch', 'male', 'female']].values
    scaler.fit(x)
    x = scaler.transform(x)
    y = df['Survived'].values
    return x, y, scaler
def read_test_data(file, scaler):
    df = pd.read_csv(file)[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
    df = df.fillna(df.mean())
    df = pd.concat([df, pd.get_dummies(df['Sex'])], axis=1).drop('Sex', axis=1)
    df = pd.concat([df, pd.get_dummies(df['Pclass'], prefix='Pclass')], axis=1).drop('Pclass', axis=1)
    x = df[['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age', 'SibSp', 'Parch', 'male', 'female']].values
    x = scaler.transform(x)
    return x, df
scaler = StandardScaler()
X_all, y_all, scaler = read_data('../input/train.csv', scaler)
X_test, df_test = read_test_data('../input/test.csv', scaler)
X_train, X_validation, y_train, y_validation = train_test_split(X_all, y_all, test_size=0.20, random_state=0)
def make_dnn(input, output_units, inner_units):
    next_input = input
    for units in inner_units:
        next_input = tf.layers.dense(next_input, units, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(), bias_initializer=tf.random_normal_initializer())
    return tf.layers.dense(next_input, output_units, activation=tf.sigmoid)
X_ph = tf.placeholder(tf.float32, shape=(None, 8), name='X')
y_ph = tf.placeholder(tf.float32, shape=(None, 1), name='y')
dnn_out = make_dnn(X_ph, 1, [8, 8, 8])
loss = tf.losses.sigmoid_cross_entropy(y_ph, dnn_out)
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
prediction = tf.round(dnn_out)
accuracy = tf.losses.mean_squared_error(1-prediction, y_ph)
initializer = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initializer)
    for step in range(MAX_ITERATION):
        batch_idx = np.random.choice(len(X_train), size=(BATCH_SIZE,))
        loss_val, accuracy_val, _ = sess.run([loss, accuracy, optimizer], feed_dict={X_ph: X_train[batch_idx], y_ph: np.reshape(y_train[batch_idx], (-1, 1))})
        if step % (MAX_ITERATION//5) == 0:
            print('Epoch ', step, ': Loss=', loss_val, 'Accuracy=', accuracy_val)
            
    loss_val, accuracy_val = sess.run([loss, accuracy], feed_dict={X_ph: X_validation, y_ph: np.reshape(y_validation, (-1, 1))})
    print('On Validation Data:')
    print('Loss=', loss_val, 'Accuracy=', accuracy_val)
    
    [prediction_val] = sess.run([prediction], feed_dict={X_ph: X_test})
    print('Prediction on test data:')
    df_test['Survived'] = prediction_val.flatten().astype(int)
    print(df_test)
    
    df_test[['PassengerId','Survived']].to_csv('./submission.csv', index=False)