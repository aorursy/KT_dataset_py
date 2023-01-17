# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

from ml_metrics import kappa

from sklearn.metrics import cohen_kappa_score

import tensorflow as tf
y_true = np.array([0.0,0.0,0.0])

y_pred = np.array([0.0,0.0,0.0])

cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_true = np.array([0.0,0.0,0.0])

y_pred = np.array([0.0,0.0,0.0])

cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_true = np.array([1.0,2.0,3.0])

y_pred = np.array([0.0,2.0,3.0])

cohen_kappa_score(y_true, y_pred, weights='quadratic')
y_true = np.array([1.0,2.0,3.0])

y_pred = np.array([1.0,2.0,3.0])

cohen_kappa_score(y_true, y_pred, weights='quadratic')
def qw_kappa_score(y_true, y_pred):

    #y_true=tf.math.argmax(y_true, axis=1)

    #y_pred=tf.math.argmax(y_pred, axis=1)

    threshold = tf.constant([0.37757874193797547])

    y_pred = tf.subtract(tf.reduce_sum(tf.cast(tf.math.greater(y_pred, threshold), dtype=tf.float32), axis=1), 1)

    y_true = tf.subtract(tf.reduce_sum(y_true, axis=1), 1)

    def sklearn_qwk(y_true, y_pred) -> np.float64:

        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    return tf.compat.v1.py_func(sklearn_qwk, (y_true, y_pred), tf.double)
qw_kappa_score(tf.constant([[0.0,0.0,0.0],

                            [0.0,0.0,0.0],

                            [0.0,0.0,0.0]]), 

               tf.constant([[0.0,0.0,0.0],

                            [0.0,0.0,0.0],

                            [0.0,0.0,0.0]]))
qw_kappa_score(tf.constant([[1.0,1.0,0.0],

                            [1.0,0.0,0.0],

                            [1.0,0.0,0.0]]),

               tf.constant([[0.9,0.5,0.0],

                            [0.9,0.3,0.0],

                            [0.9,0.1,0.0]])

               )
qw_kappa_score(tf.constant([[1.0,1.0,0.0],

                            [1.0,0.0,0.0],

                            [1.0,0.0,0.0]]),

               tf.constant([[0.9,0.5,0.0],

                            [0.9,0.3,0.0],

                            [0.9,0.1,0.0]])

               ).numpy()
qw_kappa_score(tf.constant([[1.0,1.0,0.0],

                            [1.0,0.0,0.0],

                            [1.0,0.0,0.0]]),

               tf.constant([[0.9,0.5,0.0],

                            [0.9,0.3,0.0],

                            [0.9,0.5,0.0]])

               ).numpy()
def qw_kappa_score(y_true, y_pred):

    #y_true=tf.math.argmax(y_true, axis=1)

    #y_pred=tf.math.argmax(y_pred, axis=1)

    threshold = tf.constant([0.37757874193797547])

    y_pred = tf.subtract(tf.reduce_sum(tf.cast(tf.math.greater(y_pred, threshold), dtype=tf.float32), axis=1), 1)

    y_true = tf.subtract(tf.reduce_sum(y_true, axis=1), 1)

    def sklearn_qwk(y_true, y_pred) -> np.float64:

        return cohen_kappa_score(y_true, y_pred, weights='quadratic')

    result = tf.compat.v1.py_func(sklearn_qwk, (y_true, y_pred), tf.double)

    return tf.where(tf.math.is_nan(result), tf.constant([1.0], dtype=tf.double), result)
qw_kappa_score(tf.constant([[1.0,1.0,0.0],

                            [1.0,0.0,0.0],

                            [1.0,0.0,0.0]]),

               tf.constant([[0.9,0.5,0.0],

                            [0.9,0.3,0.0],

                            [0.9,0.5,0.0]])

               )
qw_kappa_score(tf.constant([[1.0,1.0,0.0],

                            [1.0,0.0,0.0],

                            [1.0,0.0,0.0]]),

               tf.constant([[0.9,0.5,0.0],

                            [0.9,0.3,0.0],

                            [0.9,0.5,0.0]])

               ).numpy()
qw_kappa_score(tf.constant([[0.0,0.0,0.0],

                            [0.0,0.0,0.0],

                            [0.0,0.0,0.0]]), 

               tf.constant([[0.0,0.0,0.0],

                            [0.0,0.0,0.0],

                            [0.0,0.0,0.0]]))
qw_kappa_score(tf.constant([[0.0,0.0,0.0],

                            [0.0,0.0,0.0],

                            [0.0,0.0,0.0]]), 

               tf.constant([[0.0,0.0,0.0],

                            [0.0,0.0,0.0],

                            [0.0,0.0,0.0]])).numpy()