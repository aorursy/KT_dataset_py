# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#../input/test.csv

#../input/train.csv





data_train = pd.read_csv('../input/train.csv')

data_train.head()

# Data sets

IRIS_TRAINING = "../input/train.csv"

IRIS_TEST = "../input/test.csv"



with open("../input/test.csv",'r') as f:

    with open("updated_test.csv",'w') as f1:

        f.next() # skip header line

        for line in f:

            f1.write(line)

            

            
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import tensorflow as tf

import numpy as np



# Data sets

IRIS_TRAINING = "../input/train.csv"

IRIS_TEST = "../input/test.csv"



# Load datasets.

training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING,

                                                       target_dtype=np.int)

test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST,

                                                   target_dtype=np.int)



# Specify that all features have real-value data

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=132)]



# Build 3 layer DNN with 10, 20, 10 units respectively.

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,

                                            hidden_units=[10, 20, 10],

                                            n_classes=131,

                                            model_dir="/tmp/iris_model")



# Fit model.

classifier.fit(x=training_set.data, 

               y=training_set.target, 

               steps=2000)



# Evaluate accuracy.

accuracy_score = classifier.evaluate(x=test_set.data,

                                     y=test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))



# Classify two new flower samples.

#new_samples = np.array(

#    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)

#y = classifier.predict(new_samples)

#print('Predictions: {}'.format(str(y)))