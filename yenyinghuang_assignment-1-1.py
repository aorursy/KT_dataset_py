# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import tensorflow as tf
from tensorflow.python.data import Dataset

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv", sep=",")
data.head(10)
from sklearn.model_selection import train_test_split

train_target = data.SalePrice

train_x, test_x, train_y, test_y = train_test_split(data, train_target, test_size=0.3)
def my_input_fn(features, targets, batch_size=1, num_epochs=None):
   
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
linear_regressor = tf.estimator.LinearRegressor(feature_columns=train_x)

predict_training_input_fn = lambda: my_input_fn(
      train_x, 
      train_y, 
      num_epochs=1, 
      shuffle=False)

linear_regressor.train(
        input_fn=training_input_fn
    )
