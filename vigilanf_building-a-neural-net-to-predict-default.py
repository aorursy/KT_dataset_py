#%matplotlib inline

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import numpy as np

import pandas as pd

import itertools

from sklearn import preprocessing

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import metric_spec

from tensorflow.contrib.learn.python.learn.estimators import _sklearn

from tensorflow.contrib.learn.python.learn.estimators import estimator

from tensorflow.contrib.learn.python.learn.estimators import model_fn

from tensorflow.python.framework import ops

from tensorflow.python.saved_model import loader

from tensorflow.python.saved_model import tag_constants

from tensorflow.python.util import compat

tf.logging.set_verbosity(tf.logging.DEBUG) # FATAL, WARN, INFO, DEBUG

df = pd.read_csv("../input/loan.csv", low_memory=False)
df['Default_Binary'] = int(0)

for index, value in df.loan_status.iteritems():

    if value == 'Default':

        df.set_value(index,'Default_Binary',int(1))

    if value == 'Charged Off':

        df.set_value(index, 'Default_Binary',int(1))

    if value == 'Late (31-120 days)':

        df.set_value(index, 'Default_Binary',int(1))    

    if value == 'Late (16-30 days)':

        df.set_value(index, 'Default_Binary',int(1))

    if value == 'Does not meet the credit policy. Status:Charged Off':

        df.set_value(index, 'Default_Binary',int(1))    
df['Purpose_Cat'] = int(0) 

for index, value in df.purpose.iteritems():

    if value == 'debt_consolidation':

        df.set_value(index,'Purpose_Cat',int(1))

    if value == 'credit_card':

        df.set_value(index, 'Purpose_Cat',int(2))

    if value == 'home_improvement':

        df.set_value(index, 'Purpose_Cat',int(3))    

    if value == 'other':

        df.set_value(index, 'Purpose_Cat',int(4))    

    if value == 'major_purchase':

        df.set_value(index,'Purpose_Cat',int(5))

    if value == 'small_business':

        df.set_value(index, 'Purpose_Cat',int(6))

    if value == 'car':

        df.set_value(index, 'Purpose_Cat',int(7))    

    if value == 'medical':

        df.set_value(index, 'Purpose_Cat',int(8))   

    if value == 'moving':

        df.set_value(index, 'Purpose_Cat',int(9))    

    if value == 'vacation':

        df.set_value(index,'Purpose_Cat',int(10))

    if value == 'house':

        df.set_value(index, 'Purpose_Cat',int(11))

    if value == 'wedding':

        df.set_value(index, 'Purpose_Cat',int(12))    

    if value == 'renewable_energy':

        df.set_value(index, 'Purpose_Cat',int(13))     

    if value == 'educational':

        df.set_value(index, 'Purpose_Cat',int(14))  
df_train = pd.get_dummies(df.purpose).astype(int)



df_train.columns = ['debt_consolidation','credit_card','home_improvement',

                     'other','major_purchase','small_business','car','medical',

                     'moving','vacation','house','wedding','renewable_energy','educational']



# Also add the target column we created at first

df_train['Default_Binary'] = df['Default_Binary']

df_train.head()
x = np.array(df.int_rate.values).reshape(-1,1) 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df['int_rate_scaled'] = pd.DataFrame(x_scaled)

print (df.int_rate_scaled[0:5])
x = np.array(df.funded_amnt.values).reshape(-1,1) 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df['funded_amnt_scaled'] = pd.DataFrame(x_scaled)

print (df.funded_amnt_scaled[0:5])
df_train['int_rate_scaled'] = df['int_rate_scaled']

df_train['funded_amnt_scaled'] = df['funded_amnt_scaled']
# Note for future, need to add an estimator function for features

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/wide_n_deep_tutorial.py



training_set = df_train[0:500000] # Train on first 500k rows

testing_set = df_train[500001:800000] # Test on next 400k rows

prediction_set = df_train[800001:] # Predict on final ~87k rows



COLUMNS = ['debt_consolidation','credit_card','home_improvement',

           'other','major_purchase','small_business','car','medical',

           'moving','vacation','house','wedding','renewable_energy','educational',

           'funded_amnt_scaled','int_rate_scaled','Default_Binary']   



FEATURES = ['debt_consolidation','credit_card','home_improvement',

           'other','major_purchase','small_business','car','medical',

           'moving','vacation','house','wedding','renewable_energy','educational',

           'funded_amnt_scaled','int_rate_scaled'] 



#CONTINUOUS_COLUMNS = ['funded_amnt_scaled','int_rate_scaled'] 

#CATEGORICAL_COLUMNS = ['Purpose_Cat']



LABEL = 'Default_Binary'



def input_fn(data_set):

    ### Simple Version ######

    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES} # Working method for continous data DO NOT DELETE 

    labels = tf.constant(data_set[LABEL].values)

    return feature_cols, labels

    

    """

     # Creates a dictionary mapping from each continuous feature column name (k) to

    # the values of that column stored in a constant Tensor.

    continuous_cols = {k: tf.constant(data_set[k].values)

                     for k in CONTINUOUS_COLUMNS}

    # Creates a dictionary mapping from each categorical feature column name (k)

    # to the values of that column stored in a tf.SparseTensor.

    categorical_cols = {k: tf.SparseTensor(

        indices=[[i, 0] for i in range(data_set[k].size)],

        values=data_set[k].values,

        shape=[data_set[k].size, 1])

                      for k in CATEGORICAL_COLUMNS}

    # Merges the two dictionaries into one.

    #feature_cols = dict(continuous_cols.items() + categorical_cols.items()) # Throws error

    feature_cols = dict(continuous_cols)

    feature_cols.update(categorical_cols)

    # Converts the label column into a constant Tensor.

    labels = tf.constant(data_set[LABEL].values)

    return feature_cols, labels

    """
learning_rate = 0.01

feature_cols = [tf.contrib.layers.real_valued_column(k)

              for k in FEATURES]

#config = tf.contrib.learn.RunConfig(keep_checkpoint_max=1) ######## DO NOT DELETE

regressor = tf.contrib.learn.DNNRegressor(

                    feature_columns=feature_cols,

                    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),

                    hidden_units=[10, 20, 10], )



regressor.fit(input_fn=lambda: input_fn(training_set), steps=500)
# Score accuracy

ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)

loss_score = ev["loss"]

print("Loss: {0:f}".format(loss_score))
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))

predictions = list(itertools.islice(y, 87378))
plt.plot(prediction_set.int_rate_scaled, predictions, 'ro')

plt.ylabel("Model Prediction Value")

plt.xlabel("Interest Rate of Loan (Scaled between 0-1)")

plt.show()
plt.plot(prediction_set.funded_amnt_scaled, predictions, 'ro')

plt.ylabel("Model Prediction Value")

plt.xlabel("Funded Amount of Loan (Scaled between 0-1)")

plt.show()
plt.plot(prediction_set.Purpose_Cat, predictions, 'ro')

plt.ylabel("Model Prediction Value")

plt.xlabel("Loan Purpose")

plt.title("DNN Regressor Predicting Default By Loan Purpose")

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 8

fig_size[1] = 8

plt.rcParams["figure.figsize"] = fig_size

labels = ['Debt Consolidation', 'Credit Card', 'Home Improvement', 'Other',

         'Major Purchase', 'Small Business', 'Car', 'Medical',

         'Moving', 'Vacation', 'House', 'Wedding',

         'Renewable Energy']



plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14], labels, rotation='vertical')



plt.show()