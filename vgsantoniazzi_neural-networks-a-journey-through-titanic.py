import os
import tarfile
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer, LabelBinarizer
from datetime import datetime
from tensorflow.contrib.layers import fully_connected

%matplotlib inline
#
# Set Dataset path and the file name to load.
#
DATASET_PATH = '../input/'
FILE_NAME = 'train.csv'
#
# Load the dataset into Pandas (https://pandas.pydata.org/).
#
def load_dataset(dataset_path = DATASET_PATH, file_name = FILE_NAME):
    return pd.read_csv(os.path.join(dataset_path, file_name))

train_data = load_dataset()
test_data = load_dataset(DATASET_PATH, 'test.csv')
#
# Show the first 5 lines of the dataset. It's useful to start know more about all the data.
# Look at the data types, think about what you can split, others data to add. All here is useful?
#
train_data.head()
#
# Get data type, length of dataset, if there are some data missing and more useful info.
#
train_data.info()
#
# Lets see the correlation between all these variables. What really matters when you need to 
# predict `Survived`? (second column) 
#
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#
# Get some statistics about the current data. Mean, Std (http://www.purplemath.com/modules/meanmode.htm)
#
train_data.describe()
#
# Let's see some info about the amount of the data. How many survivors? What about Age and Fare?
#
train_data.hist(figsize=(20,15))
plt.show()
#
# Pclass and Age really matters for survivor people?
#
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=4, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#
# Let's think about what variables we can improve.
#
train_data.head()
#
# Some people don't have a cabin (workers) So, let's add another attribute to take care of this people.
#
train_data['HasCabin'] = train_data["Cabin"].map(lambda x: 0 if type(x) == float else 1)
test_data['HasCabin'] = test_data["Cabin"].map(lambda x: 0 if type(x) == float else 1)


#
# Some registers is about the entire family. Let's add another attribute to know the family size.
#
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

#
# When people are alone, the family size is 1. Let's add this attribute to know after if this attribute correlates with survived.
#
train_data['IsAlone'] = train_data["FamilySize"].map(lambda x: 1 if x == 1 else 0)
test_data['IsAlone'] = test_data["FamilySize"].map(lambda x: 1 if x == 1 else 0)

#
# Some people in Titanic are a child. We know women and children have the preference when got saved.
# Let's add this attribute and see the correlation after.
#
train_data['IsChild'] = train_data["Age"].map(lambda x: 1 if x < 16 else 0)
test_data['IsChild'] = test_data["Age"].map(lambda x: 1 if x < 16 else 0)
#
# Let's see how are our data now.
#
train_data.head(10)
#
# Let's correlate all variables. Yellow ones are highly correlated.
#
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#
# Ok, now we need to split our data to start doing some machine learning.
# Our labels are the `survived` column, labels are the data that we need to predict based on another.
#
# PassengerId we don't need to know. We put in another variable to use after all.
# Name is not useful to predict.
# Ticket number don't tell us anything.
# Cabin is expensive to know. Probably we can split the cabin into classes also: Class A, Class B. This is useful?
#
labels = train_data['Survived']
passenger_id = test_data['PassengerId']

train_data = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
train_data.info()
#
# How are our training data?
#
train_data[:2]
#
# To predict info with machine learning, we need to transform our data to put on our logic (Classifier, NeuralNetworks, etc)
# We need to remove from dataframe and get the raw value, Fill all empty values and Split our string into classes.
# We also Scale our values to do all math without got so expensive. (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
#
# The following 4 code blocks do this.
#
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attributes_names].values
class FillNa(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names, value):
        self.attributes_names = attributes_names
        self.value = value
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X[self.attributes_names].fillna(self.value, inplace=True)
        return X
class StringBinalizer(BaseEstimator, TransformerMixin):    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        label = LabelBinarizer()
        return label.fit_transform(X)
numerical_attributes = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'HasCabin', 'FamilySize', 'IsAlone']

numerical_pipeline = Pipeline([
    ('selector', DataFrameSelector(numerical_attributes)),
    ('imputer', Imputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

sex_pipeline = Pipeline([
    ('selector_sex', DataFrameSelector('Sex')),
    ('label_encoder', StringBinalizer())
])

embarked_pipeline = Pipeline([
    ('fill_na', FillNa('Embarked', 'C')),
    ('selector_embarked', DataFrameSelector('Embarked')),
    ('label_binalizer', StringBinalizer())

])

full_pipeline = FeatureUnion(transformer_list=[
    ('numerical_pipeline', numerical_pipeline),
    ('sex_pipeline', sex_pipeline),
    ('embarked_pipeline', embarked_pipeline)
])
#
# Now, our data is raw, standardized, only with useful data to Classifier.
#
train_data_prepared = full_pipeline.fit_transform(train_data)
test_data_prepared = full_pipeline.fit_transform(test_data)

print(train_data_prepared[:10])
# Testing Deep Neural Networks

# This Neural Network have 1 Input layer, 3 Hidden Layers and 1 Output Layer

# Try edit it's hiperparameters, add more hidden layers, train more epochs
n_inputs = 12
n_hidden = 144
n_outputs = 2

learning_rate = 0.1

n_epochs = 20

X = tf.placeholder(tf.float32, shape=(None, n_inputs))
y = tf.placeholder(tf.int32, shape=(None))

with tf.name_scope("dnn"):
    hidden_1 = fully_connected(X, n_hidden)
    hidden_2 = fully_connected(hidden_1, n_hidden)
    hidden_3 = fully_connected(hidden_2, n_hidden)
    logits = fully_connected(hidden_3, n_outputs, activation_fn=None)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy)
    loss_summary = tf.summary.scalar('Loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    
    for epoch in range(n_epochs):
        X_batch = train_data_prepared
        y_batch = labels
        sess.run(training, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        if epoch % 10 == 0:
            print(epoch, "Train Accuracy: ", acc_train)
    
    save_path = saver.save(sess, './final.ckpt')
#
# Restore last session and evaluate all predictions.
#
with tf.Session() as sess:
    saver.restore(sess, "./final.ckpt")
    X_batch = test_data_prepared
    z = logits.eval(feed_dict={X: X_batch})
    predictions = np.argmax(z, axis=1)
#
# We put all our predictions to csv to put in Kaggle platform. Send your csv and see how well is your model.
#
result = pd.DataFrame({ 'PassengerId': passenger_id, 'Survived': predictions })
result.to_csv("../03_neural_networks_part_result.csv", index=False)
