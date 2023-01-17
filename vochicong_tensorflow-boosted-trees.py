# !pip install tf-nightly-gpu-2.0-preview
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

# tf.enable_eager_execution()

from tensorflow import feature_column as fc



dftrain = pd.read_csv('/kaggle/input/train.csv')

dftest = pd.read_csv('/kaggle/input/test.csv')

dftrain.columns
dftrain.columns.equals(dftest.columns)
dfeval = dftrain.sample(dftrain.shape[0]//10, random_state=0)

dftrain_all = dftrain.copy()

dftrain = dftrain.drop(dfeval.index)



y_eval = dfeval.pop('Survived')

y_train_all = dftrain_all.pop('Survived')

y_train = dftrain.pop('Survived')



dfs = [dftrain_all, dftrain, dfeval, dftest]
dftrain.shape, dfeval.shape, dftest.shape
for df in dfs:

    df['FamilySize'] = df.SibSp + df.Parch + 1

    df['Alone'] = (df.FamilySize == 1)*1
dftrain.info()
dftrain.isna().sum()
dftrain_age_mean = dftrain.Age.mean()

for df in dfs:

    df.Age.fillna(dftrain_age_mean, inplace=True)

    print(df.Embarked.unique(), df.Embarked.mode()[0])

    df.Embarked.fillna(dftrain.Embarked.mode()[0], inplace=True)
CATEGORICAL_COLUMNS = ['Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked', 'Alone', 'FamilySize']

NUMERIC_COLUMNS = ['Age', 'Fare']

DROP_COLUMNS = dftrain.columns.drop(CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)

DROP_COLUMNS
def drop_columns(df):

    return df.drop(DROP_COLUMNS, axis=1)
dftrain.Age.describe()
dftrain.Fare.describe()
feature_columns = []



# for feature_name in NUMERIC_COLUMNS:

#   feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))



feature_columns.append(fc.bucketized_column(

    fc.numeric_column('Age'), list(range(5,80,10))

))



feature_columns.append(fc.bucketized_column(

    fc.numeric_column('Fare'), [7,13,30]

))



def one_hot_cat_column(feature_name, vocab):

  return fc.indicator_column(

      fc.categorical_column_with_vocabulary_list(

          feature_name, vocab)

  )



for feature_name in CATEGORICAL_COLUMNS:

  vocabulary = dftrain[feature_name].unique()

  feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

  

# Use entire batch since this is such a small dataset.

NUM_EXAMPLES = len(y_train)

BATCH_SIZE = 64*2

# BATCH_SIZE = NUM_EXAMPLES



def make_input_fn(X, y, n_epochs=None, shuffle=True):

  def input_fn():

    inputs = (dict(X), y)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:

      dataset = dataset.shuffle(NUM_EXAMPLES)

    # For training, cycle thru dataset as many times as need (n_epochs=None)

    dataset = dataset.repeat(n_epochs)

    dataset = dataset.batch(BATCH_SIZE)

    return dataset

  return input_fn



# Training and evaluation input functions.

train_input_fn = make_input_fn(drop_columns(dftrain), y_train)

eval_input_fn = make_input_fn(drop_columns(dfeval), y_eval, shuffle=False, n_epochs=1)
linear_est = tf.estimator.LinearClassifier(feature_columns)



# Train model.

linear_est.train(train_input_fn, max_steps=100)



# Evaluation.

result = linear_est.evaluate(eval_input_fn)

result
import math

n_batches = NUM_EXAMPLES/BATCH_SIZE

print(n_batches)

n_batches = int(math.ceil(n_batches))

n_batches
def create_est():

    return tf.estimator.BoostedTreesClassifier(

        feature_columns,

        n_batches_per_layer=n_batches,

        n_trees=150,

        max_depth=4,

        learning_rate=0.1,

    )

est = create_est()

est.train(train_input_fn, max_steps=1000)

est.evaluate(eval_input_fn)
est = create_est()

est.train(make_input_fn(drop_columns(dftrain_all), y_train_all), max_steps=1000)

est.evaluate(eval_input_fn)
test_input_fn = make_input_fn(drop_columns(dftest), dftest.index, shuffle=False, n_epochs=1)

preds = est.predict(test_input_fn)

preds = [pred['class_ids'][0] for pred in preds]

pd.DataFrame({'PassengerId': dftest.PassengerId, 'Survived': preds}).to_csv('submission.csv', index=False)

!head submission.csv