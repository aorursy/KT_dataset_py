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
import pathlib

import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



print(tf.__version__)
#import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

raw_test = pd.read_csv("../input/titanic/test.csv")

raw_dataset = pd.read_csv("../input/titanic/train.csv")



dataset = raw_dataset.copy()

quizset = raw_test.copy()

provset = dataset.append(pd.concat([quizset,gender_submission['Survived']],axis=1,sort=False)).reset_index(drop=True)

datacount = len(dataset)

#dataset_y = dataset.pop('Survived')

#dataset_x = dataset



print('dataset:', len(dataset))  # DataFrame of training and validation

print('quizset:', len(quizset))  # DataFrame of quiz(Survived columns not exist)

print('provset: ',len(provset))  # DataFrame of provisioning (Survived columns NOT valid)
provset.dtypes
# Many data is missing in Age but Name's title(Mr,Mrs,Master,etc..) will fill this gap a little.

provset['Title'] = provset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



# Titles seem to determine their fate

pd.crosstab(provset['Title'],provset['Sex'])
# Age mean,median per Title

age_mean = provset.groupby(['Title'])['Age'].mean()

age_median = provset.groupby(['Title'])['Age'].median()

age_std = provset.groupby(['Title'])['Age'].std()

pd.DataFrame([age_mean,age_median,age_std],index=['mean','median','std'])
# Missing Data fill with median per Title

for title,subset in provset.groupby(['Title']):

    provset.loc[provset['Age'].isna() & (provset['Title'] == title), 'Age'] = age_median[title]

provset
# Cabin to Deck

provset['Deck'] = provset['Cabin'].str.extract('([A-Z])')



# Deck and Survived

pd.crosstab(provset['Deck'],provset['Pclass'])
# fill na class

provset.loc[provset['Embarked'].isna(), 'Embarked'] = 'O' # other class

#provset.loc[provset['Cabin'].isna(), 'Cabin'] = 'O' # other class

provset.loc[provset['Deck'].isna(), 'Deck'] = 'O' # other class
# add family_size

provset['FamilySize'] = provset['Parch'] + provset['SibSp'] + 1



# and add IsAlone

provset['IsAlone'] = 0

provset.loc[(provset['FamilySize'] == 1), 'IsAlone'] = 1

provset
# drop unreasonable columns

#provset = provset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

provset.tail()
# one hot encoding

#provset = pd.get_dummies(provset, columns=['Sex','Embarked','Pclass','Title', 'Deck'])

#provset
dataset = provset[:datacount]

dataset_labels = dataset.pop('Survived')



# drop unreasonable columns

dataset_x = dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

dataset_x
# quizset needs PassengerId

quizset = provset[datacount:].drop('Survived',axis=1)

quizset_x = quizset.drop(['Name', 'Ticket', 'Cabin'], axis=1)
provset.head()
provset.describe()
provset.shape, dataset_x.shape, dataset_labels.shape,quizset.shape
provset.Age.hist(bins=20)
provset['Pclass'].value_counts().plot(kind='barh')
provset['Embarked'].value_counts().plot(kind='barh')
provset.groupby('Sex').Survived.mean().plot(kind='barh').set_xlabel('% survive')
fc = tf.feature_column

CATEGORICAL_COLUMNS = ['Sex', 'Embarked', 'Pclass', 'Title', 'Deck', 'IsAlone' ]

NUMERIC_COLUMNS = ['Age', 'Fare', 'FamilySize']



def one_hot_cat_column(feature_name, vocab):

    return tf.feature_column.indicator_column(

        tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocab))



feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:

    # Need to one-hot encode categorical features.

    vocabulary = dataset_x[feature_name].unique()

    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))



for feature_name in NUMERIC_COLUMNS:

    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))
example = dict(dataset_x.head(1))

class_fc = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('Pclass', (1, 2, 3)))

print('Feature value: "{}"'.format(example['Pclass'].iloc[0]))

print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())

print('Feature value: "{}"'.format(example['IsAlone'].iloc[0]))

print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())

#dataset_x = dataset_x.head(1).fillna("O")
dataset_x
tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()
provset.shape, dataset.shape, quizset.shape # before data

dataset_x.shape, dataset_labels.shape # engineered data
msk = np.random.rand(len(dataset_x)) < 0.8

train_x = dataset_x[msk]

eval_x = dataset_x[~msk]

train_labels = dataset_labels[msk]

eval_labels = dataset_labels[~msk]
# Use entire batch since this is such a small dataset.

NUM_EXAMPLES = len(train_labels)



def make_input_fn(X, y, n_epochs=None, shuffle=True):

  def input_fn():

    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))

    if shuffle:

      dataset = dataset.shuffle(NUM_EXAMPLES)

    # For training, cycle thru dataset as many times as need (n_epochs=None).

    dataset = dataset.repeat(n_epochs)

    # In memory training doesn't use batching.

    dataset = dataset.batch(NUM_EXAMPLES)

    return dataset

  return input_fn



# Training and evaluation input functions.

train_input_fn = make_input_fn(train_x, train_labels)

eval_input_fn = make_input_fn(eval_x, eval_labels, shuffle=False, n_epochs=1)
# Since data fits into memory, use entire dataset per layer. It will be faster.

# Above one batch is defined as the entire dataset.

params = {

    'n_batches_per_layer': 1,

    'n_trees': 200, # 100

    'max_depth': 10, # 6

    'center_bias': True, # False

    'l2_regularization': 0.01

}

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)

#est = tf.estimator.BoostedTreesClassifier(feature_columns,1)

# The model will stop training once the specified number of trees is built, not

# based on the number of steps.

est.train(train_input_fn, max_steps=100)

# Eval.

result = est.evaluate(eval_input_fn)

print(pd.Series(result))
# Make predictions.

sns_colors = sns.color_palette('colorblind')

pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))

df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])

# Create DFC Pandas dataframe.

labels = eval_labels.values

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])

df_dfc.describe().T
from sklearn.metrics import roc_curve



fpr, tpr, _ = roc_curve(eval_labels, probs)

plt.plot(fpr, tpr)

plt.title('ROC curve')

plt.xlabel('false positive rate')

plt.ylabel('true positive rate')

plt.xlim(0,)

plt.ylim(0,)

plt.show()
# Sum of DFCs + bias == probabality.

bias = pred_dicts[0]['bias']

dfc_prob = df_dfc.sum(axis=1) + bias

np.testing.assert_almost_equal(dfc_prob.values,

                               probs.values)
pred_dicts = list(est.predict(eval_input_fn))

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])



probs.plot(kind='hist', bins=20, title='predicted probabilities')

plt.show()
# Boilerplate code for plotting :)

def _get_color(value):

    """To make positive DFCs plot green, negative DFCs plot red."""

    green, red = sns.color_palette()[2:4]

    if value >= 0: return green

    return red



def _add_feature_values(feature_values, ax):

    """Display feature's values on left of plot."""

    x_coord = ax.get_xlim()[0]

    OFFSET = 0.15

    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):

        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)

        t.set_bbox(dict(facecolor='white', alpha=0.5))

    from matplotlib.font_manager import FontProperties

    font = FontProperties()

    font.set_weight('bold')

    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',

    fontproperties=font, size=12)



def plot_example(example):

  TOP_N = 8 # View top 8 features.

  sorted_ix = example.abs().sort_values()[-TOP_N:].index  # Sort by magnitude.

  example = example[sorted_ix]

  colors = example.map(_get_color).tolist()

  ax = example.to_frame().plot(kind='barh',

                          color=[colors],

                          legend=None,

                          alpha=0.75,

                          figsize=(10,6))

  ax.grid(False, axis='y')

  ax.set_yticklabels(ax.get_yticklabels(), size=14)



  # Add feature values.

  _add_feature_values(eval_x.iloc[ID][sorted_ix], ax)

  return ax
# Plot results.

ID = 8

example = df_dfc.iloc[ID]  # Choose ith example from evaluation set.

TOP_N = 8  # View top 8 features.

sorted_ix = example.abs().sort_values()[-TOP_N:].index

ax = plot_example(example)

ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))

ax.set_xlabel('Contribution to predicted probability', size=14)

plt.show()
# Boilerplate plotting code.

def dist_violin_plot(df_dfc, ID):

  # Initialize plot.

  fig, ax = plt.subplots(1, 1, figsize=(10, 6))



  # Create example dataframe.

  TOP_N = 8  # View top 8 features.

  example = df_dfc.iloc[ID]

  ix = example.abs().sort_values()[-TOP_N:].index

  example = example[ix]

  example_df = example.to_frame(name='dfc')



  # Add contributions of entire distribution.

  parts=ax.violinplot([df_dfc[w] for w in ix],

                 vert=False,

                 showextrema=False,

                 widths=0.7,

                 positions=np.arange(len(ix)))

  face_color = sns_colors[0]

  alpha = 0.15

  for pc in parts['bodies']:

      pc.set_facecolor(face_color)

      pc.set_alpha(alpha)



  # Add feature values.

  _add_feature_values(eval_x.iloc[ID][sorted_ix], ax)



  # Add local contributions.

  ax.scatter(example,

              np.arange(example.shape[0]),

              color=sns.color_palette()[2],

              s=100,

              marker="s",

              label='contributions for example')



  # Legend

  # Proxy plot, to show violinplot dist on legend.

  ax.plot([0,0], [1,1], label='eval set contributions\ndistributions',

          color=face_color, alpha=alpha, linewidth=10)

  legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large',

                     frameon=True)

  legend.get_frame().set_facecolor('white')



  # Format plot.

  ax.set_yticks(np.arange(example.shape[0]))

  ax.set_yticklabels(example.index)

  ax.grid(False, axis='y')

  ax.set_xlabel('Contribution to predicted probability', size=14)
dist_violin_plot(df_dfc, ID)

plt.title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))

plt.show()
# quiz_data

quizset_x_copy = quizset_x.copy()

pid = quizset_x_copy.pop('PassengerId')



def input_fn(features,batch_size=256):

    #quiz_input_fn = make_input_fn(quizset_x, None)

    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)





# Generate predictions from the model

expected = [0, 1]



predictions = est.predict(input_fn=lambda: input_fn(quizset_x,1))



def get_pred_y():

    y = []

    for pred_dict in predictions:

        class_id = pred_dict['class_ids'][0]

        probability = pred_dict['probabilities'][class_id]

        y.append(expected[class_id])

    #return np.reshape(y,(-1,1))

    return y



submission = pd.DataFrame([np.array(pid),get_pred_y()],index=['PassengerId','Survived']).T

submission
!rm ./submission.csv

submission.to_csv('submission.csv',index=False)