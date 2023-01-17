# Get the pydotplus package

# !pip install pydotplus
# Standard Libraries

import os

import numpy as np 

import pandas as pd 



# Visualization libraries

#import pydotplus

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style({"axes.facecolor": ".95"})



# Modeling and Machine Learning

from sklearn.manifold import TSNE

from sklearn.metrics import accuracy_score

from sklearn.externals.six import StringIO  

from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.linear_model import LogisticRegression





# Specify Paths

BASE_PATH = '../input/Kannada-MNIST/'

TRAIN_PATH = BASE_PATH + 'train.csv'

TEST_PATH = BASE_PATH + 'test.csv'



# Seed for reproducability

seed = 1234

np.random.seed(seed)
# File sizes and specifications

print('\n# Files and file sizes')

for file in os.listdir(BASE_PATH):

    print('{}| {} MB'.format(file.ljust(30), 

                             str(round(os.path.getsize(BASE_PATH + file) / 1000000, 2))))
# Load in training and testing data

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)

test_df.rename(columns={'id':'label'}, inplace=True)

concat_df = pd.concat([train_df, test_df])

sample_sub = pd.read_csv(BASE_PATH + 'sample_submission.csv');
def acc(y_true, y_pred):

    """

        Calculates the accuracy score between labels and predictions.

        

        :param y_true: The true labels of the data

        :param y_pred: The predictions for the data

        

        :return: a float denoting the accuracy

    """

    return round(accuracy_score(y_true, y_pred) * 100, 2)
test_df.head()
test_df.shape
train_df.head()
features = [col for col in train_df.columns if col.startswith('pixel')]

tsvd = TruncatedSVD(n_components=50).fit_transform(concat_df[features])

tsne = TSNE(n_components=3)

transformed = tsne.fit_transform(tsvd)  

# Split up the t-SNE results in training and testing data

tsne_train = pd.DataFrame(transformed[:len(train_df)], columns=['component1', 'component2', 'component3'])

tsne_test = pd.DataFrame(transformed[len(train_df):], columns=['component1', 'component2', 'component3'])
# Perform another split for t-sne feature validation

X_train, X_val, y_train, y_val = train_test_split(tsne_train, 

                                                  train_df['label'], 

                                                  test_size=0.25, 

                                                  random_state=seed)

# Train model with t-sne features

clf = DecisionTreeClassifier(max_depth=10, random_state=seed)

clf.fit(X_train, y_train)
train_preds = clf.predict(X_train)

val_preds = clf.predict(X_val)

acc_tsne_train = acc(train_preds, y_train)

acc_tsne_val = acc(val_preds, y_val)

print(f'Training accuracy with t-SNE features: {acc_tsne_train}%')

print(f'Validation accuracy with t-SNE features: {acc_tsne_val}%')
# Make predictions and save submission file

predictions = clf.predict(tsne_test)

sample_sub['label'] = predictions

sample_sub.to_csv('submission.csv', index=False)