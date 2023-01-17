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
import pandas as pd 

import matplotlib 

import numpy as np 

import scipy as sp 

import sklearn



#misc libraries

import random

import time

import math



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8
train = pd.read_csv('/kaggle/input/learn-together/train.csv')

test = pd.read_csv('/kaggle/input/learn-together/test.csv')
train.head()
train.tail()
train.shape, test.shape
#Combine train and test dataset

combined = train.append(test)

combined.reset_index(inplace=True)

combined.drop('index',inplace=True,axis=1)

combined.head()
#Types of the dataset

dtype_df = combined.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df
#Statistics of the data

combined.describe()
#Find out null values

missing_df = combined.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df['missing_ratio'] = (missing_df['missing_count'] / combined.shape[0]) * 100

missing_df = missing_df.loc[missing_df['missing_count'] > 0].sort_values(by=['missing_ratio'],ascending=False)

print(missing_df)
#Changed the Wilderness Area and Soil Type as categorical values for better EDA

Wilderness_Area = train.filter(like='Wilderness_Area')

Wilderness = pd.Series(Wilderness_Area.columns[np.where(Wilderness_Area!=0)[1]])

Soil_Type = train.filter(like='Soil_Type')

Soil = pd.Series(Soil_Type.columns[np.where(Soil_Type!=0)[1]])

train_eda = train.copy()

train_eda['Wilderness_Area'] = Wilderness

train_eda['Soil_Type'] = Soil

train_eda = train_eda.drop(Wilderness_Area.columns, axis=1)

train_eda = train_eda.drop(Soil_Type.columns, axis=1)

train_eda.head()
#Replace Wilderness Area to meaning value as provided in the documentation

train_eda.replace('Wilderness_Area1', 'Rawah', inplace=True)

train_eda.replace('Wilderness_Area2', 'Neota', inplace=True)

train_eda.replace('Wilderness_Area3', 'Comanche Peak', inplace=True)

train_eda.replace('Wilderness_Area4', 'Cache la Poudre', inplace=True)
#Convert Covert_Type to integer

train_eda['Cover_Type'] = train_eda['Cover_Type'].astype(int)

train_eda.head()
numeric_cols = train_eda.drop(['Cover_Type', 'Wilderness_Area', 'Soil_Type'], axis=1).columns

print(numeric_cols)

category_cols = ['Cover_Type', 'Wilderness_Area', 'Soil_Type']

print(category_cols)
# Let’s plot the distribution of each feature

def plot(dataset, typ, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width,height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataset.shape[1]) / cols)

    for i, column in enumerate(dataset.columns):

        ax = fig.add_subplot(rows, cols, i + 1)

        ax.set_title(column)

        if dataset.dtypes[column] == np.object:

            g = sns.countplot(y=column, data=dataset)

            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]

            g.set(yticklabels=substrings)

            plt.xticks(rotation=25)

        else:

            if typ == 'dist':

                g = sns.distplot(dataset[column])

            elif typ == 'box':

                g = sns.boxplot(dataset[column], orient='v')

            plt.xticks(rotation=25)
#Distribution plot on all features

plot(train_eda, 'dist', cols=3, width=20, height=20, hspace=0.75, wspace=0.2)
plt.style.use('seaborn-whitegrid')

sns.countplot(y=train_eda['Soil_Type']);
#Boxplot on numerical features

plot(train_eda[numeric_cols], 'box', cols=3, width=20, height=20, hspace=0.75, wspace=0.2)
#Correlation map to see how features are correlated 

corrmat = train_eda.corr()

plt.subplots(figsize=(15,12))

sns.heatmap(corrmat, vmax=0.9, square=True, cmap="YlGnBu")
correlation = train_eda.corr()

k= 11

cols = correlation.nlargest(k,'Cover_Type')['Cover_Type'].index

print(cols)

cm = np.corrcoef(train[cols].values.T)

f , ax = plt.subplots(figsize = (14,12))

sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',

            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,8))



sns.stripplot(x='Cover_Type', y='Elevation', data=train_eda, alpha=0.3, jitter=True, ax=axis1);

axis1.set_title('Elevation vs Cover Type Comparison')

axis1.set_ylabel('Elevation in meters')

sns.barplot(x='Cover_Type', y='Aspect', data=train_eda, ax=axis2);

axis2.set_title('Aspect vs Cover Type Comparison')

axis2.set_ylabel('Aspect in degrees azimuth')

sns.violinplot(x="Cover_Type", y="Slope", data=train_eda, inner=None, color=".8", ax=axis3)

sns.stripplot(x="Cover_Type", y="Slope", data=train_eda, jitter=True, ax=axis3)

axis3.set_title('Slope vs Cover Type Comparison')

axis3.set_ylabel('Slope in degrees')
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,8))



sns.stripplot(x='Cover_Type', y='Horizontal_Distance_To_Hydrology', data=train_eda, 

              alpha=0.3, jitter=True, ax=axis1);

axis1.set_title('Horizontal Distance to Hydrology  vs Cover Type')

axis1.set_ylabel('Horizontal Distance to Hydrology')

sns.stripplot(x='Cover_Type', y='Vertical_Distance_To_Hydrology', data=train_eda, 

              alpha=0.3, jitter=True, ax=axis2);

axis2.set_title('Vertical Distance to Hydrology vs Cover Type')

axis2.set_ylabel('Vertical Distance to Hydrology')

sns.stripplot(x='Cover_Type', y='Horizontal_Distance_To_Roadways', data=train_eda, 

              alpha=0.3, jitter=True, ax=axis3);

axis3.set_title('Horizontal Distance to Roadways vs Cover Type')

axis3.set_ylabel('Horizontal Distance to Roadways in degrees')
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,8))



sns.boxplot(x='Cover_Type', y='Hillshade_9am', data=train_eda, ax=axis1);

axis1.set_title('Hillshade 9am vs Cover Type')

axis1.set_ylabel('Hillshade index at 9am')

sns.boxplot(x='Cover_Type', y='Hillshade_Noon', data=train_eda, ax=axis2);

axis2.set_title('Hillshade Noon vs Cover Type')

axis2.set_ylabel('Hillshade index at noon')

sns.boxplot(x='Cover_Type', y='Hillshade_3pm', data=train_eda, ax=axis3);

axis3.set_title('Hillshade 3pm vs Cover Type')

axis3.set_ylabel('Hillshade index at 3pm')
ax = sns.violinplot(x="Cover_Type", y="Horizontal_Distance_To_Fire_Points", data=train_eda, 

                    inner=None, color=".8")

ax = sns.stripplot(x="Cover_Type", y="Horizontal_Distance_To_Fire_Points", data=train_eda, 

                   jitter=True)

ax.set_title('Hor Dist to Fire Points vs Cover Type')

ax.set_ylabel('Horz Dist to nearest wildfire ignition points')
f = sns.factorplot(x='Cover_Type', col='Wilderness_Area', kind='count', data=train_eda)

f.fig.set_size_inches(16, 8)
plt.figure(figsize=(10,20))

sns.countplot(y="Soil_Type", hue="Cover_Type", data=train_eda)

plt.show()
ax = sns.scatterplot(x="Hillshade_9am", y="Hillshade_3pm", hue="Cover_Type", legend="full", data=train_eda)
ax = sns.scatterplot(x="Horizontal_Distance_To_Hydrology", y="Vertical_Distance_To_Hydrology", \

                     hue="Cover_Type", legend="full", palette='Set1', data=train_eda)
ax = sns.scatterplot(x="Aspect", y="Hillshade_3pm", \

                     hue="Cover_Type", legend="full", palette = "RdBu", data=train_eda)
ax = sns.scatterplot(x="Hillshade_Noon", y="Hillshade_3pm", \

                     hue="Cover_Type", legend="full", palette = "winter_r", data=train_eda)
train_eda.skew(), train_eda.kurt()
sns.distplot(train_eda.skew(),color='blue',axlabel ='Skewness')
plt.figure(figsize = (12,8))

sns.distplot(train_eda.kurt(),color='r',axlabel ='Kurtosis',norm_hist= False, kde = True,rug = False)

plt.show()
sns.distplot(train['Cover_Type'], kde=False)
train.shape, test.shape
train_X, test_X, train_y, test_y = train_test_split(train.drop('Cover_Type', \

                        axis=1), train['Cover_Type'], random_state = 0)

train_X.shape, test_X.shape, train_y.shape, test_y.shape
rfc = RandomForestClassifier(n_estimators=300, random_state=42)

#rfc.fit(train_X, train_y)



# Train it on the training set

results = cross_val_score(rfc, train_X, train_y, cv=5)



# Evaluate the accuracy on the test set

print("CV accuracy score: {:.2f}%".format(results.mean()*100))
rfc.fit(train_X, train_y)

valid_pred = rfc.predict(test_X)

acc = accuracy_score(test_y, valid_pred)

print("Accuracy score: {:.2f}%".format(acc*100))
#Plotting confustion matrix

def plot_cm(y_true, y_pred, figsize=(10,10)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, cmap= "OrRd", annot=annot, fmt='', ax=ax)

    

plot_cm(test_y, valid_pred)
# make predictions 

test_preds = rfc.predict(test)



# save to submit

output = pd.DataFrame({'Id': test.Id,

                       'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)
output.head()