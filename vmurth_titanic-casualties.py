# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#@title Run on TensorFlow 2.x

#%tensorflow_version 2.x



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_df = pd.read_csv('/kaggle/input/titanic/train.csv');

test_df = pd.read_csv('/kaggle/input/titanic/test.csv');





titanic_df.describe()

#titanic_df.head()
#Variable Notes

#pclass: A proxy for socio-economic status (SES)

#1st = Upper

#2nd = Middle

#3rd = Lower



#age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5



#sibsp: The dataset defines family relations in this way...

#Sibling = brother, sister, stepbrother, stepsister

#Spouse = husband, wife (mistresses and fiancés were ignored)



#parch: The dataset defines family relations in this way...

#Parent = mother, father

#Child = daughter, son, stepdaughter, stepson

#Some children travelled only with a nanny, therefore parch=0 for them.



#titanic_df.loc[titanic_df['Name'].str.contains("Jack")]

titanic_df.columns

test_df.info()
a = sorted(titanic_df['Fare'].unique())

#pd.value_counts(titanic_df['Fare'])

#print(a)



titanic_df.loc[titanic_df['Fare'] < 10]
#scatter plots

fig, ax = plt.subplots()

titanic_df.plot.scatter(x='Fare', y='Age', c='Survived', colormap='winter', ax=ax)
#Z score calculation

def zscore(df_col):

    return (df_col - df_col.mean())/df_col.std()

#replacing null values with average values

titanic_df['Age'].fillna((titanic_df['Age'].mean()), inplace=True)

titanic_df['Fare'].fillna((titanic_df['Fare'].mean()), inplace=True)

test_df['Age'].fillna((test_df['Age'].mean()), inplace=True)

test_df['Fare'].fillna((test_df['Fare'].mean()), inplace=True)





#Normalize features using Z score

titanic_df['Age_Z'] = zscore(titanic_df['Age'])

titanic_df['Fare_Z'] = zscore(titanic_df['Fare'])

test_df['Age_Z'] = zscore(test_df['Age'])

test_df['Fare_Z'] = zscore(test_df['Fare'])



#Convert the Gender columns from strings to to 0:Man 1:Woman

titanic_df.replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

test_df.replace(to_replace=['male', 'female'], value=[0, 1], inplace=True)

import ipywidgets as widgets

from ipywidgets import interactive, interact_manual



#import matplotlib.pyplot as plt



def scatterplot(df):



    #fig, ax = plt.subplots()

    

    def makeplot(f1, f2, b):

        

        #print(df.head(250))

        df[b].replace({True: 1, False: 0}, inplace=True)

        df.plot.scatter(x=f1, \

                        y=f2, \

                        c=b, \

                        colormap='winter', \

                        #ax=ax

                       )

        plt.show()



    iplot = interactive(makeplot, \

                        f1=list(df.select_dtypes('number').columns), \

                        f2=list(df.select_dtypes('number').columns), \

                        b=list(df.columns))

    output = iplot.children[-1]

    output.layout.height = '550px'

    return iplot



scatterplot(titanic_df.reindex())
scatterplot(test_df.reindex())
#histograms



plt.rcParams["figure.figsize"] = (15, 10) # (w, h)

fig, axes = plt.subplots(nrows=2, ncols=2)

ax0, ax1, ax2, ax3 = axes.flatten()



expTransform = 1

lived = titanic_df.loc[titanic_df['Survived'] == 1, 'Age_Z'];

died = titanic_df.loc[titanic_df['Survived'] == 0, 'Age_Z'];

ax0.hist([lived**expTransform,died**expTransform], stacked=True)

ax0.legend(['Lived', 'Died'])

ax0.set_title("Train Age_Z")





#pclass: A proxy for socio-economic status (SES)

#1st = Upper

#2nd = Middle

#3rd = Lower

lived = titanic_df.loc[titanic_df['Survived'] == 1, 'Pclass'];

died = titanic_df.loc[titanic_df['Survived'] == 0, 'Pclass'];

ax1.hist([lived,died], stacked=True)

ax1.legend(['Lived', 'Died'])

ax1.set_title("Train Class (1 = Highest)")



#TODO: bin these

expTransform = 1

lived = titanic_df.loc[titanic_df['Survived'] == 1, 'Fare_Z'];

died = titanic_df.loc[titanic_df['Survived'] == 0, 'Fare_Z'];

ax2.hist([lived**expTransform,died**expTransform], stacked=True, bins=100)

ax2.legend(['Lived', 'Died'])

ax2.set_title("Train Fare_Z")



lived = titanic_df.loc[titanic_df['Survived'] == 1, 'Sex'];

died = titanic_df.loc[titanic_df['Survived'] == 0, 'Sex'];

ax3.hist([lived,died], stacked=True)

ax3.legend(['Lived', 'Died'])

ax3.set_title("Train Gender")

#histograms



plt.rcParams["figure.figsize"] = (15, 10) # (w, h)

fig, axes = plt.subplots(nrows=2, ncols=2)

ax0, ax1, ax2, ax3 = axes.flatten()



expTransform = 1

lived = test_df.loc[titanic_df['Survived'] == 1, 'Age_Z'];

died = test_df.loc[titanic_df['Survived'] == 0, 'Age_Z'];

ax0.hist([lived**expTransform,died**expTransform], stacked=True)

ax0.legend(['Lived', 'Died'])

ax0.set_title("Test Age_Z")





#pclass: A proxy for socio-economic status (SES)

#1st = Upper

#2nd = Middle

#3rd = Lower

lived = test_df.loc[titanic_df['Survived'] == 1, 'Pclass'];

died = test_df.loc[titanic_df['Survived'] == 0, 'Pclass'];

ax1.hist([lived,died], stacked=True)

ax1.legend(['Lived', 'Died'])

ax1.set_title("Test Class (1 = Highest)")



#TODO: bin these

expTransform = 1

lived = test_df.loc[titanic_df['Survived'] == 1, 'Fare_Z'];

died = test_df.loc[titanic_df['Survived'] == 0, 'Fare_Z'];

ax2.hist([lived**expTransform,died**expTransform], stacked=True, bins=100)

ax2.legend(['Lived', 'Died'])

ax2.set_title("Test Fare_Z")



lived = test_df.loc[titanic_df['Survived'] == 1, 'Sex'];

died = test_df.loc[titanic_df['Survived'] == 0, 'Sex'];

ax3.hist([lived,died], stacked=True)

ax3.legend(['Lived', 'Died'])

ax3.set_title("Test Gender")
titanic_df.info()

test_df.info()



#X_train = titanic_df[['Sex', 'Pclass', 'Age_Z', 'Fare_Z']]

X_train = titanic_df[['Sex', 'Pclass', 'Fare_Z']]

y_train = titanic_df['Survived']

#X_test = test_df[['Sex', 'Pclass', 'Age_Z', 'Fare_Z']]

X_test = test_df[['Sex', 'Pclass', 'Fare_Z']]



#X_test = X_test.dropna(axis=0)



X_train.info()

X_test.info()
from sklearn.linear_model import LogisticRegression



logmodel = LogisticRegression()



logmodel.fit(X_train, y_train)



#titanic_df.info()



predictions = logmodel.predict(X_test)

print(predictions)
#import tensorflow as tf

#from tensorflow import feature_column

#from tensorflow.keras import layers



# Create an empty list that will eventually hold all created feature columns.

#feature_columns = []



# Create a numerical feature column to represent median_income.

#median_income = tf.feature_column.numeric_column("median_income")

#feature_columns.append(median_income)



# Create a numerical feature columns.

#tr = tf.feature_column.numeric_column('age_z')

#feature_columns.append(tr)

#tr = tf.feature_column.numeric_column('fare_z')

#feature_columns.append(tr)

#tr = tf.feature_column.numeric_column('age_z')

#feature_columns.append(tr)

                                      

                                      

# Convert the list of feature columns into a layer that will later be fed into

# the model. 

#feature_layer = layers.DenseFeatures(feature_columns)



# Print the first 3 and last 3 rows of the feature_layer's output when applied

# to train_df_norm:

#feature_layer(dict(train_df_norm))
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")