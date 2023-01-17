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

import pandas as pd 

import matplotlib.pyplot as plt      

%matplotlib inline 

import seaborn as sns
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df.shape
df.head()
 # Check if there are any null values in data set



df.isnull().values.any()
columns = list(df)[0:-1] 

df[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2)); 
df.corr()
def plot_corr(df, size=11):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)

    

plot_corr(df)
sns.pairplot(df,diag_kind='kde')
n_true = len(df.loc[df['Outcome'] == True])

n_false = len(df.loc[df['Outcome'] == False])

print("Number of true cases: {0} ({1:2.2f}%)".format(n_true, (n_true / (n_true + n_false)) * 100 ))

print("Number of false cases: {0} ({1:2.2f}%)".format(n_false, (n_false / (n_true + n_false)) * 100))
from sklearn.model_selection import train_test_split



X = df.drop('Outcome',axis=1)  # Predictor feature columns (8 X m)

Y = df['Outcome']              # Predicted class (1=True, 0=False) (1 X m)



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 1 is just any random seed number



x_train.head()
print("{0:0.2f}% data is in training set".format((len(x_train)/len(df.index)) * 100))

print("{0:0.2f}% data is in test set".format((len(x_test)/len(df.index)) * 100))
print("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(df.loc[df['Outcome'] == 1]), (len(df.loc[df['Outcome'] == 1])/len(df.index)) * 100))

print("Original Diabetes False Values   : {0} ({1:0.2f}%)".format(len(df.loc[df['Outcome'] == 0]), (len(df.loc[df['Outcome'] == 0])/len(df.index)) * 100))

print("")

print("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))

print("")

print("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))

print("")
x_train.head()
from sklearn.impute import SimpleImputer

rep_0 = SimpleImputer(missing_values=0, strategy="mean")

cols=x_train.columns

x_train = pd.DataFrame(rep_0.fit_transform(x_train))

x_test = pd.DataFrame(rep_0.fit_transform(x_test))



x_train.columns = cols

x_test.columns = cols



x_train.head()
from sklearn import metrics



from sklearn.linear_model import LogisticRegression



# Fit the model on train

model = LogisticRegression(solver="liblinear")

model.fit(x_train, y_train)

#predict on test

y_predict = model.predict(x_test)





coef_df = pd.DataFrame(model.coef_)

coef_df['intercept'] = model.intercept_

print(coef_df)
model_score = model.score(x_test, y_test)

print(model_score)
cm = metrics.confusion_matrix(y_test, y_predict, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)