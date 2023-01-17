import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
! pip install pandas numpy scikit-learn matplotlib seaborn
#  import other packages



import pandas as pd

import numpy as np # Linear Algebra
# clone the dataset repository



! git clone https://github.com/education454/diabetes_dataset
! ls diabetes_dataset
# Load Data set



pdf = pd.read_csv('/kaggle/input/diabetes-data-set/diabetes-dataset.csv')
# Show first five rows from data set



pdf.head()
# pandas info method applied here

pdf.info()
#  show shape



pdf.shape
#  show how many patients has diabetes or not ? ? ?

 

print(pdf.Outcome.value_counts())

#  show summary statastics for all numeric columns



pdf.describe()
# Column wise na values counts



pdf.isna().sum()
# show how many column has 0 values and sum it



featureList = ['Glucose', 'BloodPressure',	'SkinThickness',	'Insulin',	'BMI']

print(pdf[featureList].isin({0}).sum())
# Set "0" to Mean values of perticular columns



for col in featureList:

  pdf[col] = pdf[col].replace({ 0 : pdf[col].mean() })
# show "0" values count

pdf[featureList].isin({0}).sum()
# Import visulazation packages



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

pdf.head()
plt.figure(figsize=(10, 5))

sns.barplot('Outcome', 'Pregnancies', data=pdf, ) 



plt.title("Bars of Outcome and Pregnancies")

plt.xlabel("Outcome")

plt.ylabel("Pregnancies")

plt.show()
plt.figure(figsize=(10, 7))



sns.heatmap(pdf.corr(), annot=True, linewidths=0.2, fmt='.1f', cmap='coolwarm') # cmap='RdYlBu'

plt.show()
# select feature colulmn

feature_columns = pdf[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']]



feature_columns.head()
# Outcome colummn

outcome_column = pdf['Outcome']

outcome_column.head()
# import package



from sklearn.model_selection import train_test_split
# split into training and testing sets



X_train, X_test, y_train, y_test = train_test_split( feature_columns, outcome_column, test_size=0.2, random_state=5) 

# show train data and shapes



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix
#  create model instance 



model = LogisticRegression()
#  Model Fitting



model = model.fit(X_train, y_train)
# set score

score = model.predict(X_train)
# Model Score



# print("Accuracy Score: ", accuracy_score(X_train, y_train))



print("Training Score: ", model.score(X_train, y_train))

print("Testing Score:  ", model.score(X_test, y_test))

# Model Accuracy

pred = model.predict(X_test)

print("Model Accuracy is : ", pred)
# Model Intercept

model.intercept_
# Model Coefficient



model.coef_
accuracy_score(y_test, pred)