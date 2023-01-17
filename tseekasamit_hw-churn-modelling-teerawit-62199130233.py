import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
df.info()
df.head(10)
df.describe(include='number')
df.describe(include='object')
#check NULL

print(df.isnull().sum().sum())

print(df.isnull().any())

import missingno as msno

msno.matrix(df)
df.dtypes
df.shape
#numberic_columns

f_columns = ['CreditScore', 'Age', 'Tenure', 'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']



X = df.loc[:, f_columns].values

y = df.iloc[:, -1].values

print(X.shape, y.shape)
#make train/test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



X_train.shape, y_train.shape, X_test.shape, y_test.shape
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X= StandardScaler()

X_train= sc_X.fit_transform(X_train)

X_test= sc_X.transform(X_test)
#Fitting K-Nearest Neighbor Classification to the Training Set

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors =5,metric = 'minkowski',p=2)

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)



# Predicting the Test Set results

from sklearn.metrics import confusion_matrix

result = confusion_matrix(y_pred,y_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_KN = confusion_matrix(y_test, y_pred)

cm_KN
#Performace with confusion metrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_KN = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



print("Performance for K-Nearest Neighbor :")

print("\n Accuracy = " + str(accuracy_KN*100), '%')

print("\n Precision = " + str(precision*100),'%')

print("\n Recall = " + str(recall*100),'%')

print("\n f1 = " + str(f1*100),'%')
# Fitting SVM to the Training Set

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', degree = 3, random_state = 0) #degree for non-linear

classifier.fit(X_train, y_train) 



# Predicting the Test Set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_SVM_linear = confusion_matrix(y_test, y_pred)

cm_SVM_linear
#Performace with confusion metrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_li = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



print("Performance for Support Vector Machine (Linear) :")

print("\n Accuracy = " + str(accuracy_li*100), '%')

print("\n Precision = " + str(precision*100),'%')

print("\n Recall = " + str(recall*100),'%')

print("\n f1 = " + str(f1*100),'%')
# Fitting SVM to the Training Set

from sklearn.svm import SVC

classifier = SVC(kernel = 'poly', degree = 5, random_state = 0) #degree for non-linear

classifier.fit(X_train, y_train) 



# Predicting the Test Set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_SVM_poly = confusion_matrix(y_test, y_pred)

cm_SVM_poly
#Performace with confusion metrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_poly = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



print("Performance for Support Vector Machine (Poly) :")

print("\n Accuracy = " + str(accuracy_poly*100), '%')

print("\n Precision = " + str(precision*100),'%')

print("\n Recall = " + str(recall*100),'%')

print("\n f1 = " + str(f1*100),'%')
# Fitting Naive Bayes to the Training Set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)



# Predicting the Test Set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm_NB = confusion_matrix(y_test, y_pred)

cm_NB
#Performace with confusion metrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_nb = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



print("Performance for Naive Bayes :")

print("\n Accuracy = " + str(accuracy_nb*100), '%')

print("\n Precision = " + str(precision*100),'%')

print("\n Recall = " + str(recall*100),'%')

print("\n f1 = " + str(f1*100),'%')
# set font

plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = 'Helvetica'



# set the style of the axes and the text color

plt.rcParams['axes.edgecolor']='#333F4B'

plt.rcParams['axes.linewidth']=0.8

plt.rcParams['xtick.color']='#333F4B'

plt.rcParams['ytick.color']='#333F4B'

plt.rcParams['text.color']='#333F4B'



# create some fake data

percentages = pd.Series([accuracy_KN,accuracy_li,accuracy_poly,accuracy_nb], 

                        index=['K-Nearest Neighbor','SVM Liner','SVM Poly','Naive Bayes'])

df = pd.DataFrame({'percentage' : percentages})

df = df.sort_values(by='percentage')



# we first need a numeric placeholder for the y axis

my_range=list(range(1,len(df.index)+1))



fig, ax = plt.subplots(figsize=(5,3.5))



# create for each expense type an horizontal line that starts at x = 0 with the length 

# represented by the specific expense percentage value.

plt.hlines(y=my_range, xmin=0, xmax=df['percentage'], color='#007ACC', alpha=0.2, linewidth=5)



# create for each expense type a dot at the level of the expense percentage value

plt.plot(df['percentage'], my_range, "o", markersize=5, color='#007ACC', alpha=0.6)



# set labels

ax.set_xlabel('Accuracy', fontsize=15, fontweight='black', color = '#333F4B')

ax.set_ylabel('')



# set axis

ax.tick_params(axis='both', which='major', labelsize=12)

plt.yticks(my_range, df.index)



# add an horizonal label for the y axis 

fig.text(-0.23, 0.96, 'Algorithm', fontsize=15, fontweight='black', color = '#333F4B')



# change the style of the axis spines

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['left'].set_smart_bounds(True)

ax.spines['bottom'].set_smart_bounds(True)



# set the spines position

ax.spines['bottom'].set_position(('axes', -0.04))

ax.spines['left'].set_position(('axes', 0.015))