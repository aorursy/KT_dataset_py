import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

#import os

#print(os.listdir("../input"))

#print(os.getcwd())



dataset=pd.read_csv('../input/diabetes.csv')

dataset.columns
dataset.shape
dataset.head(5)
dataset.describe()
# taking the transpose of the dataset

dataset.describe().T
# Minimum value is observed to be 0 for Glucose, BloodPressure,SkinThickness,Insulin, BMI 

# The same can be observed in the histograms that follows

# This cannot be true that means ..they are incorrect values and we deal with the incorrect values by first replacing 

# the 0's with nan's





dataset_copy =dataset.copy(deep=True)

dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(to_replace=0,value=np.NaN)     

 

print(dataset_copy.isnull().sum())

# visualise the missing data

sns.heatmap(dataset_copy.isnull(),cbar=False)

fig =plt.gcf()

fig.set_size_inches(10,8)

len(dataset_copy.index)
# With so much of Missing data in the columns : Insulin and then SkinThickness, it is better to drop these two columns

# Let us see what's the percentage of missing data

def missing(data):

    print("Missing values in %")

    print(round(((data.isnull().sum() * 100)/ len(data)),2).sort_values(ascending=False))

    

missing(dataset_copy)    

# clearly Insulin has Most of the missing data. We drop this column for the analysis purpose and see how 

# it impacts out prediction model
dataset_copy2 = dataset_copy.drop(columns=['Insulin'], axis=1)

dataset_copy2.head(5)
# To fill these NaNs, the data distribution needs to be understood
# Dataset Correlation Matrix

corr=dataset_copy2.corr()

corr

import matplotlib.pyplot as plt

import seaborn as sns

sns.heatmap(corr,annot=True)

fig =plt.gcf()

fig.set_size_inches(10,8)

plt.show()
# Now we shall visualise the data to have a better understanding of the various features values distribution



dataset_copy2.hist(bins=50,figsize=(20,15))

plt.show()
# check for missing data

# dataset.isnull().sum() # actual data set 

dataset_copy2.isnull().sum()
# The data visualisation has helped us in better way otherwise we would not have obtained the incorrect data present 

# in the dataset. We cannot just eliminate the patients wit null/zero values. This would remove a lot of important 

# data. Another way if to calculate the mean value for a column and substitute the value
# Imputing the nan values for the columns in accordance with their distribution, we get



dataset_copy2['Glucose'].fillna(dataset_copy2['Glucose'].mean(), inplace = True)

dataset_copy2['BloodPressure'].fillna(dataset_copy2['BloodPressure'].mean(), inplace = True)

dataset_copy2['SkinThickness'].fillna(dataset_copy2['SkinThickness'].mean(), inplace = True)

#dataset_copy2['Insulin'].fillna(dataset_copy2['Insulin'].mean(), inplace = True)

dataset_copy2['BMI'].fillna(dataset_copy2['BMI'].mean(), inplace = True)



# Number of Pregnancy can be 0 - so we do not do transformation on that feature

dataset_copy2.isnull().sum()
#Plotting after NaN removal



dataset_copy2.hist(bins=50,figsize=(20,15))

plt.show()
# Skewness:

# A left-skewed distribution has a long left tail. Left-skewed distributions are also called negatively-skewed 

# distributions. That’s because there is a long tail in the negative direction on the number line. 

# The mean is also to the left of the peak.



# A right-skewed distribution has a long right tail. Right-skewed distributions are also called positive-skewed

# distributions. That’s because there is a long tail in the positive direction on the number line. 

# The mean is also to the right of the peak.



# Splitting the data into training set and Test Set



dataset_copy2.head(5)


X=dataset_copy2.iloc[:,0:7].values

y=dataset_copy2.iloc[:,-1].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)
from sklearn.preprocessing import StandardScaler,MinMaxScaler

sc_X = StandardScaler()

#sc_X =MinMaxScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.fit_transform(X_test)
#X_traindf = pd.DataFrame(data=X_train_scaled)

#X_traindf.head()

# Select and train a model

# Comparing Multiple Algorithms
## import all the algorithms we want to test

#from sklearn.linear_model import LogisticRegression

#classifier=LogisticRegression()

#classifier.fit(X_train,y_train)



## Fitting Naive Bayes to the Training set

#from sklearn.naive_bayes import GaussianNB

#classifier = GaussianNB()

#classifier.fit(X_train, y_train)



from sklearn.svm import SVC

classifier = SVC(kernel='linear')

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)



from sklearn.metrics import accuracy_score

print('The accuracy of this model is: ', accuracy_score(y_pred, y_test))
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor
# import the sklearn utility to compare algorithms

from sklearn import model_selection
models =[]



models.append(('LR', LogisticRegression(solver='lbfgs')))

models.append(('KNN', KNeighborsClassifier()))

models.append(('NB', GaussianNB()))

#models.append(('SVC', SVC(gamma='auto')))

#models.append(('LSVC', LinearSVC()))

models.append(('RFC', RandomForestClassifier(n_estimators=100)))

models.append(('DTR', DecisionTreeRegressor()))
seed=7

results=[]

names =[]

X=X_train

Y=y_train

# Every algorithm is tested and results are

# collected and printed









for name, model in models:

    kfold = model_selection.KFold(

        n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(

        model, X, Y, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (

        name, cv_results.mean(), cv_results.std())

    print(msg)
# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
#Logistic Regression seems to be the preferred algorithm for these datasets