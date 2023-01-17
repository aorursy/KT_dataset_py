#import libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import pandas_profiling

import seaborn as sns
#importing dataset

dataset = pd.read_csv('../input/titanic/train.csv')

dataset1 = pd.read_csv('../input/titanic/test.csv')
#creating report using pandas profiling

report1 = pandas_profiling.ProfileReport(dataset)

report1
#information of data

dataset.info()
#more information

dataset.describe()
#analysing data using seaborn

sns.set(style="ticks", color_codes=True);

sns.catplot(x="Sex", kind="count", palette="ch:.25", data=dataset);
sns.catplot(x="Survived", y="Age", hue="Sex",

            kind="violin",inner = 'stick',palette='pastel', split=True,

            data=dataset);
sns.jointplot(x="Age", y="Fare",kind = 'hex', data = dataset)
sns.lmplot(x="Fare", y="Age", data=dataset);
sns.pairplot(dataset);
X = dataset.iloc[:, np.r_[0:1,2:3,4:8,9:10]].values



#taking care of missing data

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer.fit(X[:,np.r_[3:4, 6:7]])
X[:,np.r_[3:4, 6:7]] = imputer.transform(X[:,np.r_[3:4, 6:7]])
Y = dataset.iloc[:,1].values
x_test = dataset1.iloc[:,np.r_[0:2, 3:7, 8:9]].values
#taking care of missing data in test dataset

imputer1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')

imputer1.fit(x_test[:,np.r_[3:4, 6:7]])
x_test[:,np.r_[3:4, 6:7]] = imputer1.transform(x_test[:,np.r_[3:4, 6:7]])
#Encoding categorical data 

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

X[:,2] = le.fit_transform(X[:,2])


leTest = LabelEncoder()

x_test[:,2] = leTest.fit_transform(x_test[:,2])
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X[:,np.r_[0:1, 3:7]] = sc.fit_transform(X[:,np.r_[0:1, 3:7]])

x_test[:,np.r_[0:1, 3:7]] = sc.transform(x_test[:,np.r_[0:1, 3:7]])

#from sklearn.decomposition import PCA

#kpca = PCA(n_components = 2)

#X = kpca.fit_transform(X)

#x_test = kpca.transform(x_test)
#applying SVM

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X,Y)
y_pred = classifier.predict(x_test)

y_pred
#Submitting the Output

my_submission = pd.DataFrame({'PassengerId': dataset1.PassengerId, 'Survived': y_pred})
# you could use any filename. We choose submission here

my_submission.to_csv('kapilv_titanic_submission.csv', index=False)