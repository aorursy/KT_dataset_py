# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np
train = pd.read_csv("../input/train.csv", nrows=2000)

test = pd.read_csv("../input/test.csv", nrows=2000)
print("Train Shape --> ", train.shape)

print("Test  Shape --> ", test.shape)

print("Training data sample: \n",train.head())
print("Number of missing values",train.isnull().sum().sum())
print("Training data types -->",tuple(train.columns.to_series().groupby(train.dtypes).groups))

print("Training data types -->",tuple(test.columns.to_series().groupby(test.dtypes).groups))
from sklearn.preprocessing import LabelEncoder

def convertcat2cont(df):

    print("Before -->",tuple(df.columns.to_series().groupby(df.dtypes).groups))

    for cf1 in catFeatureslist:

        le = LabelEncoder()

        le.fit(df[cf1].unique())

        df[cf1] = le.transform(df[cf1])

    print("After -->",tuple(df.columns.to_series().groupby(df.dtypes).groups))
# Understanding categorical and continuous features

catCount = sum(str(x).isalpha() for x in train.iloc[1,:])

print("Number of categories features: ",catCount)

contCount = sum(not str(x).isalpha() for x in train.iloc[1,:])

print("Number of Continuous features: ",contCount)
catFeatureslist = []

contFeatureslist = []

for colName,x in train.iloc[1,:].iteritems():

	if(str(x).isalpha()):

		catFeatureslist.append(colName)

	else:

		contFeatureslist.append(colName)
print("Number of categories features: ",len(catFeatureslist))

print("Number of Continuous features: ",len(contFeatureslist))
# Box plots for continuous features



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#plt.figure(figsize=(13,9))

sns.boxplot(train[contFeatureslist])
# Correlation between continuous features

correlationMatrix = train[contFeatureslist].corr().abs()

plt.subplots(figsize=(12, 8))

sns.heatmap(correlationMatrix,annot=True)
# Mask unimportant features (less than 0.5)

sns.heatmap(correlationMatrix, mask=correlationMatrix < .5, cbar=False)

plt.show()
# Analysis of loss feature



#plt.figure(figsize=(12,8))

sns.distplot(train["loss"])

sns.boxplot(train["loss"])
# use log to remove the skewness

#plt.figure(figsize=(12,8))

sns.distplot(np.log1p(train["loss"]))
# Unique categorical values per each category

print(train[catFeatureslist].apply(pd.Series.nunique))
#Analysis of categorical features with levels between 5-10
filterG5_10 = list((train[catFeatureslist].apply(pd.Series.nunique) > 5) & 

                (train[catFeatureslist].apply(pd.Series.nunique) < 10))
catFeaturesG5_10List = [i for (i, v) in zip(catFeatureslist, filterG5_10) if v]
len(catFeaturesG5_10List)
ncol = 2

nrow = 4

try:

    for rowIndex in range(nrow):

        f,axList = plt.subplots(nrows=1,ncols=ncol,sharey=True) #,figsize=(13, 9))

        features = catFeaturesG5_10List[rowIndex*ncol:ncol*(rowIndex+1)]

        

        for axIndex in range(len(axList)):

            sns.boxplot(x=features[axIndex], y="loss", data=train, ax=axList[axIndex])

                        

            # With original scale it is hard to visualize because of outliers

            axList[axIndex].set(yscale="log")

            axList[axIndex].set(xlabel=features[axIndex], ylabel='log loss')

except IndexError:

    print("")
#convert categorical variables to continuous

convertcat2cont(train)

convertcat2cont(test)
#Correlation between categorical variables

filterG2 = list((train[catFeatureslist].apply(pd.Series.nunique) == 2))

catFeaturesG2List = [i for (i, v) in zip(catFeatureslist, filterG2) if v]

catFeaturesG2List.append("loss")



corrCatMatrix = train[catFeaturesG2List].corr().abs()



s = corrCatMatrix.unstack()

sortedSeries= s.order(kind="quicksort",ascending=False)



print("Top 5 most correlated categorical feature pairs: \n")

print(sortedSeries[sortedSeries != 1.0][0:9])
print("train --> ", train.shape)

X_train = train.drop(['id','loss'], axis=1)

y_train = train['loss']

print("X features -->", X_train.shape)

print("y feature --->", y_train.shape)



print("test --> ", test.shape)

X_test = test.drop(['id'], axis=1)

# predict y_test

print("X features -->", X_test.shape)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=50, random_state=3)

def doRFR(X_train,y_train,X_test):

    rfr.fit(X_train,y_train)

    print("Accuracy on training set: {:.3f}".format(rfr.score(X_train, y_train)))

    y_predrfr = rfr.predict(X_test)

    print("RandomForestRegressor")

    print(y_predrfr[0])
# any object or category will make the analysis fail

print(tuple(X_train.columns.to_series().groupby(X_train.dtypes).groups))

print(tuple(X_test.columns.to_series().groupby(X_test.dtypes).groups))
y_pred = doRFR(X_train,y_train,X_test)
preds = pd.DataFrame({"id": test['id'],"loss": y_pred})

preds.head(5)

preds.to_csv('AllStateClaimsSeverity_yyyymmdd.csv', index=False)
print(check_output(["ls"]).decode("utf8"))