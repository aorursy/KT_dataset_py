# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df_train = pd.read_csv('../input/train.csv')

df_train.drop('Id',axis=1,inplace = True)

print(df_train.info())



X_train, X_cv, Y_train, Y_cv = train_test_split(df_train.drop('Cover_Type',axis=1),

                                                df_train['Cover_Type'], test_size=0.2)

X_test = pd.read_csv('../input/test.csv')

test_ids = X_test['Id']

X_test.drop('Id',inplace=True,axis=1)



# Let's plot the correlation of all features but Soil_Type's. 

col_list = df_train.columns

col_list = [col for col in col_list if not col[0:4]=='Soil']

fig, ax = plt.subplots(figsize=(10,10))  

sns.heatmap(df_train[col_list].corr(),square=True,linewidths=1)

plt.title('Correlation of Variables')



plt.figure(figsize=(10,10))

sns.boxplot(y='Elevation',x='Cover_Type', data= df_train )

plt.title('Elevation vs. Cover_Type')





sns.pairplot( df_train, hue='Cover_Type',vars=['Elevation','Aspect','Slope','Hillshade_9am','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Fire_Points'],diag_kind="kde")

plt.show()

## Starting the learning phase...



## Let's try random forest.

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 300, max_depth=15,min_samples_leaf=2)

clf.fit(X_train, Y_train)



print('Random Forest train score =', clf.score(X=X_train,y=Y_train))

print('Random Forest test score =', clf.score(X=X_cv,y=Y_cv))

print('----------------------')



randForPrediction = clf.predict(X_test)



# Let's get the feature importances as well.

featureImp = [(i, clf.feature_importances_[i]) for i in range(len(clf.feature_importances_))]

featureImp =sorted(featureImp,key=lambda x: x[1],reverse=True)

indList= [x[0] for x in featureImp]

plt.figure(figsize=(20,10))

plt.title('Feature Importance')

plt.bar(range(len(clf.feature_importances_)), [x[1] for x in featureImp])

plt.xticks(range(len(clf.feature_importances_)),

           df_train.drop('Cover_Type',axis=1).columns[indList],rotation=90)

plt.show()

# # Let's try XGBOOST

from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators=300,max_depth=5)

clf.fit(X_train, Y_train)  

print('XGB train score =', clf.score(X=X_train,y=Y_train))

print('XGB test score =', clf.score(X=X_cv,y=Y_cv))

print('----------------------')



XGBPrediction = clf.predict(X_test)
# Finally, let's try SVM.

from sklearn import svm





clf = svm.SVC(C=10,gamma=0.0000001)

clf.fit(X_train, Y_train)  



print('SVM train score =', clf.score(X=X_train,y=Y_train))

print('SVM test score =', clf.score(X=X_cv,y=Y_cv)) 

print('SVM number of support vectors =', len(clf.support_))

print('----------------------')



SVMPrediction = clf.predict(X_test)
from subprocess import check_output



finalPrediction = SVMPrediction

loc_submission = "submission.csv"   



with open(loc_submission, "w") as outfile:

    outfile.write("Id,Cover_Type\n")

    for e, val in enumerate(finalPrediction):

      outfile.write("%s,%s\n"%(test_ids[e],val))