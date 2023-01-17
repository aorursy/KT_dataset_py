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

import math

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

%matplotlib inline

font = {"family":"HGMaruGothicMPRO"}

matplotlib.rc("font",**font)

import seaborn as sns

from pandas import Series,DataFrame

import numpy as np

import re
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#dfs =  pd.read_csv('../input/train_small.csv')
df.info()
df.head()
sns.distplot(df["ConvertedSalary"],kde=False, rug=False, bins=50)
print(df.isnull().any())
df.describe()
df2 = df.drop(["Hobby","OpenSource","Employment","FormalEducation","UndergradMajor","CompanySize","DevType","YearsCoding","YearsCodingProf","JobSatisfaction","CareerSatisfaction","HopeFiveYears","JobSearchStatus","LastNewJob","UpdateCV","Currency","SalaryType","CurrencySymbol","CommunicationTools","TimeFullyProductive","TimeAfterBootcamp","AgreeDisagree1","AgreeDisagree2","AgreeDisagree3","FrameworkWorkedWith","OperatingSystem","CheckInCode","AdBlocker","AdBlockerDisable","AdsAgreeDisagree1","AdsAgreeDisagree2","AdsAgreeDisagree3","AdsActions","AIDangerous","AIInteresting","AIResponsible","AIFuture","EthicsChoice","EthicsReport","EthicsResponsible","EthicalImplications","StackOverflowRecommend","StackOverflowVisit","StackOverflowHasAccount","StackOverflowParticipate","StackOverflowJobs","StackOverflowDevStory","StackOverflowJobsRecommend","StackOverflowConsiderMember","HypotheticalTools1","HypotheticalTools2","HypotheticalTools3","HypotheticalTools4","HypotheticalTools5","WakeTime","HoursComputer","HoursOutside","SkipMeals","ErgonomicDevices","Exercise","Gender","SexualOrientation","EducationParents","RaceEthnicity","Age","Dependents","MilitaryUS","SurveyTooLong","SurveyEasy"

],axis=1)

test1 = test.drop(["Hobby","OpenSource","Employment","FormalEducation","UndergradMajor","CompanySize","DevType","YearsCoding","YearsCodingProf","JobSatisfaction","CareerSatisfaction","HopeFiveYears","JobSearchStatus","LastNewJob","UpdateCV","Currency","SalaryType","CurrencySymbol","CommunicationTools","TimeFullyProductive","TimeAfterBootcamp","AgreeDisagree1","AgreeDisagree2","AgreeDisagree3","FrameworkWorkedWith","OperatingSystem","CheckInCode","AdBlocker","AdBlockerDisable","AdsAgreeDisagree1","AdsAgreeDisagree2","AdsAgreeDisagree3","AdsActions","AIDangerous","AIInteresting","AIResponsible","AIFuture","EthicsChoice","EthicsReport","EthicsResponsible","EthicalImplications","StackOverflowRecommend","StackOverflowVisit","StackOverflowHasAccount","StackOverflowParticipate","StackOverflowJobs","StackOverflowDevStory","StackOverflowJobsRecommend","StackOverflowConsiderMember","HypotheticalTools1","HypotheticalTools2","HypotheticalTools3","HypotheticalTools4","HypotheticalTools5","WakeTime","HoursComputer","HoursOutside","SkipMeals","ErgonomicDevices","Exercise","Gender","SexualOrientation","EducationParents","RaceEthnicity","Age","Dependents","MilitaryUS","SurveyTooLong","SurveyEasy"

],axis=1)
df2.info()
#X = pd.get_dummies(columns=["Hobby","OpenSource","Country","Student","Employment","FormalEducation","UndergradMajor","CompanySize","DevType","YearsCoding","YearsCodingProf","JobSatisfaction","CareerSatisfaction","HopeFiveYears","JobSearchStatus","LastNewJob","UpdateCV","Currency","SalaryType","CurrencySymbol","CommunicationTools","TimeFullyProductive","TimeAfterBootcamp","AgreeDisagree1","AgreeDisagree2","AgreeDisagree3","FrameworkWorkedWith","OperatingSystem","CheckInCode","AdBlocker","AdBlockerDisable","AdsAgreeDisagree1","AdsAgreeDisagree2","AdsAgreeDisagree3","AdsActions","AIDangerous","AIInteresting","AIResponsible","AIFuture","EthicsChoice","EthicsReport","EthicsResponsible","EthicalImplications","StackOverflowRecommend","StackOverflowVisit","StackOverflowHasAccount","StackOverflowParticipate","StackOverflowJobs","StackOverflowDevStory","StackOverflowJobsRecommend","StackOverflowConsiderMember","HypotheticalTools1","HypotheticalTools2","HypotheticalTools3","HypotheticalTools4","HypotheticalTools5","WakeTime","HoursComputer","HoursOutside","SkipMeals","ErgonomicDevices","Exercise","Gender","SexualOrientation","EducationParents","RaceEthnicity","Age","Dependents","MilitaryUS","SurveyTooLong","SurveyEasy","NumberMonitors"],data=df)
#test2 = pd.get_dummies(columns=["Hobby","OpenSource","Country","Student","Employment","FormalEducation","UndergradMajor","CompanySize","DevType","YearsCoding","YearsCodingProf","JobSatisfaction","CareerSatisfaction","HopeFiveYears","JobSearchStatus","LastNewJob","UpdateCV","Currency","SalaryType","CurrencySymbol","CommunicationTools","TimeFullyProductive","TimeAfterBootcamp","AgreeDisagree1","AgreeDisagree2","AgreeDisagree3","FrameworkWorkedWith","OperatingSystem","CheckInCode","AdBlocker","AdBlockerDisable","AdsAgreeDisagree1","AdsAgreeDisagree2","AdsAgreeDisagree3","AdsActions","AIDangerous","AIInteresting","AIResponsible","AIFuture","EthicsChoice","EthicsReport","EthicsResponsible","EthicalImplications","StackOverflowRecommend","StackOverflowVisit","StackOverflowHasAccount","StackOverflowParticipate","StackOverflowJobs","StackOverflowDevStory","StackOverflowJobsRecommend","StackOverflowConsiderMember","HypotheticalTools1","HypotheticalTools2","HypotheticalTools3","HypotheticalTools4","HypotheticalTools5","WakeTime","HoursComputer","HoursOutside","SkipMeals","ErgonomicDevices","Exercise","Gender","SexualOrientation","EducationParents","RaceEthnicity","Age","Dependents","MilitaryUS","SurveyTooLong","SurveyEasy","NumberMonitors"]

#                   ,data=test)
X=pd.get_dummies(columns=["NumberMonitors","Country","Student"],data=df2)

test2=pd.get_dummies(columns=["NumberMonitors","Country","Student"],data=test1)
train2 = X

test3 = test2

test3.set_index("Respondent")
y = df2[["ConvertedSalary"]]

yID= test3[["Respondent"]]

train3= train2.drop(["Respondent"],axis=1)

test4 = test3.drop(["Respondent"],axis=1)
m_col=set(train3.columns)-set(test4.columns)

for c in m_col:

    test4[c]=0



m_col2=set(test4.columns)-set(train3.columns)

for c in m_col2:

    train3[c]=0
test4=test4.sort_index(axis=1)

train3=train3.sort_index(axis=1)
test3.describe()
train2=train2.sort_index(axis=1)
import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

model = xgb.XGBRegressor()

xg_reg = xgb.XGBRegressor(objective='reg:linear',reg_alpha=0.9,max_depth=6,n_estimators=116,seed=123,gamma=2,n_jobs=3,reg_lambda=3)
X_train,X_test,y_train,y_test = train_test_split(train3,y)

model.fit(X_train,y_train)

#xg_reg.fit(X_train, y_train)
pred_out =  xg_reg.predict(test4)

#pred =  model.predict(X_test)

#print(np.sqrt(mean_squared_error(y_test,pred)))

print(pred_out)
#値がすべて同じになってしまっている

#上の学習時に機能していて、予測時に失敗している

#test4に異常がないか調べたものの特になし

#pred_out=model.predict(test4)

pred_out=xg_reg.predict(test4)

print(pred_out)
yID["Prediction"] = pred_out

#final_df.to_csv("Output_1.csv")

yID.head(100)
submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.ConvertedSalary = pred_out

submission.to_csv('submission.csv')