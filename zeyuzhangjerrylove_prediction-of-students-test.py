### Project Title: Prediction of Students' Test

### Authors: Zeyu Zhang and Yuchen Feng

### Date: April 15th, 2020

### Data Set Link: https://www.kaggle.com/spscientist/students-performance-in-exams



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn



from sklearn import preprocessing

from sklearn.model_selection import cross_val_score as cvs

from sklearn.linear_model import LogisticRegression as lr

from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.tree import DecisionTreeClassifier as dtc

from sklearn.feature_selection import SelectKBest, f_classif
passScore = 50

excellentScore = 90

dataSet = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")

dataSet.head(10)
dataSet.isnull().sum()
dataSet.describe()
dataSet.groupby(['gender']).agg(['mean'])
dataSet.groupby(['race/ethnicity']).agg(['mean'])
dataSet.groupby(['parental level of education']).agg(['mean'])
dataSet.groupby(['lunch']).agg(['mean'])
dataSet.groupby(['test preparation course']).agg(['mean'])
dataSet['Pass_Math'] = np.where(dataSet['math score'] >= passScore, 'Pass', 'Fail')

dataSet.Pass_Math.value_counts()
dataSet['Pass_Reading'] = np.where(dataSet['reading score'] >= passScore, 'Pass', 'Fail')

dataSet.Pass_Reading.value_counts()
dataSet['Pass_Writing'] = np.where(dataSet['writing score'] >= passScore, 'Pass', 'Fail')

dataSet.Pass_Writing.value_counts()
dataSet['Pass_All'] = np.where((dataSet['math score'] >= passScore) & 

                               (dataSet['reading score'] >= passScore) & 

                               (dataSet['writing score'] >= passScore), 'Pass', 'Fail')

dataSet.Pass_All.value_counts()
dataSet['Excellent_Math'] = np.where(dataSet['math score'] >= excellentScore, 'Pass', 'Not Pass')

dataSet.Excellent_Math.value_counts()
dataSet['Excellent_Reading'] = np.where(dataSet['reading score'] >= excellentScore, 'Pass', 'Not Pass')

dataSet.Excellent_Reading.value_counts()
dataSet['Excellent_Writing'] = np.where(dataSet['writing score'] >= excellentScore, 'Pass', 'Not Pass')

dataSet.Excellent_Writing.value_counts()
dataSet['Excellent_All'] = np.where((dataSet['math score'] >= excellentScore) & 

                                    (dataSet['reading score'] >= excellentScore) &

                                    (dataSet['writing score'] >= excellentScore), 'Pass', 'Not Pass')

dataSet.Excellent_All.value_counts()
sns.set(style = 'whitegrid')

sns.countplot(x = 'gender', data = dataSet, hue = 'Pass_All', palette = 'Set1')

plt.title('Number of Students Passed All Courses', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

sns.countplot(x = 'gender', data = dataSet, hue = 'Excellent_All', palette = 'Set3')

plt.title('Number of Students Excellent for All', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

sns.countplot(x = 'race/ethnicity', data = dataSet, hue = 'Pass_All', palette = 'Set1')

plt.title('Number of Students Passed All Courses', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

sns.countplot(x = 'race/ethnicity', data = dataSet, hue = 'Excellent_All', palette = 'Set3')

plt.title('Number of Students Excellent for All', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

plot = sns.countplot(x = 'parental level of education', data = dataSet, hue = 'Pass_All', palette = 'Set1')

plt.setp(plot.get_xticklabels(), rotation = 45)

plt.title('Number of Students Passed All', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

plot = sns.countplot(x = 'parental level of education', data = dataSet, hue = 'Excellent_All', palette = 'Set3')

plt.setp(plot.get_xticklabels(), rotation = 45)

plt.title('Number of Students Excellent for All', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

sns.countplot(x = 'lunch', data = dataSet, hue = 'Pass_All', palette = 'Set1')

plt.title('Number of Students Passed All', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

sns.countplot(x = 'lunch', data = dataSet, hue = 'Excellent_All', palette = 'Set3')

plt.title('Number of Students Excellent for All', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

sns.countplot(x = 'test preparation course', data = dataSet, hue = 'Pass_All', palette = 'Set1')

plt.title('Number of Students Passed All', fontweight = 20, fontsize = 15)
sns.set(style = 'whitegrid')

sns.countplot(x = 'test preparation course', data = dataSet, hue = 'Excellent_All', palette = 'Set3')

plt.title('Number of Students Excellent for All', fontweight = 20, fontsize = 15)
dataSet.loc[dataSet['gender'] == 'male', 'gender'] = 1

dataSet.loc[dataSet['gender'] == 'female', 'gender'] = 0



dataSet.loc[dataSet['race/ethnicity'] == 'group A', 'race/ethnicity'] = 0

dataSet.loc[dataSet['race/ethnicity'] == 'group B', 'race/ethnicity'] = 1

dataSet.loc[dataSet['race/ethnicity'] == 'group C', 'race/ethnicity'] = 2

dataSet.loc[dataSet['race/ethnicity'] == 'group D', 'race/ethnicity'] = 3

dataSet.loc[dataSet['race/ethnicity'] == 'group E', 'race/ethnicity'] = 4



dataSet.loc[dataSet['parental level of education'] == 'some high school', 'parental level of education'] = 0

dataSet.loc[dataSet['parental level of education'] == 'high school', 'parental level of education'] = 1

dataSet.loc[dataSet['parental level of education'] == "associate's degree", 'parental level of education'] = 2

dataSet.loc[dataSet['parental level of education'] == 'some college', 'parental level of education'] = 3

dataSet.loc[dataSet['parental level of education'] == "bachelor's degree", 'parental level of education'] = 4

dataSet.loc[dataSet['parental level of education'] == "master's degree", 'parental level of education'] = 5



dataSet.loc[dataSet['lunch'] == 'standard', 'lunch'] = 1

dataSet.loc[dataSet['lunch'] == 'free/reduced', 'lunch'] = 0



dataSet.loc[dataSet['test preparation course'] == 'completed', 'test preparation course'] = 1

dataSet.loc[dataSet['test preparation course'] == 'none', 'test preparation course'] = 0



dataSet.loc[dataSet['Pass_Math'] == 'Pass', 'Pass_Math'] = 1

dataSet.loc[dataSet['Pass_Math'] == 'Fail', 'Pass_Math'] = 0



dataSet.loc[dataSet['Pass_Reading'] == 'Pass', 'Pass_Reading'] = 1

dataSet.loc[dataSet['Pass_Reading'] == 'Fail', 'Pass_Reading'] = 0



dataSet.loc[dataSet['Pass_Writing'] == 'Pass', 'Pass_Writing'] = 1

dataSet.loc[dataSet['Pass_Writing'] == 'Fail', 'Pass_Writing'] = 0



dataSet.loc[dataSet['Pass_All'] == 'Pass', 'Pass_All'] = 1

dataSet.loc[dataSet['Pass_All'] == 'Fail', 'Pass_All'] = 0



dataSet.loc[dataSet['Excellent_Math'] == 'Pass', 'Excellent_Math'] = 1

dataSet.loc[dataSet['Excellent_Math'] == 'Not Pass', 'Excellent_Math'] = 0



dataSet.loc[dataSet['Excellent_Reading'] == 'Pass', 'Excellent_Reading'] = 1

dataSet.loc[dataSet['Excellent_Reading'] == 'Not Pass', 'Excellent_Reading'] = 0



dataSet.loc[dataSet['Excellent_Writing'] == 'Pass', 'Excellent_Writing'] = 1

dataSet.loc[dataSet['Excellent_Writing'] == 'Not Pass', 'Excellent_Writing'] = 0



dataSet.loc[dataSet['Excellent_All'] == 'Pass', 'Excellent_All'] = 1

dataSet.loc[dataSet['Excellent_All'] == 'Not Pass', 'Excellent_All'] = 0



dataSet.head(10)
features = dataSet.iloc[:,:5]



target_PassMath = dataSet.iloc[:,8].astype(int)

target_PassReading = dataSet.iloc[:,9].astype(int)

target_PassWriting = dataSet.iloc[:,10].astype(int)

target_PassAll = dataSet.iloc[:,11].astype(int)

target_ExcellentMath = dataSet.iloc[:,12].astype(int)

target_ExcellentReading = dataSet.iloc[:,13].astype(int)

target_ExcellentWriting = dataSet.iloc[:,14].astype(int)

target_ExcellentAll = dataSet.iloc[:,15].astype(int)



features_cross = preprocessing.scale(features)



modelLR = lr()

modelRFC = rfc()

modelDTC = dtc()



cvsScoreLR_PassMath = np.mean(cvs(modelLR, features_cross, target_PassMath, cv = 5))

cvsScoreLR_PassReading = np.mean(cvs(modelLR, features_cross, target_PassReading, cv = 5))

cvsScoreLR_PassWriting = np.mean(cvs(modelLR, features_cross, target_PassWriting, cv = 5))

cvsScoreLR_PassAll = np.mean(cvs(modelLR, features_cross, target_PassAll, cv = 5))

cvsScoreLR_ExcellentMath = np.mean(cvs(modelLR, features_cross, target_ExcellentMath, cv = 5))

cvsScoreLR_ExcellentReading = np.mean(cvs(modelLR, features_cross, target_ExcellentReading, cv = 5))

cvsScoreLR_ExcellentWriting = np.mean(cvs(modelLR, features_cross, target_ExcellentWriting, cv = 5))

cvsScoreLR_ExcellentAll = np.mean(cvs(modelLR, features_cross, target_ExcellentAll, cv = 5))



cvsScoreRFC_PassMath = np.mean(cvs(modelRFC, features_cross, target_PassMath, cv = 5))

cvsScoreRFC_PassReading = np.mean(cvs(modelRFC, features_cross, target_PassReading, cv = 5))

cvsScoreRFC_PassWriting = np.mean(cvs(modelRFC, features_cross, target_PassWriting, cv = 5))

cvsScoreRFC_PassAll = np.mean(cvs(modelRFC, features_cross, target_PassAll, cv = 5))

cvsScoreRFC_ExcellentMath = np.mean(cvs(modelRFC, features_cross, target_ExcellentMath, cv = 5))

cvsScoreRFC_ExcellentReading = np.mean(cvs(modelRFC, features_cross, target_ExcellentReading, cv = 5))

cvsScoreRFC_ExcellentWriting = np.mean(cvs(modelRFC, features_cross, target_ExcellentWriting, cv = 5))

cvsScoreRFC_ExcellentAll = np.mean(cvs(modelRFC, features_cross, target_ExcellentAll, cv = 5))



cvsScoreDTC_PassMath = np.mean(cvs(modelDTC, features_cross, target_PassMath, cv = 5))

cvsScoreDTC_PassReading = np.mean(cvs(modelDTC, features_cross, target_PassReading, cv = 5))

cvsScoreDTC_PassWriting = np.mean(cvs(modelDTC, features_cross, target_PassWriting, cv = 5))

cvsScoreDTC_PassAll = np.mean(cvs(modelDTC, features_cross, target_PassAll, cv = 5))

cvsScoreDTC_ExcellentMath = np.mean(cvs(modelDTC, features_cross, target_ExcellentMath, cv = 5))

cvsScoreDTC_ExcellentReading = np.mean(cvs(modelDTC, features_cross, target_ExcellentReading, cv = 5))

cvsScoreDTC_ExcellentWriting = np.mean(cvs(modelDTC, features_cross, target_ExcellentWriting, cv = 5))

cvsScoreDTC_ExcellentAll = np.mean(cvs(modelDTC, features_cross, target_ExcellentAll, cv = 5))



print('Cross Validation Score of Logistic Regression (PassMath) is: ', 

      cvsScoreLR_PassMath.astype(str))

print('Cross Validation Score of Logistic Regression (PassReading) is: ', 

      cvsScoreLR_PassReading.astype(str))

print('Cross Validation Score of Logistic Regression (PassWriting) is: ', 

      cvsScoreLR_PassWriting.astype(str))

print('Cross Validation Score of Logistic Regression (PassAll) is: ', 

      cvsScoreLR_PassAll.astype(str))

print('Cross Validation Score of Logistic Regression (ExcellentMath) is: ', 

      cvsScoreLR_ExcellentMath.astype(str))

print('Cross Validation Score of Logistic Regression (ExcellentReading) is: ', 

      cvsScoreLR_ExcellentReading.astype(str))

print('Cross Validation Score of Logistic Regression (ExcellentWriting) is: ', 

      cvsScoreLR_ExcellentWriting.astype(str))

print('Cross Validation Score of Logistic Regression (ExcellentAll) is: ', 

      cvsScoreLR_ExcellentAll.astype(str))

print()

print('Cross Validation Score of Random Forest Classifier (PassMath) is: ', 

      cvsScoreRFC_PassMath.astype(str))

print('Cross Validation Score of Random Forest Classifier (PassReading) is: ', 

      cvsScoreRFC_PassReading.astype(str))

print('Cross Validation Score of Random Forest Classifier (PassWriting) is: ', 

      cvsScoreRFC_PassWriting.astype(str))

print('Cross Validation Score of Random Forest Classifier (PassAll) is: ', 

      cvsScoreRFC_PassAll.astype(str))

print('Cross Validation Score of Random Forest Classifier (ExcellentMath) is: ', 

      cvsScoreRFC_ExcellentMath.astype(str))

print('Cross Validation Score of Random Forest Classifier (ExcellentReading) is: ', 

      cvsScoreRFC_ExcellentReading.astype(str))

print('Cross Validation Score of Random Forest Classifier (ExcellentWriting) is: ', 

      cvsScoreRFC_ExcellentWriting.astype(str))

print('Cross Validation Score of Random Forest Classifier (ExcellentAll) is: ', 

      cvsScoreRFC_ExcellentAll.astype(str))

print()

print('Cross Validation Score of Decision Tree Classfier (PassMath) is: ', 

      cvsScoreDTC_PassMath.astype(str))

print('Cross Validation Score of Decision Tree Classfier (PassReading) is: ', 

      cvsScoreDTC_PassReading.astype(str))

print('Cross Validation Score of Decision Tree Classfier (PassWriting) is: ', 

      cvsScoreDTC_PassWriting.astype(str))

print('Cross Validation Score of Decision Tree Classfier (PassAll) is: ', 

      cvsScoreDTC_PassAll.astype(str))

print('Cross Validation Score of Decision Tree Classfier (ExcellentMath) is: ', 

      cvsScoreDTC_ExcellentMath.astype(str))

print('Cross Validation Score of Decision Tree Classfier (ExcellentReading) is: ', 

      cvsScoreDTC_ExcellentReading.astype(str))

print('Cross Validation Score of Decision Tree Classfier (ExcellentWriting) is: ', 

      cvsScoreDTC_ExcellentWriting.astype(str))

print('Cross Validation Score of Decision Tree Classfier (ExcellentAll) is: ', 

      cvsScoreDTC_ExcellentAll.astype(str))
relationship_PassAll = SelectKBest(f_classif, k = 5)

relationship_PassAll.fit(features, target_PassAll)

relationship_ExcellentAll = SelectKBest(f_classif, k = 5)

relationship_ExcellentAll.fit(features, target_ExcellentAll)

relationship_ExcellentReading = SelectKBest(f_classif, k = 5)

relationship_ExcellentReading.fit(features, target_ExcellentReading)



rCoeff_PassAll = -np.log(relationship_PassAll.pvalues_)

rCoeff_ExcellentAll = -np.log(relationship_ExcellentAll.pvalues_)

rCoeff_ExcellentReading = -np.log(relationship_ExcellentReading.pvalues_)



plt.bar(range(5), rCoeff_PassAll)

plt.xticks(range(5), features, rotation = 45)

plt.title('Relationship Between Features and Passing Status', fontweight = 15, fontsize = 10)

plt.ylabel('Relationship Coefficient')

plt.show()
plt.bar(range(5), rCoeff_ExcellentAll)

plt.xticks(range(5), features, rotation = 45)

plt.title('Relationship Between Features and Excellent Status', fontweight = 15, fontsize = 10)

plt.ylabel('Relationship Coefficient')

plt.show()
plt.bar(range(5), rCoeff_ExcellentReading)

plt.xticks(range(5), features, rotation = 45)

plt.title('Relationship Between Features and Excellent Reading Status', fontweight = 15, fontsize = 10)

plt.ylabel('Relationship Coefficient')

plt.show()