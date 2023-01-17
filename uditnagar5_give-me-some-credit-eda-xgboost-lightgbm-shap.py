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

from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
import os, datetime, sys, random, time
import seaborn as sns
import xgboost as xgs
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mlxtend import classifier
plt.style.use('fivethirtyeight')
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from scipy import stats, special
import shap
import catboost as ctb
trainingData = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
testData = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv')
trainingData.head()
trainingData.info()
trainingData.describe()
print(trainingData.shape)
print(testData.shape)
testData.head()
testData.info()
testData.describe()
finalTrain = trainingData.copy()
finalTest = testData.copy()
finalTest.drop('SeriousDlqin2yrs', axis=1, inplace = True)
trainID = finalTrain['Unnamed: 0']
testID = finalTest['Unnamed: 0']

finalTrain.drop('Unnamed: 0', axis=1, inplace=True)
finalTest.drop('Unnamed: 0', axis=1, inplace=True)
fig, axes = plt.subplots(1,2,figsize=(12,6))
finalTrain['SeriousDlqin2yrs'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=axes[0])
axes[0].set_title('SeriousDlqin2yrs')
#ax[0].set_ylabel('')
sns.countplot('SeriousDlqin2yrs',data=finalTrain,ax=axes[1])
axes[1].set_title('SeriousDlqin2yrs')
plt.show()
fig = plt.figure(figsize=[30,30])
for col,i in zip(finalTrain.columns,range(1,13)):
    axes = fig.add_subplot(7,2,i)
    sns.regplot(finalTrain[col],finalTrain.SeriousDlqin2yrs,ax=axes)
plt.show()
print("Unique values in '30-59 Days' values that are more than or equal to 90:",np.unique(finalTrain[finalTrain['NumberOfTime30-59DaysPastDueNotWorse']>=90]
                                                                                          ['NumberOfTime30-59DaysPastDueNotWorse']))


print("Unique values in '60-89 Days' when '30-59 Days' values are more than or equal to 90:",np.unique(finalTrain[finalTrain['NumberOfTime30-59DaysPastDueNotWorse']>=90]
                                                                                                       ['NumberOfTime60-89DaysPastDueNotWorse']))


print("Unique values in '90 Days' when '30-59 Days' values are more than or equal to 90:",np.unique(finalTrain[finalTrain['NumberOfTime30-59DaysPastDueNotWorse']>=90]
                                                                                                    ['NumberOfTimes90DaysLate']))


print("Unique values in '60-89 Days' when '30-59 Days' values are less than 90:",np.unique(finalTrain[finalTrain['NumberOfTime30-59DaysPastDueNotWorse']<90]
                                                                                           ['NumberOfTime60-89DaysPastDueNotWorse']))


print("Unique values in '90 Days' when '30-59 Days' values are less than 90:",np.unique(finalTrain[finalTrain['NumberOfTime30-59DaysPastDueNotWorse']<90]
                                                                                        ['NumberOfTimes90DaysLate']))


print("Proportion of positive class with special 96/98 values:",
      round(finalTrain[finalTrain['NumberOfTime30-59DaysPastDueNotWorse']>=90]['SeriousDlqin2yrs'].sum()*100/
      len(finalTrain[finalTrain['NumberOfTime30-59DaysPastDueNotWorse']>=90]['SeriousDlqin2yrs']),2),'%')
finalTrain.loc[finalTrain['NumberOfTime30-59DaysPastDueNotWorse'] >= 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 13
finalTrain.loc[finalTrain['NumberOfTime60-89DaysPastDueNotWorse'] >= 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 11
finalTrain.loc[finalTrain['NumberOfTimes90DaysLate'] >= 90, 'NumberOfTimes90DaysLate'] = 17
print("Unique values in 30-59Days", np.unique(finalTrain['NumberOfTime30-59DaysPastDueNotWorse']))
print("Unique values in 60-89Days", np.unique(finalTrain['NumberOfTime60-89DaysPastDueNotWorse']))
print("Unique values in 90Days", np.unique(finalTrain['NumberOfTimes90DaysLate']))
print("Unique values in '30-59 Days' values that are more than or equal to 90:",np.unique(finalTest[finalTest['NumberOfTime30-59DaysPastDueNotWorse']>=90]
                                                                                          ['NumberOfTime30-59DaysPastDueNotWorse']))


print("Unique values in '60-89 Days' when '30-59 Days' values are more than or equal to 90:",np.unique(finalTest[finalTest['NumberOfTime30-59DaysPastDueNotWorse']>=90]
                                                                                                       ['NumberOfTime60-89DaysPastDueNotWorse']))


print("Unique values in '90 Days' when '30-59 Days' values are more than or equal to 90:",np.unique(finalTest[finalTest['NumberOfTime30-59DaysPastDueNotWorse']>=90]
                                                                                                    ['NumberOfTimes90DaysLate']))


print("Unique values in '60-89 Days' when '30-59 Days' values are less than 90:",np.unique(finalTest[finalTest['NumberOfTime30-59DaysPastDueNotWorse']<90]
                                                                                           ['NumberOfTime60-89DaysPastDueNotWorse']))


print("Unique values in '90 Days' when '30-59 Days' values are less than 90:",np.unique(finalTest[finalTest['NumberOfTime30-59DaysPastDueNotWorse']<90]
                                                                                        ['NumberOfTimes90DaysLate']))
finalTest.loc[finalTest['NumberOfTime30-59DaysPastDueNotWorse'] >= 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 19
finalTest.loc[finalTest['NumberOfTime60-89DaysPastDueNotWorse'] >= 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 9
finalTest.loc[finalTest['NumberOfTimes90DaysLate'] >= 90, 'NumberOfTimes90DaysLate'] = 18

print("Unique values in 30-59Days", np.unique(finalTest['NumberOfTime30-59DaysPastDueNotWorse']))
print("Unique values in 60-89Days", np.unique(finalTest['NumberOfTime60-89DaysPastDueNotWorse']))
print("Unique values in 90Days", np.unique(finalTest['NumberOfTimes90DaysLate']))
print('Debt Ratio: \n',finalTrain['DebtRatio'].describe())
print('\nRevolving Utilization of Unsecured Lines: \n',finalTrain['RevolvingUtilizationOfUnsecuredLines'].describe())
quantiles = [0.75,0.8,0.81,0.85,0.9,0.95,0.975,0.99]

for i in quantiles:
    print(i*100,'% quantile of debt ratio is: ',finalTrain.DebtRatio.quantile(i))
finalTrain[finalTrain['DebtRatio'] >= finalTrain['DebtRatio'].quantile(0.95)][['SeriousDlqin2yrs','MonthlyIncome']].describe()
finalTrain[(finalTrain["DebtRatio"] > finalTrain["DebtRatio"].quantile(0.95)) & (finalTrain['SeriousDlqin2yrs'] == finalTrain['MonthlyIncome'])]
finalTrain = finalTrain[-((finalTrain["DebtRatio"] > finalTrain["DebtRatio"].quantile(0.95)) & (finalTrain['SeriousDlqin2yrs'] == finalTrain['MonthlyIncome']))]
finalTrain
finalTrain[finalTrain['RevolvingUtilizationOfUnsecuredLines']>10].describe()
finalTrain[finalTrain['RevolvingUtilizationOfUnsecuredLines']>13].describe()
finalTrain = finalTrain[finalTrain['RevolvingUtilizationOfUnsecuredLines']<=13]
finalTrain
def MissingHandler(df):
    DataMissing = df.isnull().sum()*100/len(df)
    DataMissingByColumn = pd.DataFrame({'Percentage Nulls':DataMissing})
    DataMissingByColumn.sort_values(by='Percentage Nulls',ascending=False,inplace=True)
    return DataMissingByColumn[DataMissingByColumn['Percentage Nulls']>0]

MissingHandler(finalTrain)
finalTrain['MonthlyIncome'].fillna(finalTrain['MonthlyIncome'].median(), inplace=True)
finalTrain['NumberOfDependents'].fillna(0, inplace = True)
MissingHandler(finalTrain)
MissingHandler(finalTest)
finalTest['MonthlyIncome'].fillna(finalTrain['MonthlyIncome'].median(), inplace=True)
finalTest['NumberOfDependents'].fillna(0, inplace = True)
MissingHandler(finalTest)
print(finalTrain.shape)
print(finalTest.shape)
fig = plt.figure(figsize = [15,10])
mask = np.zeros_like(finalTrain.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(finalTrain.corr(), cmap=sns.diverging_palette(150, 275, s=80, l=55, n=9), mask = mask, annot=True, center = 0)
plt.title("Correlation Matrix (HeatMap)", fontsize = 15)
SeriousDlqIn2Yrs = finalTrain['SeriousDlqin2yrs']

finalTrain.drop('SeriousDlqin2yrs', axis = 1 , inplace = True)

finalData = pd.concat([finalTrain, finalTest])

finalData.shape
#New Features

finalData['MonthlyIncomePerPerson'] = finalData['MonthlyIncome']/(finalData['NumberOfDependents']+1)
finalData['MonthlyIncomePerPerson'].fillna(0, inplace=True)

finalData['MonthlyDebt'] = finalData['MonthlyIncome']*finalData['DebtRatio']
finalData['MonthlyDebt'].fillna(finalData['DebtRatio'],inplace=True)
finalData['MonthlyDebt'] = np.where(finalData['MonthlyDebt']==0, finalData['DebtRatio'],finalData['MonthlyDebt'])

finalData['isRetired'] = np.where((finalData['age'] > 65), 1, 0)

finalData['RevolvingLines'] = finalData['NumberOfOpenCreditLinesAndLoans']-finalData['NumberRealEstateLoansOrLines']

finalData['hasRevolvingLines']=np.where((finalData['RevolvingLines']>0),1,0)

finalData['hasMultipleRealEstates'] = np.where((finalData['NumberRealEstateLoansOrLines']>=2),1,0)

finalData['incomeDivByThousand'] = finalData['MonthlyIncome']/1000
finalData.shape
MissingHandler(finalData)
columnList = list(finalData.columns)
columnList

fig = plt.figure(figsize=[20,20])
for col,i in zip(columnList,range(1,19)):
    axes = fig.add_subplot(6,3,i)
    sns.distplot(finalData[col],ax=axes, kde_kws={'bw':1.5}, color='purple')
plt.show()
def SkewMeasure(df):
    nonObjectColList = df.dtypes[df.dtypes != 'object'].index
    skewM = df[nonObjectColList].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending = False)
    skewM=pd.DataFrame({'skew':skewM})
    return skewM[abs(skewM)>0.5].dropna()

skewM = SkewMeasure(finalData)
skewM
for i in skewM.index:
    finalData[i] = special.boxcox1p(finalData[i],0.15) #lambda = 0.15
    
SkewMeasure(finalData)
fig = plt.figure(figsize=[20,20])
for col,i in zip(columnList,range(1,19)):
    axes = fig.add_subplot(6,3,i)
    sns.distplot(finalData[col],ax=axes, kde_kws={'bw':1.5}, color='purple')
plt.show()
trainDF = finalData[:len(finalTrain)]
testDF = finalData[len(finalTrain):]
print(trainDF.shape)
print(testDF.shape)
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(trainDF.to_numpy(),SeriousDlqIn2Yrs.to_numpy(),test_size=0.3,random_state=2020)
lgbAttributes = lgb.LGBMClassifier(objective='binary', n_jobs=-1, random_state=2020, importance_type='gain')

lgbParameters = {
    'max_depth' : [2,3,4,5],
    'learning_rate': [0.05, 0.1,0.125,0.15],
    'colsample_bytree' : [0.2,0.4,0.6,0.8,1],
    'n_estimators' : [400,500,600,700,800,900],
    'min_split_gain' : [0.15,0.20,0.25,0.3,0.35], #equivalent to gamma in XGBoost
    'subsample': [0.6,0.7,0.8,0.9,1],
    'min_child_weight': [6,7,8,9,10],
    'scale_pos_weight': [10,15,20],
    'min_data_in_leaf' : [100,200,300,400,500,600,700,800,900],
    'num_leaves' : [20,30,40,50,60,70,80,90,100]
}

lgbModel = model_selection.RandomizedSearchCV(lgbAttributes, param_distributions = lgbParameters, cv = 5, random_state=2020)

lgbModel.fit(xTrain,yTrain.flatten(),feature_name=trainDF.columns.to_list())
bestEstimatorLGB = lgbModel.best_estimator_
bestEstimatorLGB
bestEstimatorLGB = lgb.LGBMClassifier(colsample_bytree=0.4, importance_type='gain', max_depth=5,
               min_child_weight=6, min_data_in_leaf=600, min_split_gain=0.25,
               n_estimators=900, num_leaves=50, objective='binary',
               random_state=2020, scale_pos_weight=10, subsample=0.9).fit(xTrain,yTrain.flatten(),feature_name=trainDF.columns.to_list())
yPredLGB = bestEstimatorLGB.predict_proba(xTest)
yPredLGB = yPredLGB[:,1]
yTestPredLGB = bestEstimatorLGB.predict(xTest)
print(metrics.classification_report(yTest,yTestPredLGB))
metrics.confusion_matrix(yTest,yTestPredLGB)
LGBMMetrics = pd.DataFrame({'Model': 'LightGBM', 
                            'MSE': round(metrics.mean_squared_error(yTest, yTestPredLGB)*100,2),
                            'RMSE' : round(np.sqrt(metrics.mean_squared_error(yTest, yTestPredLGB)*100),2),
                            'MAE' : round(metrics.mean_absolute_error(yTest, yTestPredLGB)*100,2),
                            'MSLE' : round(metrics.mean_squared_log_error(yTest, yTestPredLGB)*100,2), 
                            'RMSLE' : round(np.sqrt(metrics.mean_squared_log_error(yTest, yTestPredLGB)*100),2),
                            'Accuracy Train' : round(bestEstimatorLGB.score(xTrain, yTrain) * 100,2),
                            'Accuracy Test' : round(bestEstimatorLGB.score(xTest, yTest) * 100,2),
                            'F-Beta Score (β=2)' : round(metrics.fbeta_score(yTest, yTestPredLGB, beta=2)*100,2)},index=[1])

LGBMMetrics
fpr,tpr,_ = metrics.roc_curve(yTest,yPredLGB)
rocAuc = metrics.auc(fpr, tpr)
plt.figure(figsize=(12,6))
plt.title('ROC Curve')
sns.lineplot(fpr, tpr, label = 'AUC for LightGBM Model = %0.2f' % rocAuc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
lgb.plot_importance(bestEstimatorLGB, importance_type='gain')
X = pd.DataFrame(xTrain, columns=trainDF.columns.to_list())

explainer = shap.TreeExplainer(bestEstimatorLGB)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values[1], X)
xgbAttribute = xgs.XGBClassifier(tree_method='gpu_hist',n_jobs=-1, gpu_id=0)

xgbParameters = {
    'max_depth' : [2,3,4,5,6,7,8],
    'learning_rate':[0.05,0.1,0.125,0.15],
    'colsample_bytree' : [0.2,0.4,0.6,0.8,1],
    'n_estimators' : [400,500,600,700,800,900],
    'gamma':[0.15,0.20,0.25,0.3,0.35],
    'subsample': [0.6,0.7,0.8,0.9,1],
    'min_child_weight': [6,7,8,9,10],
    'scale_pos_weight': [10,15,20]
    
}

xgbModel = model_selection.RandomizedSearchCV(xgbAttribute, param_distributions = xgbParameters, cv = 5, random_state=2020)

xgbModel.fit(xTrain,yTrain.flatten())
bestEstimatorXGB = xgbModel.best_estimator_
bestEstimatorXGB
bestEstimatorXGB = xgs.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.25, gpu_id=0,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.125, max_delta_step=0, max_depth=5,
              min_child_weight=9,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=800, n_jobs=-1, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=10, subsample=1,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None).fit(xTrain,yTrain.flatten())
yPredXGB = bestEstimatorXGB.predict_proba(xTest)
yPredXGB = yPredXGB[:,1]

yTestPredXGB = bestEstimatorXGB.predict(xTest)
print(metrics.classification_report(yTest,yTestPredXGB))
metrics.confusion_matrix(yTest,yTestPredXGB)
XGBMetrics = pd.DataFrame({'Model': 'XGBoost', 
                            'MSE': round(metrics.mean_squared_error(yTest, yTestPredXGB)*100,2),
                            'RMSE' : round(np.sqrt(metrics.mean_squared_error(yTest, yTestPredXGB)*100),2),
                            'MAE' : round(metrics.mean_absolute_error(yTest, yTestPredXGB)*100,2),
                            'MSLE' : round(metrics.mean_squared_log_error(yTest, yTestPredXGB)*100,2), 
                            'RMSLE' : round(np.sqrt(metrics.mean_squared_log_error(yTest, yTestPredXGB)*100),2),
                            'Accuracy Train' : round(bestEstimatorLGB.score(xTrain, yTrain) * 100,2),
                            'Accuracy Test' : round(bestEstimatorLGB.score(xTest, yTest) * 100,2),
                            'F-Beta Score (β=2)' : round(metrics.fbeta_score(yTest, yTestPredXGB, beta=2)*100,2)},index=[2])

XGBMetrics
fpr,tpr,_ = metrics.roc_curve(yTest,yPredXGB)
rocAuc = metrics.auc(fpr, tpr)
plt.figure(figsize=(12,6))
plt.title('ROC Curve')
sns.lineplot(fpr, tpr, label = 'AUC for XGBoost Model = %0.2f' % rocAuc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
bestEstimatorXGB.get_booster().feature_names = trainDF.columns.to_list()
xgs.plot_importance(bestEstimatorXGB, importance_type='gain')
# resolve a conflict/bug with latest version of XGBoost and SHAP
mybooster = bestEstimatorXGB.get_booster()
model_bytearray = mybooster.save_raw()[4:]
def myfun(self=None):
    return model_bytearray

mybooster.save_raw = myfun


X = pd.DataFrame(xTrain, columns=trainDF.columns.to_list())

explainer = shap.TreeExplainer(mybooster)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)
frames = [LGBMMetrics, XGBMetrics]
TrainingResult = pd.concat(frames)
TrainingResult.T
lgbProbs = bestEstimatorLGB.predict_proba(testDF)
lgbDF = pd.DataFrame({'ID': testID, 'Probability': lgbProbs[:,1]})
lgbDF.to_csv('submission.csv', index=False)
lgbDF
