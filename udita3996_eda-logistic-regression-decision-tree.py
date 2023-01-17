# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
wine_data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

wine_data.head()
wine_data.shape
#Check for null data

wine_data.isnull().sum()
sns.set()

fig = plt.figure(figsize = [15,20])

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

cnt = 1

for col in cols :

    plt.subplot(4,3,cnt)

    sns.distplot(wine_data[col],hist_kws=dict(edgecolor="k", linewidth=1,color='grey'),color='red')

    cnt+=1

plt.show()  
sns.pairplot(wine_data)

plt.show()
fig = plt.figure(figsize = [15,10])

sns.heatmap(wine_data.corr(),annot = True, cmap = 'Greens', center = 0)

plt.show()
sns.set_style("whitegrid")

fig = plt.figure(figsize = [15,20])

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

cnt = 1

for col in cols :

    plt.subplot(4,3,cnt)

    sns.barplot(data = wine_data, x = 'quality', y = col)

    cnt+=1

plt.show()  
sns.set()

fig = plt.figure(figsize = [15,20])

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

cnt = 1

for col in cols :

    plt.subplot(4,3,cnt)

    sns.boxplot(data = wine_data, y = col)

    cnt+=1

plt.show()
# Assuming a wine with quality > 6.5 is 'good' [1] & others are 'ordinary' [0]

wine_data['quality'] = wine_data.quality.apply(lambda x : 1 if x > 6.5 else 0)
sns.countplot(data = wine_data, x = 'quality')

plt.show()
sns.set()

fig = plt.figure(figsize = [15,20])

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

cnt = 1

for col in cols :

    plt.subplot(4,3,cnt)

    sns.violinplot(x="quality", y=col, data=wine_data)

    cnt+=1

plt.show()
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import classification_report, confusion_matrix
x = wine_data.drop('quality',1)

y = wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)
scaler = StandardScaler()

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

X_train[cols] = scaler.fit_transform(X_train[cols])

X_test[cols] = scaler.fit_transform(X_test[cols])

X_train.head()
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

res = logm1.fit()

res.summary()
logreg = LogisticRegression()

rfe = RFE(logreg, n_features_to_select = 11)

rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]



# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train[col].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = res.predict(sm.add_constant(X_train)).values.reshape(-1)

y_train_pred_final = pd.DataFrame({'quality':y_train.values, 'quality_prob':y_train_pred})

y_train_pred_final.head()
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
draw_roc(y_train_pred_final.quality, y_train_pred_final.quality_prob)
numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.quality_prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.quality, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
y_train_pred_final['final_predicted'] = y_train_pred_final.quality_prob.map( lambda x: 1 if x > 0.18 else 0)

y_train_pred_final.head()
X_test_sm = sm.add_constant(X_test)

y_pred_log = res.predict(X_test_sm)

y_pred_log
predictions_log = pd.DataFrame({'actual_quality' : y_test, 'quality_prob' : y_pred_log})

predictions_log['pred_quality'] = predictions_log.quality_prob.map( lambda x: 1 if x > 0.18 else 0)

predictions_log.drop(['quality_prob'],axis = 1, inplace = True)
metrics.accuracy_score(predictions_log.actual_quality, predictions_log.pred_quality)
confusionMatrix = metrics.confusion_matrix(predictions_log.actual_quality, predictions_log.pred_quality)

confusionMatrix
print(classification_report(predictions_log.actual_quality, predictions_log.pred_quality))
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
# Split the data into training and testing data

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)
# Create a Decision Tree

dt_basic = DecisionTreeClassifier(max_depth=10)

# Fit the training data

dt_basic.fit(x_train,y_train)

# Predict based on test data

y_preds = dt_basic.predict(x_test)
# Calculate Accuracy

accuracy_value = metrics.accuracy_score(y_test,y_preds)

accuracy_value
# Create and print confusion matrix

confusion_matrix(y_test,y_preds)
print(classification_report(y_test,y_preds))
# Calculate the number of nodes in the tree

dt_basic.tree_.node_count
# Create a Parameter grid

param_grid = {

    'max_depth' : range(4,20,4),

    'min_samples_leaf' : range(20,200,40),

    'min_samples_split' : range(20,200,40),

    'criterion' : ['gini','entropy'] 

}

n_folds = 5
dtree = DecisionTreeClassifier()

grid = GridSearchCV(dtree, param_grid, cv = n_folds, n_jobs = -1,return_train_score=True)
grid.fit(x_train,y_train)
cv_result = pd.DataFrame(grid.cv_results_)

cv_result.head()
grid.best_params_
grid.best_score_
best_grid = grid.best_estimator_

best_grid
best_grid.fit(x_train,y_train)
y_preds = best_grid.predict(x_test)
# Calculate Accuracy

accuracy_value = metrics.accuracy_score(y_test,y_preds)

accuracy_value
# Create and print confusion matrix

confusion_matrix(y_test,y_preds)
print(classification_report(y_test,y_preds))