import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

sns.set("notebook")

# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
# Glimpse of the train dataset.

df_train.head()
# Glimpse of the test dataset.

df_test.head()
# No. of rows/columns in the train dataset.

df_train.shape
df_test.shape
# Showing the information about the data type in each column of dataframe as well no. of rows.

df_train.info(verbose=True)
df_test.info(verbose=True)
df_train.describe(include = 'all')
df_test.describe(include = 'all')
# Percentage missing values for any cloumn having missing values.

null_columns=df_train.columns[df_train.isnull().any()]

round((100* df_train[null_columns].isnull().sum())/len(df_train),2)
# Percentage missing values for any cloumn having missing values in test data.

null_columns1=df_test.columns[df_test.isnull().any()]

round((100* df_test[null_columns1].isnull().sum())/len(df_test),2)
df_train = df_train.drop(['Cabin', 'Name'],axis = 1)
df_test = df_test.drop(['Cabin','Name'], axis=1)
df_train['Ticket'].describe()
df_train['Ticket'].head(10)
df_train = df_train.drop(['Ticket'], axis = 1)

df_test = df_test.drop(['Ticket'], axis = 1)
df_train['Embarked'].describe()
df_test['Embarked'].describe()
df_train['Embarked'] = df_train['Embarked'].replace(np.nan, "S")

df_test['Embarked'] = df_test['Embarked'].replace(np.nan, "S")
df_train['Age'].describe()
plt.figure(figsize=(12,9))

df_train.Age[df_train.Survived==1].plot(kind='hist', label = 'Survived')

df_train.Age[df_train.Survived==0].plot(kind='hist', alpha = 0.75,label = 'Died')

plt.title("Histogram of Survivors & Non Survivors with respect to Age");

plt.xlabel("Age Bins");

plt.legend();
plt.figure(figsize=(12,9));

df_train.Fare[df_train.Survived==1].plot(kind='hist', label = 'Survived');

df_train.Fare[df_train.Survived==0].plot(kind='hist', alpha = 0.75,label = 'Died');

plt.title("Histogram of Survivors & Non Survivors with respect to Fare");

plt.xlabel("Fare Bins");

plt.legend();
# Selecting numeric columns for outlier analysis and treatment

num_cols = ['Age','Fare']

df_train[num_cols].describe(percentiles=[.25,.5,.75,.90,.95,.99])
sns.catplot(x="Survived", y="Age", hue="Sex",kind="box", data=df_train);
sns.catplot(x="Survived", y="Fare", hue="Sex",kind="box", data=df_train);
# Percentage missing values for any cloumn having missing values.

null_columns=df_train.columns[df_train.isnull().any()]

round((100* df_train[null_columns].isnull().sum())/len(df_train),2)
# Percentage missing values for any cloumn having missing values in test data.

null_columns1=df_test.columns[df_test.isnull().any()]

round((100* df_test[null_columns1].isnull().sum())/len(df_test),2)
df_train['Age'].describe()
df_test['Age'].describe()
df_train["Age"].hist(bins=10)
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].median())

df_test["Age"] = df_test["Age"].fillna(df_test["Age"].median())

## As there is no missing value in Fare cloumn for Train and only 1 value missing in Test, we will fill the missing value in Fare column with median too.

df_train["Fare"] = df_train["Fare"].fillna(df_train["Fare"].median())

df_test["Fare"] = df_test["Fare"].fillna(df_test["Fare"].median())
# Let's check finally of any missing values.

null_columns=df_train.columns[df_train.isnull().any()]

round((100* df_train[null_columns].isnull().sum())/len(df_train),2)
# Percentage missing values for any cloumn having missing values in test data.

null_columns1=df_test.columns[df_test.isnull().any()]

round((100* df_test[null_columns1].isnull().sum())/len(df_test),2)
df_train.shape
df_test.shape
cuts = [0,5,12,18,35,60,100]

labels = ["Infant","Child","Teenager","Young Adult","Adult","Senior"]

df_train['Age_Category'] = pd.cut(df_train['Age'], cuts, labels = labels)

df_test['Age_Category'] = pd.cut(df_test['Age'], cuts, labels = labels)
df_train['Family'] = df_train['SibSp'] + df_train['Parch']

df_test['Family'] = df_test['SibSp'] + df_test['Parch']
## now we can drop the Age column.

df_train = df_train.drop(['Age','SibSp','Parch'], axis=1)

df_test = df_test.drop(['Age','SibSp','Parch'], axis=1)
df_train["Survived"].value_counts()
## Survival Rates

round(100 * sum(df_train["Survived"])/len(df_train["Survived"].index),2)
sns.distplot(df_train['Fare'], bins=10);
df_train["Age_Category"].describe()
plt.figure(figsize=(12,9));

sns.countplot(df_train["Age_Category"]);
sns.catplot('Sex', data = df_train, kind = 'count');

plt.title('Passengers by Sex');
sns.catplot('Pclass', data = df_train, kind = 'count');

plt.title('Passengers by Passenger Class');
sns.catplot('Embarked', data = df_train, kind = 'count');

plt.title('Passengers by Embarked');
sns.countplot(x = "Sex", hue = "Survived", data=df_train);
sns.countplot(x = "Pclass", hue = "Survived", data=df_train);

sns.countplot(x = "Embarked", hue = "Survived", data=df_train);
sns.factorplot(x="Sex", y="Survived",kind='bar', data=df_train);

plt.title('Survival rate by Gender');

plt.ylabel("Survival Rate");
sns.factorplot(x="Pclass", y="Survived", hue = 'Sex',kind='bar', data=df_train);

plt.title('Survival Rate by Class');

plt.ylabel("Survival Rate");
sns.factorplot(x="Embarked", y="Survived", hue = 'Sex',kind='bar', data=df_train)

plt.title('Survival Rate by Port of Embarkment');

plt.ylabel("Survival Rate");
# List of variables to map



varlist =  ["Sex"]



# Defining the map function

def binary_map(x):

    return x.map({'male': 0, "female": 1})



# Applying the function to the housing list

df_train[varlist] = df_train[varlist].apply(binary_map)

df_test[varlist] = df_test[varlist].apply(binary_map)
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(df_train[['Embarked', 'Age_Category']], drop_first=True)

dummy2 = pd.get_dummies(df_test[['Embarked', 'Age_Category']], drop_first=True)



df_train = pd.concat([df_train, dummy1], axis=1)

df_test = pd.concat([df_test, dummy2], axis=1)
# We have created dummies for the below variable, so we can drop it

df_train = df_train.drop(['Embarked', 'Age_Category'], 1)

df_test = df_test.drop(['Embarked', 'Age_Category'], 1)
## Putting the feature variable to X

X_train = df_train.drop(['PassengerId', 'Survived'], axis=1)

X_test = df_test.drop(['PassengerId'], axis=1)

## Putting response variable to y

y_train = df_train['Survived']
X_test.shape
X_train.shape
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['Fare']] = scaler.fit_transform(X_train[['Fare']])
X_test[['Fare']] = scaler.transform(X_test[['Fare']])
X_train.head()
X_test.head()
y_train.head()
# Let's see the correlation matrix 

plt.figure(figsize = (20,10))        # Size of the figure

sns.heatmap(df_train.corr(),annot = True)

plt.show()
## Importing Statsmodel library for modelling Logistic Regression

import statsmodels.api as sm
# Logistic regression model

logm = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

res = logm.fit()

print(res.summary())
col = X_train.columns

col
col1 = col.drop("Embarked_Q", 1)
X_train_sm1 = sm.add_constant(X_train[col1])

logm1 = sm.GLM(y_train,X_train_sm1, family = sm.families.Binomial())

res1 = logm1.fit()

print(res1.summary())
col2 = col1.drop("Fare", 1)
X_train_sm2 = sm.add_constant(X_train[col2])

logm2 = sm.GLM(y_train,X_train_sm2, family = sm.families.Binomial())

res2 = logm2.fit()

print(res2.summary())
# Getting the predicted values on the train set

y_train_pred = res2.predict(X_train_sm2)

y_train_pred.head(10)
y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final = pd.DataFrame({'Survived':y_train.values, 'Survival_Prob':y_train_pred})

y_train_pred_final['PassengerId'] = df_train['PassengerId']

y_train_pred_final.head()
y_train_pred_final['Predicted'] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
# Confusion Matrix

from sklearn import metrics

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.Predicted)

print(confusion)
#### Predicted       not_Survived       Survived

#### Actual

#### not_Survived         424             74

#### Survived              94            211 
# Let's check the overall accuracy.

metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Predicted)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['Features'] = X_train[col2].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col2].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by='VIF', ascending=False)

vif
col3 = col2.drop("Pclass",1)

col3
# Let's re-run the model using the selected variables

X_train_sm3 = sm.add_constant(X_train[col3])

logm3 = sm.GLM(y_train,X_train_sm3, family = sm.families.Binomial())

res3 = logm3.fit()

print(res3.summary())
# Getting the predicted values on the train set

y_train_pred = res3.predict(X_train_sm3)

y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final["Survival_Prob"] = y_train_pred
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['Predicted'] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Predicted))
vif = pd.DataFrame()

vif['Features'] = X_train[col3].columns

vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col3].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
col4 = col3.drop("Sex",1)

col4
# Let's re-run the model using the selected variables

X_train_sm4 = sm.add_constant(X_train[col4])

logm4 = sm.GLM(y_train,X_train_sm4, family = sm.families.Binomial())

res4 = logm4.fit()

print(res4.summary())
# Getting the predicted values on the train set

y_train_pred = res4.predict(X_train_sm4)

y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred[:10]
y_train_pred_final["Survival_Prob"] = y_train_pred
# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['Predicted'] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Predicted))
# Getting the predicted values on the train set

y_train_pred = res3.predict(X_train_sm3)

y_train_pred = y_train_pred.values.reshape(-1)

y_train_pred_final["Survival_Prob"] = y_train_pred

# Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['Predicted'] = y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()
# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.Predicted))
# Let's take a look at the confusion matrix again 

confusion = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final.Predicted )

confusion
TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)
# Let us calculate specificity

TN / float(TN+FP)
# Calculate false positive rate - predicting Surival when passenger does not have survived

print(FP/ float(TN+FP))
# positive predictive value 

print (TP / float(TP+FP))
# Negative predictive value

print (TN / float(TN+ FN))
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
fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Survived, y_train_pred_final.Survival_Prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Survived, y_train_pred_final.Survival_Prob)
### Let's create columns with different probability cutoffs 

numbers = [float(x)/10 for x in range(10)]

for i in numbers:

    y_train_pred_final[i]= y_train_pred_final.Survival_Prob.map(lambda x: 1 if x > i else 0)

y_train_pred_final.head()
# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.

cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

from sklearn.metrics import confusion_matrix



# TP = confusion[1,1] # true positive 

# TN = confusion[0,0] # true negatives

# FP = confusion[0,1] # false positives

# FN = confusion[1,0] # false negatives



num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:

    cm1 = metrics.confusion_matrix(y_train_pred_final.Survived, y_train_pred_final[i] )

    total1=sum(sum(cm1))

    accuracy = (cm1[0,0]+cm1[1,1])/total1

    

    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)
# Let's plot accuracy sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

plt.show()
precision = confusion[1,1]/(confusion[0,1]+confusion[1,1])
recall = confusion[1,1]/(confusion[1,0]+confusion[1,1])

(precision, recall)
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_pred_final.Survived, y_train_pred_final.Predicted)

recall_score(y_train_pred_final.Survived, y_train_pred_final.Predicted)
from sklearn.metrics import precision_recall_curve

p, r, thresholds = precision_recall_curve(y_train_pred_final.Survived, y_train_pred_final.Survival_Prob)
plt.plot(thresholds, p[:-1], "g-")

plt.plot(thresholds, r[:-1], "r-")

plt.show()
X_test = X_test[col3]
X_test.head()
X_test_sm = sm.add_constant(X_test)
y_test_pred = res3.predict(X_test_sm)
y_test_pred.head()
# Converting y_pred to a dataframe which is an array

y_test_pred_final = pd.DataFrame(y_test_pred)
# Putting PassengerId 

y_test_pred_final['PassengerId']= df_test['PassengerId']


y_test_pred_final.shape
y_test_pred_final.rename(columns={0: "Survival_Prob"}, inplace=True)

y_test_pred_final.describe()
y_test_pred_final['Survived'] = y_test_pred_final.Survival_Prob.map(lambda x: 1 if x > 0.5 else 0)
submission = y_test_pred_final.drop('Survival_Prob', axis=1)
submission
submission.to_csv("submission.csv",index=False)