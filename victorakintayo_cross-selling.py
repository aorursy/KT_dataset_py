# Data Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, f1_score, recall_score, precision_score


from sklearn.model_selection import RandomizedSearchCV
df_train = pd.read_csv('../input/dataset/train.csv')
df_test = pd.read_csv('../input/dataset/test.csv')
sample_submission = pd.read_csv('../input/sample/sample_submission.csv')


df_train.head()
df_test.head()
df_train.describe()
df_train.info()
df_train.isna().sum()
df_test.isna().sum()
df_train['Gender'].value_counts()
df_train = df_train.drop(['id'], axis=1)
df_test = df_test.drop(['id'], axis=1)
df_train.head()
df_train['Response'].value_counts()

R_plot = sns.countplot(x='Response', data=df_train)
R_plot.set_title("Response distribution", fontsize=15)
plt.xlabel("0 = Not interested, 1 = Interested")
R_plot.set_ylabel("Count", fontsize=15);
pd.crosstab(df_train['Response'], df_train['Gender'], margins=True, margins_name='Totals')
pd.crosstab(df_train['Response'], df_train['Gender']).plot(kind="bar", figsize=(8,4), color=["dodgerblue", "orange"])

plt.title("Response distribution by Gender")
plt.xlabel("0 = Not interested, 1 = Interested")
plt.ylabel("Count")
plt.legend(["Female", "Male"])
df_train['Driving_License'].value_counts()
df_train = df_train.drop("Driving_License", axis=1)
df_test = df_test.drop("Driving_License",axis=1)
df_train.head()
pd.crosstab(df_train['Response'], df_train['Previously_Insured'],margins=True, margins_name='Total')
pd.crosstab(df_train['Response'], df_train['Previously_Insured']).plot(kind="bar", figsize=(10,6), color=sns.color_palette("colorblind"))

plt.title("Response distribution for Previously_Insured")
plt.xlabel("0 = Not interested, 1 = Interested")
plt.ylabel("Count")
plt.legend(["Customer has no Vehicle Insurance Subscription", "Customer has Vehicle Insurance Subscription "])
plt.xticks(rotation=0);
pd.crosstab(df_train['Response'], df_train['Vehicle_Age'], margins=True, margins_name='Total')
pd.crosstab(df_train['Response'], df_train['Vehicle_Age']).plot(kind="bar", figsize=(10,6), color=sns.color_palette("colorblind"))

plt.title("Response distribution by Vehicle Age")
plt.xlabel("0 = Not interested, 1 = Interested")
plt.ylabel("Count")
plt.legend(["1-2 Year", "< 1 Year", "> 2 Years"])
plt.xticks(rotation=0);
pd.crosstab(df_train['Response'], df_train['Vehicle_Damage'], margins=True, margins_name='Total')
pd.crosstab(df_train['Response'], df_train['Vehicle_Damage']).plot(kind="bar", figsize=(10,6), color=sns.color_palette("colorblind"))

plt.title("Response distribution by Vehicle_Damage")
plt.xlabel("0 = Not interested, 1 = Interested")
plt.ylabel("Count")
plt.legend(["Vehicle Damage", "No Vehicle Damage"])
plt.xticks(rotation=0);
pd.crosstab(df_train['Previously_Insured'], df_train['Vehicle_Damage'], margins=True, margins_name='Total')
pd.crosstab(df_train['Previously_Insured'], df_train['Vehicle_Damage']).plot(kind="bar", figsize=(10,6), color=sns.color_palette("colorblind"))

plt.title("Previously Insured vs Vehicle_Damage")
plt.xlabel("0 = Not Insured, 1 = Insured")
plt.ylabel("Count")
plt.legend(["Vehicle Damage", "No Vehicle Damage"])
plt.xticks(rotation=0);
df_train['Annual_Premium'].describe()
df_train['Policy_Sales_Channel'].describe()
df_train['Vintage'].describe()
df_train.info()
#Convert datatypes
df_test['Vehicle_Age']=df_test['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
df_test['Gender']=df_test['Gender'].replace({'Male':1,'Female':0})
df_test['Vehicle_Damage']=df_test['Vehicle_Damage'].replace({'Yes':1,'No':0})

df_train['Vehicle_Age']=df_train['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
df_train['Gender']=df_train['Gender'].replace({'Male':1,'Female':0})
df_train['Vehicle_Damage']=df_train['Vehicle_Damage'].replace({'Yes':1,'No':0})


df_train['Region_Code']=df_train['Region_Code'].astype(int)
df_test['Region_Code']=df_test['Region_Code'].astype(int)
df_train['Policy_Sales_Channel']=df_train['Policy_Sales_Channel'].astype(int)
df_test['Policy_Sales_Channel']=df_test['Policy_Sales_Channel'].astype(int)
len(df_test)/(len(df_test)+len(df_train))
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
eval_set = [(X_train, y_train), (X_test, y_test)]
print(len(train_set), "train + ", len(test_set), "test")
from xgboost import XGBClassifier
xgboost=XGBClassifier()
xgboost=xgboost.fit(X_train,y_train, eval_metric=["error", "logloss"], eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
col_1=['Gender', 'Age', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
cat_col=['Gender', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = xgboost.predict(X_test)
xgboost=xgboost.fit(X_train,y_train, eval_metric=["error", "logloss"], eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
probs_y_train = xgboost.predict_proba(X_train)[:, 1]
probs_y_test = xgboost.predict_proba(X_test)[:, 1]
# evaluate predictions
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgboost, X = X_train, y = y_train, cv = 10)
print("Accuracy:{:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation:{:.2f} %".format(accuracies.std()*100))
results = xgboost.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)


from matplotlib import pyplot

# plot log loss
fig, ax = pyplot.subplots(figsize=(12,12))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()

# plot classification error
fig, ax = pyplot.subplots(figsize=(12,12))
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()
Important_features = pd.Series(xgboost.feature_importances_, index=X_t.columns)
Important_features.nlargest(15).plot(kind='barh')
plt.show()
y_pred= xgboost.predict_proba(df_test)[:, 1]
predictions = [round(value) for value in y_pred]
sample_submission['Response']=predictions
sample_submission.to_csv("Predictions.csv", index = False)