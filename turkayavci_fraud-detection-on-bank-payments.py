# Necessary imports



## Data loading, processing and for more

import pandas as pd

import numpy as np

from imblearn.over_sampling import SMOTE



## Visualization

import seaborn as sns

import matplotlib.pyplot as plt

# set seaborn style because it prettier

sns.set()



## Metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc



## Models

import xgboost as xgb

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier
# read the data and show first 5 rows

data = pd.read_csv("../input/bs140513_032310.csv")

data.head(5)
data.info()
# Create two dataframes with fraud and non-fraud data 

df_fraud = data.loc[data.fraud == 1] 

df_non_fraud = data.loc[data.fraud == 0]



sns.countplot(x="fraud",data=data)

plt.title("Count of Fraudulent Payments")

plt.show()

print("Number of normal examples: ",df_non_fraud.fraud.count())

print("Number of fradulent examples: ",df_fraud.fraud.count())

#print(data.fraud.value_counts()) # does the same thing above
print("Mean feature values per category",data.groupby('category')['amount','fraud'].mean())
# Create two dataframes with fraud and non-fraud data 

pd.concat([df_fraud.groupby('category')['amount'].mean(),df_non_fraud.groupby('category')['amount'].mean(),\

           data.groupby('category')['fraud'].mean()*100],keys=["Fraudulent","Non-Fraudulent","Percent(%)"],axis=1,\

          sort=False).sort_values(by=['Non-Fraudulent'])
# Plot histograms of the amounts in fraud and non-fraud data 

plt.figure(figsize=(30,10))

sns.boxplot(x=data.category,y=data.amount)

plt.title("Boxplot for the Amount spend in category")

plt.ylim(0,4000)

plt.legend()

plt.show()
# Plot histograms of the amounts in fraud and non-fraud data 

plt.hist(df_fraud.amount, alpha=0.5, label='fraud',bins=100)

plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud',bins=100)

plt.title("Histogram for fraudulent and nonfraudulent payments")

plt.ylim(0,10000)

plt.xlim(0,1000)

plt.legend()

plt.show()
print((data.groupby('age')['fraud'].mean()*100).reset_index().rename(columns={'age':'Age','fraud' : 'Fraud Percent'}).sort_values(by='Fraud Percent'))
print("Unique zipCodeOri values: ",data.zipcodeOri.nunique())

print("Unique zipMerchant values: ",data.zipMerchant.nunique())

# dropping zipcodeori and zipMerchant since they have only one unique value

data_reduced = data.drop(['zipcodeOri','zipMerchant'],axis=1)
data_reduced.columns
# turning object columns type to categorical for easing the transformation process

col_categorical = data_reduced.select_dtypes(include= ['object']).columns

for col in col_categorical:

    data_reduced[col] = data_reduced[col].astype('category')

# categorical values ==> numeric values

data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)

data_reduced.head(5)
X = data_reduced.drop(['fraud'],axis=1)

y = data['fraud']

print(X.head(),"\n")

print(y.head())
y[y==1].count()
sm = SMOTE(random_state=42)

X_res, y_res = sm.fit_resample(X, y)

y_res = pd.DataFrame(y_res)

print(y_res[0].value_counts())
# I won't do cross validation since we have a lot of instances

X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.3,random_state=42,shuffle=True,stratify=y_res)
# %% Function for plotting ROC_AUC curve



def plot_roc_auc(y_test, preds):

    '''

    Takes actual and predicted(probabilities) as input and plots the Receiver

    Operating Characteristic (ROC) curve

    '''

    fpr, tpr, threshold = roc_curve(y_test, preds)

    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
# The base score should be better than predicting always non-fraduelent

print("Base accuracy score we must beat is: ", 

      df_non_fraud.fraud.count()/ np.add(df_non_fraud.fraud.count(),df_fraud.fraud.count()) * 100)
# %% K-ello Neigbors



knn = KNeighborsClassifier(n_neighbors=5,p=1)



knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)





print("Classification Report for K-Nearest Neighbours: \n", classification_report(y_test, y_pred))

print("Confusion Matrix of K-Nearest Neigbours: \n", confusion_matrix(y_test,y_pred))

plot_roc_auc(y_test, knn.predict_proba(X_test)[:,1])
# %% Random Forest Classifier



rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,

                                verbose=1,class_weight="balanced")



rf_clf.fit(X_train,y_train)

y_pred = rf_clf.predict(X_test)



print("Classification Report for Random Forest Classifier: \n", classification_report(y_test, y_pred))

print("Confusion Matrix of Random Forest Classifier: \n", confusion_matrix(y_test,y_pred))

plot_roc_auc(y_test, rf_clf.predict_proba(X_test)[:,1])
XGBoost_CLF = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, 

                                objective="binary:hinge", booster='gbtree', 

                                n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 

                                subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 

                                scale_pos_weight=1, base_score=0.5, random_state=42, verbosity=True)



XGBoost_CLF.fit(X_train,y_train)



y_pred = XGBoost_CLF.predict(X_test)



print("Classification Report for XGBoost: \n", classification_report(y_test, y_pred))

print("Confusion Matrix of XGBoost: \n", confusion_matrix(y_test,y_pred))

plot_roc_auc(y_test, XGBoost_CLF.predict_proba(X_test)[:,1])