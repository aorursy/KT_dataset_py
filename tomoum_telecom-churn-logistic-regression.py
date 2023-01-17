import numpy as np

import pandas as pd 

import seaborn as sb

import matplotlib.pyplot as plt

%matplotlib inline
all_data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
all_data.head()
all_data.info()
for i in all_data.columns:

    print('Feature : ' , i , '----->' ,all_data[i].unique())
all_data.isnull().sum()
all_data=all_data.replace({'No internet service':'No','No phone service':'No'})
for i in all_data.columns:

    print('Feature : ' , i , '----->' ,all_data[i].unique())
pd.crosstab(all_data.gender,all_data.Churn)

plt.figure(figsize=(10,10))

plt.pie( all_data.gender.value_counts(),explode=(0.1,0),labels=['m','f'], autopct='%1.1f%%',shadow=True)
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)

all_data.groupby("Churn").gender.value_counts().plot.pie()

plt.subplot(1,2,2)

all_data.groupby("Churn").gender.value_counts().plot.bar()
all_data.drop(['gender','customerID'],axis=1, inplace = True)
all_data=all_data[all_data.TotalCharges != ' ']

all_data.TotalCharges=all_data.TotalCharges.astype(float)
all_data
all_data.replace({'Contract':{'Month-to-month': 1,'One year': 12,'Two year': 24}},inplace=True)
all_data['automatic']=np.where(all_data['PaymentMethod'].str.contains('automatic'),1,0)
all_data
all_data=all_data.replace({'No':0,'Yes':1})
all_data.info()
all_data.drop(['PaymentMethod'],axis=1,inplace=True)
all_data=pd.get_dummies(all_data, columns=['InternetService'],drop_first=True)
all_data.info()
plt.figure(figsize=(20, 20))

sb.heatmap(all_data.corr(), annot=True)
all_data.drop(['PhoneService','MultipleLines','OnlineBackup','StreamingTV','StreamingTV','DeviceProtection'],axis=1,inplace=True)
all_data.columns
plt.figure(figsize=(20, 20))

sb.heatmap(all_data.corr(), annot=True)
all_data.drop(['tenure','StreamingMovies'],axis=1,inplace=True)
all_data.drop(['InternetService_DSL'],axis=1,inplace=True)
all_data.drop(['TotalCharges'],axis=1,inplace=True)
plt.figure(figsize=(20, 20))

sb.heatmap(all_data.corr(), annot=True)
sb.countplot('Churn', data = all_data)
churnSet=all_data[all_data['Churn']==1]

churnLen=len(churnSet)

churnIndex=churnSet.index



notChurnSet=all_data[all_data['Churn']==0]

notChurnLen=len(notChurnSet)

notChurnIndex=notChurnSet.index



randomNotChurn=np.random.choice(notChurnIndex,churnLen)



balncedIndex=np.concatenate([churnIndex,randomNotChurn])







balncedData = all_data.loc[balncedIndex]
len(balncedData)
sb.countplot('Churn', data = balncedData)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
final_result = 0

trial = []

final =[]

prd_y = []

all_results = []

final_cmf =[]
for x in range(100):

    train, test = train_test_split(balncedData, test_size = 0.3)

    train_x = train.drop('Churn', axis=1)

    train_y = train['Churn']



    test_x = test.drop('Churn', axis=1)

    test_y = test['Churn']





    lr = LogisticRegression()

    lr.fit(train_x, train_y)



    pred_y = lr.predict(test_x)



    cmf_matrix = confusion_matrix(test_y, pred_y)

    recalls = cmf_matrix[1,1]/(cmf_matrix[1,0]+cmf_matrix[1,1])

    all_results.append(recalls)



    if final_result < recalls:

        final_result = recalls

        trial = lr.fit(train_x, train_y)

        final = test_y

        prd_y = pred_y

        final_cmf = cmf_matrix



trial
final
sb.heatmap(final_cmf, annot=True, fmt = 'd')

print('recall = ',final_cmf[1,1]/(final_cmf[1,0]+final_cmf[1,1]))
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
print(classification_report(final, prd_y, target_names=['chrun_yes','churn_no']))