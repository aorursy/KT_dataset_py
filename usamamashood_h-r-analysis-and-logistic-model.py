import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
import os
folder_name = os.listdir('../input/')[0]

folder_name
df = pd.read_csv(f'../input/{folder_name}/HR_comma_sep.csv')



df

df.describe(percentiles=[.05,.25,.5,.75,.90,.95,.98,1])

df.info()
df.isnull().sum()
df.count()
df.nunique()
df.corr()
plt.figure(figsize=(10,10))



sns.heatmap(df.corr(),annot=True)
pd.value_counts(df['salary'])
salary_value = np.array(pd.value_counts(df['salary']).index)

salary_value
df1  = df.copy()
df1['salary'] = df1['salary'].map({salary_value[0]:0,salary_value[1]:1,salary_value[2]:2})
df1
pd.value_counts(df['Department'])
department_values = np.array(pd.value_counts(df['Department']).index)

department_values
df.columns =[i.lower() for i in df.columns]

df1.columns =[i.lower() for i in df1.columns]
df.columns
df1.columns
dummy = pd.get_dummies(df1['department'],drop_first=True)
dummy.columns = [i.lower() for i in dummy.columns]
dummy.head(10)
df1 = pd.concat([df1,dummy],axis=1)
df1
df1 = df1.drop(['department'],axis=1)
df1
left_0 = df[df['left']==0]

left_0
left_1 = df[df['left']==1]

left_1
count_dic = {0 : pd.value_counts(left_0['salary']).to_numpy(),

             1 : pd.value_counts(left_1['salary']).to_numpy()}
count_dic['name'] =  salary_value

count_dic
fig = plt.figure()

x = np.arange(0,len(salary_value))

ax = fig.add_axes([0,0,1,1])

ax.bar(x+0, count_dic[0],width = 0.25 )

ax.bar(x+0.25, count_dic[1],width = 0.25)

ax.set_yscale('log')

ax.set_xticks([0.125,1.125,2.125])

ax.set_xticklabels(salary_value)

ax.legend([0,1])

plt.show()
dummy['left'] = df['left']
plt.figure(figsize=(10,10))



sns.heatmap(dummy.corr(),annot=True)
dummy_salary = pd.get_dummies(df['salary'])

dummy_salary['left'] = df['left']

dummy_salary.corr()
plt.figure(figsize=(10,10))



sns.heatmap(dummy_salary.corr(),annot=True)
fig,plot = plt.subplots(1,2,figsize=(16,6))



plot[0].bar(x,dummy_salary.corr()['left'][0:3].values,color='c')

plot[0].set_xticks(x)

plot[0].set_xticklabels(salary_value)

plot[0].set_xlabel('salary')

plot[0].set_ylabel('correlation')

plot[0].set_title('left/salary relation')

# plot[0] = dummy_salary.corr()['left'][0:3].plot(kind='bar')

# plot[1]=dummy.corr()['left'][0:9].plot(kind='bar')

plot[1].bar(np.arange(0,9),dummy.corr()['left'][0:9].values,color='m')

plot[1].set_xticks(np.arange(0,9))

plot[1].set_xticklabels(dummy.columns[0:9],rotation='vertical')

plot[1].set_xlabel('department')

plot[1].set_ylabel('correlation')

plot[1].set_title('left/department relation')

plt.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df1.drop('left', axis=1),df1[['left']] , test_size=0.3, random_state=100)

import statsmodels.api as sm
x_train_constant = sm.add_constant(x_train)
model = sm.GLM(y_train, x_train_constant, family=sm.families.Binomial())

model=model.fit()

model.summary()
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
logr = RFE(LogisticRegression(),10)

logr = logr.fit(x_train,y_train)

# logr.support_
rfe_col = x_train.columns[logr.support_]
x_train_constant = sm.add_constant(x_train[rfe_col])

model = sm.GLM(y_train, x_train_constant, family=sm.families.Binomial())

model=model.fit()

model.summary()
def vif(data):

    data_frame = pd.DataFrame(columns=['col_name','vif'])

    x_var_name = data.columns

    for i in range(len(x_var_name)):

        y_temp = data[x_var_name[i]]

        x_temp = data.drop(x_var_name[i],axis=1)

        r2 = sm.OLS(y_temp,x_temp).fit().rsquared

        vif = round(1/(1-r2),2)

        data_frame.loc[i] = [x_var_name[i],vif]

    return data_frame.sort_values(by='vif', ascending=False)  
x_train=x_train[rfe_col]

vif(x_train)
main_model = LogisticRegression()

main_model.fit(x_train,y_train)
x_test = x_test[rfe_col]
pred = main_model.predict(x_test)
pred_prob = main_model.predict_proba(x_test)
pred_prob
prediction = y_test

prediction.index = np.arange(0,len(y_test))
prediction['pred_prob']= pred_prob[:,0]
prediction['pred'] = prediction.pred_prob.apply(lambda x:0 if x>=0.5 else 1)
prediction
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(prediction['left'],prediction['pred'])
accuracy_score(y_test['left'],pred)
confusion_matrix = confusion_matrix(prediction['left'],prediction['pred'])

confusion_matrix
TN = confusion_matrix[0,0]

FP = confusion_matrix[0,1]

TP = confusion_matrix[1,1]

FN = confusion_matrix[1,0]
TP/float(TP+TN) #recall, sensitivity
TN/float(FP+TN) #specificity