# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
application_train=pd.read_csv('../input/application_train.csv')
bureau=pd.read_csv('../input/bureau.csv')
bureau_balance=pd.read_csv('../input/bureau_balance.csv')
credit_card_balance=pd.read_csv('../input/credit_card_balance.csv')
installments_payments=pd.read_csv('../input/installments_payments.csv')
pos_cash_balance=pd.read_csv('../input/POS_CASH_balance.csv')
previous_application=pd.read_csv('../input/previous_application.csv')
print('the data has beens loaded!')
file_list={ 'application_train': application_train, 
           'bureau': bureau, 
           'bureau_balance': bureau_balance,
           'credit_card_balance': credit_card_balance,
           'installments_payments': installments_payments , 
           'pos_cash_balance': pos_cash_balance,
           'previous_application': previous_application}
for file_name in file_list.keys():
    print('the %s has %d rows , %d columns and %d features' %( file_name, file_list[file_name].shape[0], file_list[file_name].shape[1],len(file_list[file_name].columns.values)))
    
for file_keys,file_values in file_list.items():
    count=file_values.isnull().sum()
    total=file_values.isnull().count()
    percentage=count/total *100
    df=pd.concat([count,percentage],axis=1,keys=['count','percentage'])
    print('The percentage of missing data of features in %s :\n %r \n' %(file_keys, df.sort_values(by=['percentage'],ascending=False).head(10)))
target_1=np.sum(application_train['TARGET']==1)
target_0=np.sum(application_train['TARGET']==0)
plt.figure()
plt.pie([target_0,target_1],explode=[0,0.25],labels=['0','1'],shadow=True,autopct='%0.2f %%')
plt.show()
def plot_categorical_features(file,feature):
    '''
    use this function to plot categorical features
    '''
    plt.figure()
    plt.title('The bar chart of feature: '+ feature)
    data=file[feature].value_counts()
    labels=data.index
    x=list(range(len(data)))
    plt.bar(x,data.values)
    plt.xticks(x,labels,rotation='vertical')
    plt.ylabel('COUNT')
    
    
plot_categorical_features(application_train,'OCCUPATION_TYPE')
plot_categorical_features(application_train,'NAME_EDUCATION_TYPE')
plot_categorical_features(application_train,'NAME_INCOME_TYPE')
def plot_numberical_features(file,feature,bins=50):
    data=file[feature]
    plt.figure()
    plt.title('The distribution of feature: ' + feature)
    sns.distplot(data.dropna(),rug=True,bins=bins)
    plt.show()
    
plot_numberical_features(application_train,'AMT_CREDIT')