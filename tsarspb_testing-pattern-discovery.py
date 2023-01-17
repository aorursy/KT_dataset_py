# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data_mut = pd.read_csv('../input/train.csv',index_col=None,sep=",")
data_apr=data_mut[['Survived','Sex','Pclass']]
print("**************** checking the initial data info ********************")
#print(data_apr.info())
#print(data_apr.describe())
#print(data_apr.head())

codes = {'male':1, 'female':0}
data_apr['Sex'] = data_apr['Sex'].map(codes)
dummies = pd.get_dummies(data_apr["Pclass"],prefix='class')
data_apr=pd.concat([data_apr,dummies], axis=1)
dummies=pd.get_dummies(data_apr["Sex"],prefix='sex')
data_apr=pd.concat([data_apr,dummies], axis=1)
dummies=pd.get_dummies(data_apr["Survived"],prefix='survived')
data_apr=pd.concat([data_apr,dummies], axis=1)

data_apr=data_apr.drop(['Pclass','Sex','Survived'],axis=1)

print("**************** getting the data info after the data preparation *******************")
#print(data_apr.info())
#print(data_apr.describe())
print(data_apr.head())

#getting the itemsets from apriori
frequent_itemsets = apriori(data_apr, min_support=0.0001, use_colnames=True)
print("**************** printing the itemsets *********************")
print(frequent_itemsets.sort_values(['support'],ascending=False))

#getting the rules info
print("**************** getting the resulting info for data with high confidence with survived as consequents ******************")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
conf_data = rules[(rules["consequents"]== {'survived_0'}) | (rules["consequents"]== {'survived_1'})].sort_values(['confidence'],ascending=False)
print(conf_data[['antecedents','consequents','confidence']])

print("**************** getting the resulting info for data with high confidence with survived as antecedents ******************")
conf_data = rules[(rules["antecedents"]== {'survived_0'}) | (rules["antecedents"]== {'survived_1'})].sort_values(['confidence'],ascending=False)
print(conf_data[['antecedents','consequents','confidence']])


# Any results you write to the current directory are saved as output.
