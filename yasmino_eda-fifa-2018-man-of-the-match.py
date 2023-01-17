import matplotlib.pyplot as plt
from scipy.stats import skew
import pandas as pd
import numpy as np
import seaborn as sns

%matplotlib inline
import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
data = pd.read_csv('../input/FIFA 2018 Statistics.csv')
data.shape
numerical_features   = data.select_dtypes(include = [np.number]).columns
categorical_features = data.select_dtypes(include= [np.object]).columns
numerical_features
categorical_features
data.describe()
data.hist(figsize=(30,30))
plt.plot()
skew_values = skew(data[numerical_features], nan_policy = 'omit')
pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)
# Missing values
missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])
# encode target variable 'Man of the match' into binary format
data['Man of the Match'] = data['Man of the Match'].map({'Yes': 1, 'No': 0})
sns.countplot(x = 'Man of the Match', data = data)
plt.figure(figsize=(30,10))
sns.heatmap(data[numerical_features].corr(), square=True, annot=True,robust=True, yticklabels=1)
var = ['Man of the Match','Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 
       'Fouls Committed', 'Own goal Time']
corr = data.corr()
corr = corr.filter(items = ['Man of the Match'])
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
var = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 
       'Fouls Committed', 'Own goal Time']
plt.figure(figsize=(15,10))
sns.heatmap((data[var].corr()), annot=True)
var1 = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 'Fouls Committed']
var1.append('Man of the Match')
sns.pairplot(data[var1], hue = 'Man of the Match', palette="husl")
plt.show()
dummy_data = data[var1]
plt.figure(figsize=(20,10))
sns.boxplot(data = dummy_data)
plt.show()
data.drop(['Own goal Time', 'Own goals', '1st Goal'], axis = 1, inplace= True)
categorical_features
def uniqueCategories(x):
    columns = list(x.columns).copy()
    for col in columns:
        print('Feature {} has {} unique values: {}'.format(col, len(x[col].unique()), x[col].unique()))
        print('\n')
uniqueCategories(data[categorical_features].drop('Date', axis = 1))
data.drop('Date', axis = 1, inplace=True)
data.drop(['Corners', 'Fouls Committed', 'On-Target'], axis = 1, inplace=True)
print(data.shape)
data.head()
cleaned_data  = pd.get_dummies(data)
print(cleaned_data.shape)
cleaned_data.head()