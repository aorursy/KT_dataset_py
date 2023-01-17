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
pd.pandas.set_option('display.max_columns',None)
dataset=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



## print shape of dataset with rows and columns

print(dataset.shape)
## print the top5 records

dataset.head()
## Here we will check the percentage of nan values present in each feature

## 1 -step make the list of features which has missing values

features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]

## 2- step print the feature name and the percentage of missing values



for feature in features_with_na:

    print(feature, np.round(dataset[feature].isnull().mean(), 4),  ' % missing values')
import matplotlib.pyplot as plt

import seaborn as sns

for feature in features_with_na:

    data = dataset.copy()

    

    # let's make a variable that indicates 1 if the observation was missing or zero otherwise

    data[feature] = np.where(data[feature].isnull(), 1, 0)

    

    # let's calculate the mean SalePrice where the information is missing or present

    data.groupby(feature)['SalePrice'].median().plot.bar(color=['red','green','orange','cyan'])

    plt.title(feature)

    plt.show()
print("Id of Houses {}".format(len(dataset.Id)))
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']



print('Number of numerical variables: ', len(numerical_features))



# visualise the numerical variables

dataset[numerical_features].head()
# list of variables that contain year information

year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]



year_feature
# let's explore the content of these year variables

for feature in year_feature:

    print(feature, dataset[feature].unique())
## Lets analyze the Temporal Datetime Variables

## We will check whether there is a relation between year the house is sold and the sales price



dataset.groupby('YrSold')['SalePrice'].median().plot()

plt.xlabel('Year Sold')

plt.ylabel('Median House Price')

plt.title("House Price vs YearSold")
year_feature
## Here we will compare the difference between All years feature with SalePrice



for feature in year_feature:

    if feature!='YrSold':

        data=dataset.copy()

        ## We will capture the difference between year variable and year the house was sold for

        data[feature]=data['YrSold']-data[feature]



        plt.scatter(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalePrice')

        plt.show()

## Numerical variables are usually of 2 type

## 1. Continous variable and Discrete Variables



discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]

print("Discrete Variables Count: {}".format(len(discrete_feature)))

discrete_feature


dataset[discrete_feature].head()


## Lets Find the realtionship between them and Sale PRice



for feature in discrete_feature:

    data=dataset.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar(color=['red','green','pink','blue','orange','brown','cyan','violet'])

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]

print("Continuous feature Count {}".format(len(continuous_feature)))
## Lets analyse the continuous values by creating histograms to understand the distribution



for feature in continuous_feature:

    data=dataset.copy()

    data[feature].hist(bins=25)

    plt.xlabel(feature)

    plt.ylabel("Count")

    plt.title(feature)

    plt.show()
## We will be using logarithmic transformation





for feature in continuous_feature:

    data=dataset.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature]=np.log(data[feature])

        data['SalePrice']=np.log(data['SalePrice'])

        plt.scatter(data[feature],data['SalePrice'])

        plt.xlabel(feature)

        plt.ylabel('SalesPrice')

        plt.title(feature)

        plt.show()
for feature in continuous_feature:

    data=dataset.copy()

    if 0 in data[feature].unique():

        pass

    else:

        data[feature]=np.log(data[feature])

        data.boxplot(column=feature)

        plt.ylabel(feature)

        plt.title(feature)

        plt.show()
categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']

categorical_features
dataset[categorical_features].head()
for feature in categorical_features:

    print('The feature is {} and number of categories are {}'.format(feature,len(dataset[feature].unique())))
## Find out the relationship between categorical variable and dependent feature SalesPrice

for feature in categorical_features:

    data=dataset.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar(color=['red','green','pink','blue','orange','brown','cyan','violet'])

    plt.xlabel(feature)

    plt.ylabel('SalePrice')

    plt.title(feature)

    plt.show()
## Always remember there way always be a chance of data leakage so we need to split the data first and then apply feature

## Engineering

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(dataset,dataset['SalePrice'],test_size=0.1,random_state=0)

X_train.shape, X_test.shape
## Let us capture all the nan values

## First lets handle Categorical features which are missing

features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']



for feature in features_nan:

    print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))
## Replace missing value with a new label

def replace_cat_feature(dataset,features_nan):

    data=dataset.copy()

    data[features_nan]=data[features_nan].fillna('Missing')

    return data



dataset=replace_cat_feature(dataset,features_nan)



dataset[features_nan].isnull().sum()
dataset[features_nan].head()
numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']



## We will print the numerical nan variables and percentage of missing values



for feature in numerical_with_nan:

    print("{}: {}% missing value".format(feature,np.around(dataset[feature].isnull().mean(),4)))
for feature in numerical_with_nan:

    median=dataset[feature].median()

    dataset[feature+'_nan']=np.where(dataset[feature].isnull(),1,0)

    dataset[feature].fillna(median,inplace=True)



dataset[numerical_with_nan].isnull().sum()
dataset.head(20)
year_feature


for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:

       

    dataset[feature]=dataset['YrSold']-dataset[feature]
dataset.head()


dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()
dataset.head()
skewed=['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']

dataset[skewed].head()
for feature in skewed:

    dataset[feature]=np.log(dataset[feature])



dataset[skewed].head()
cat_fea=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']

cat_fea
for feature in cat_fea:

    temp=dataset.groupby(feature)['SalePrice'].count()/len(dataset)

    temp_df=temp[temp>0.01].index

    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')
dataset[cat_fea].head(50)