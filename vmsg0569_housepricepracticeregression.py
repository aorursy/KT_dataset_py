# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats
# show function will help print data for quick testing

def show(*args):

    for arg in args:

        print(arg, sep='\n')

        

# display option for float values

pd.set_option('display.float_format', lambda x: '%.3f' % x)



# inline printing of matplotlib

%matplotlib inline
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
show(train_data, train_data.shape)
# storing all the columns of train_data

all_columns = train_data.columns

all_X = train_data.columns[:-1]

all_y = train_data.columns[-1:]

show(all_X, all_y, all_X.shape, all_y.shape)
# getting all the null data values in the dataset - X (independent variables)

show(train_data[all_X].isnull().sum().sort_values(ascending=False))

missing_X = train_data[all_X].isnull().sum().sort_values(ascending=False)

missing_X = missing_X[missing_X > 1]

show(missing_X, missing_X.shape)



# We have 19 columns with missing data

missing_X.plot.bar()
# percentage of missing data calculation

missing_X_perc = (train_data[all_X].isnull().sum()/train_data[all_X].isnull().count())*100

missing_X_perc.sort_values(ascending=False)

missing_X_perc[missing_X_perc>0].sort_values(ascending = False)
# verifying that there are no missing data in the SalePrice

train_data[all_y].isnull().sum()
# all the columns with the highest number of missing values

missing_X.index
# drop the missing columns from the dataset

# specifying 1 in the axis to drop the columns

train_data = train_data.drop(missing_X.index, 1)
# data is cleansed now

show(train_data.columns,missing_X)
# removing the 1 null value in the Electrical column

train_data = train_data.drop(train_data[train_data['Electrical'].isnull()].index)
# confirming that no null values exist in the train_data

train_data.isnull().sum().max()

# Analysing the SalePrice
train_data['SalePrice']
train_data['LandContour'].dtype
quan_columns = [col for col in train_data if train_data.dtypes[col] != 'object']

qual_columns = [col for col in train_data if train_data.dtypes[col] == 'object']
show(quan_columns, qual_columns)

show(len(quan_columns) + len(qual_columns))

show(len(quan_columns), len(qual_columns))
# removing Id, SalePrice from the Quantitative Columns

quan_columns.remove('Id')

quan_columns.remove('SalePrice')
show(len(quan_columns),len(qual_columns))
#Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

f = pd.melt(train_data, value_vars = quan_columns)

g = sns.FacetGrid(f, col = 'variable', col_wrap = 3, sharex = False, sharey = False)

g = g.map(sns.distplot, 'value')

#g.savefig('quantitative.png')
quan_columns
qual_columns
for i in train_data[qual_columns]:

    print(i, '\t',train_data[i].unique())
def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)



f = pd.melt(train_data, id_vars=['SalePrice'], value_vars=qual_columns)

g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False, height=5)

g = g.map(boxplot, "value", "SalePrice")



#save as fig

#g.savefig('Qualitative_Data_Box_Plots.png')
def Anova(frame):

    anv = pd.DataFrame()

    # qual_columns are the qualitative data

    anv['features'] = qual_columns

    pvals = []

    for feature in qual_columns:

        samples = []

        for unique_value_per_variable in frame[feature].unique():

            # adding to samples the each row per the unique values per the variable

            s = frame[frame[feature] == unique_value_per_variable]['SalePrice'].values

            samples.append(s)

        # The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. The test is applied to samples from two or more groups, possibly with differing sizes.

        pval = stats.f_oneway(*samples)[1]

        pvals.append(pval)

    anv['pval'] = pvals

    return anv.sort_values('pval')



#building graph

a = Anova(train_data)

a['disparity'] = np.log(1./a['pval'].values)

qual_graph = sns.barplot(data=a, x='features', y='disparity')

x=plt.xticks(rotation=90)

qual_graph
a
show(quan_columns, qual_columns)
train_data.columns
train_data['GrLivArea']
train_data.shape
show(quan_columns, qual_columns)
new_quan_col = quan_columns

new_qual_col = qual_columns
new_quan_col.remove('TotalBsmtSF')
show(len(new_quan_col),new_quan_col)
new_quan_col = ['LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','1stFlrSF','GrLivArea','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','GarageCars','MoSold','YrSold']

new_qual_col = ['Neighborhood', 'ExterQual', 'KitchenQual', 'Foundation', 'HeatingQC']
# verifying new_quan_col + new_qual_col

show(new_quan_col, new_qual_col)
new_train_data = train_data[new_quan_col + new_qual_col]
new_train_data.shape
new_train_data
new_train_data[['MoSold','YrSold']]

new_train_data['MoSold_YrSold'] = new_train_data["MoSold"].astype(str) + new_train_data["YrSold"].astype(str)
show(new_train_data, new_train_data.shape)
new_train_data = new_train_data.drop(['MoSold','YrSold'], axis=1)
show(new_train_data.shape, new_train_data.columns)

show(new_train_data)
# Now we have cleansed data
new_train_data['SalePrice'] = train_data['SalePrice']
new_train_data.shape
#Splitting 17 columns as X and the target 'SalePrice' as y

X = new_train_data.iloc[:,:-1].values

y = new_train_data.iloc[:,17].values
show(X, y)

show(X.shape, y.shape)
new_train_data
# importing LabelEncoder

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X
# applying label encoder on X

X[:,16] = labelencoder_X.fit_transform(X[:, 16])
# Applied label encoder on the columns 1,2,3,4,7 to 16
# Applying one hot encoder on the columsn 1,2,3,4,7 to 16
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()
onehotencoder
onehotencoder_X = OneHotEncoder(categorical_features = [1,2,3,4,7,8,9,10,11,12,13,14,15,16])
#Applying one hot encoder on the required columns

X= onehotencoder_X.fit_transform(X).toarray()
#testing

X.shape
# train test split out of 1459, 323 dataframe

# train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)
show(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# fitting the decision tree regression to the dataset

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred.shape
y_test.shape
# determining error rate between y_pred and y_test

# error function

def error(actual, predicted):

    actual = np.log(actual)

    predicted = np.log(predicted)

    return np.sqrt(np.sum(np.square(actual-predicted))/len(actual))
error(y_test, y_pred)
y_test