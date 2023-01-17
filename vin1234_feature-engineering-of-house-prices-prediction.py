import pandas as pd                  # Pandas for data analysis 

import numpy as np                   # Numpy for mathematical process 

import scipy.stats as stats          # For statistics

from scipy import stats

from scipy.stats import norm, skew  



%matplotlib inline



import pandas_profiling              # Perform various statistical operations on your data

import seaborn as sns                # For data visualization  

import matplotlib.pyplot as plt      # For data visualization 
house_train=pd.read_csv('../input/train.csv') # Read the training File

print(house_train.shape)

house_train.head()
# Drop the "Id" Column as it has no relavency with remaining data. It's just a high cardinal continious number range.



house_train.drop(['Id'],inplace=True,axis=1)

house_train.shape
# Lets seperate our the target variable from train data.

# Sale_price=house_train['SalePrice']



# house_train.drop(['SalePrice'],axis=1,inplace=True)
house_test = pd.read_csv('../input/test.csv')



# Here we store the id's of test data seperately, Because of final submission.

test_id=house_test['Id']



house_test.drop('Id',inplace=True,axis=1)

house_test.head()
house_data=house_train.append(house_test,sort=False)
house_data.head()
# check the final dataframe

house_data.shape
house_data.info()
house_data.describe()
# I commented this out because it takes a lot much time.

# house_data.profile_report(title='Pandas Profiling before Data Preprocessing', style={'full_width':True})
# plotly

import plotly

from plotly.offline import init_notebook_mode, iplot

plotly.offline.init_notebook_mode(connected=True)

init_notebook_mode(connected=True)

import plotly.offline as py

import plotly.graph_objs as go

# Top 25 columsn with their null values.

# house_data.isnull().sum().sort_values(ascending=False)[:25]
# create trace 

trace = go.Bar(

                x = house_data.isnull().sum().sort_values(ascending=False)[:25].index,

                y = house_data.isnull().sum().sort_values(ascending=False)[:25].values,

                name = "Null Value Count",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)))

data = [trace]



layout = go.Layout(

                   title="Null Value Count for each feature",

                   xaxis= dict(title= 'Feature Name',ticklen= 15,zeroline= False),

                   yaxis= dict(title= 'Count',ticklen= 15,zeroline= False))

fig = go.Figure(data = data, layout = layout)

iplot(fig)
house_data['PoolQC'].value_counts()
# Impute the missing values with None.

house_data['PoolQC'].fillna('None',inplace=True)
house_data['MiscFeature'].value_counts()
house_data['MiscFeature'].fillna('None',inplace=True)
house_data['Alley'].value_counts()
house_data['Alley'].fillna('None',inplace=True)
house_data['Fence'].value_counts()
house_train['Fence'].fillna('None',inplace=True)
house_data['FireplaceQu'].value_counts()
house_data['FireplaceQu'].fillna("None",inplace=True)
house_data[['Neighborhood','LotFrontage']]
house_data.groupby('Neighborhood')['LotFrontage'].median()
house_data["LotFrontage"] = house_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    house_data[col] = house_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    house_data[col] = house_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    house_data[col] = house_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    house_data[col] = house_data[col].fillna('None')
house_data["MasVnrType"] = house_data["MasVnrType"].fillna("None")

house_data["MasVnrArea"] =house_data["MasVnrArea"].fillna(0)
house_data['MSZoning'].isnull().sum()
house_data['MSZoning'].value_counts()
house_data['MSZoning'] = house_data['MSZoning'].fillna(house_data['MSZoning'].mode()[0])
house_data['Fence'].fillna('None',inplace=True)
# fill with the attribute which have max count. i.e mode() of the columns



# Functional

house_data['Functional'].fillna(house_data['Functional'].mode()[0],inplace=True)

#Exterior1st

house_data['Exterior1st'].fillna(house_data['Exterior1st'].mode()[0],inplace=True)

# Exterior2nd

house_data['Exterior2nd'].fillna(house_data['Exterior2nd'].mode()[0],inplace=True)

# KitchenQual

house_data['KitchenQual'].fillna(house_data['KitchenQual'].mode()[0],inplace=True)

# SaleType

house_data['SaleType'].fillna(house_data['SaleType'].mode()[0],inplace=True)

# Electrical

house_data['Electrical'].fillna(house_data['Electrical'].mode()[0],inplace=True)
print(house_train['Utilities'].value_counts())

print('------'*15)

print(house_test['Utilities'].value_counts())
house_data['Utilities'].value_counts()
all_data = house_data.drop(['Utilities'], axis=1)
all_data.isnull().sum().sort_values(ascending=False).head(10)
all_data['Fence'].value_counts()
all_data.isnull().sum().sort_values(ascending=False).head()
all_data.head()
all_data.shape
# checking for the ouliers



fig, ax = plt.subplots()

ax.scatter(x = house_train['GrLivArea'], y = house_train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()     
#Deleting outliers

# That's how you drop the ouliers but currently we are keeping them, Will try to see how droping the outliers impacts the result



#--------------------------------------------------------------------------



# house_train = house_train.drop(house_train[(house_train['GrLivArea']>4000) & (house_train['SalePrice']<300000)].index)



# #Check the graphic again

# fig, ax = plt.subplots()

# ax.scatter(house_train['GrLivArea'], house_train['SalePrice'])

# plt.ylabel('SalePrice', fontsize=13)

# plt.xlabel('GrLivArea', fontsize=13)

# plt.show()
# all_data.profile_report(title='Pandas Profiling after Data Preprocessing', style={'full_width':True})
#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn import preprocessing



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = preprocessing.LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
plt.figure(figsize=(10,8))

sns.distplot(house_train['SalePrice'],fit=norm)

# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(house_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Positively skewed.



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
# Q-Q plot

plt.figure()

stats.probplot(house_train['SalePrice'],plot=plt)

plt.show()
target=np.log(house_train['SalePrice'])

print(target.skew())

stats.probplot(target,plot=plt)

plt.show()

#Better than earlier.

Sale_price = np.log1p(house_train['SalePrice'])
len(Sale_price)
all_data.head()
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: {} ".format(len(skewed_feats)))

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness.plot(kind='bar',figsize=(12,7))
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)

    

#all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data = pd.get_dummies(all_data)

print(all_data.shape)
all_data.head()
train=all_data[all_data['SalePrice'].notnull()]
test=all_data[all_data['SalePrice'].isnull()]

test.drop(['SalePrice'],axis=1,inplace=True)
print(train.shape)

train.head(5)
print(test.shape)

test.head(5)
plt.figure(figsize=(10,8))

corr=house_data.corr()

sns.heatmap(corr)

numeric_feature=house_data.select_dtypes(include=np.number)
highly_corr=corr['SalePrice'].sort_values(ascending=False)

new=highly_corr[highly_corr.apply(lambda x:x>0.4)].to_frame().index

plt.figure(figsize=(10,10))

corr_2=numeric_feature[new].corr()

sns.heatmap(corr_2)
house_train["SalePrice"] = np.log1p(house_train["SalePrice"])

y_train=house_train['SalePrice'].values
#shows the positively coorrelated features with SalePrice



#new=pd.DataFrame(data=highly_corr,index=highly_corr.keys())

new=highly_corr.to_frame()

# Or we can use directly .to_frame()

    #sns.set(style='darkgrid')

#plt.subplots(figsize=(8,8))



new.plot(kind='bar',figsize=(10,10))

#The plot shows the correlation between the Sale_price and the other features of the train data.

print(new)
highly_corr[highly_corr.apply(lambda x:x<0)]

# negatively correlated features with the SalePrice.
print(train.shape)

train.head()
print(test.shape)

test.head()
train.head()
Y=train['SalePrice']

X=train.drop('SalePrice',axis=1)
Y.head()
X.head()