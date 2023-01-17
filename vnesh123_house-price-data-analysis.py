import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#Read data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
#Create a full set of both train & test data

df_full = df_train.append(df_test, ignore_index=True, sort=False)
df_full.shape
#Drop target from full set

df_full= df_full.drop(['SalePrice'], axis=1)

df_full.shape
#extracted labels for future usage

df_label = df_train['SalePrice']
print(df_train.shape)

print(df_test.shape)
df_train.info()
df_train.columns
df_train.dtypes.value_counts()
df_train.sample(5)
#Check null values in both training & test dataset

total = df_full.isnull().sum().sort_values(ascending=False)

percent = (total/len(df_full.values)) * 100



missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])

missing_data.head(20)
#Get columns with more than 10 percent missing data for drop

missingdata_to_del = [ind for ind in missing_data.index if missing_data.loc[ind,'Percent'] > 10]

print(missingdata_to_del)
#Drop more missing data columns from both training & testing dataset 

df_train = df_train.drop(columns=missingdata_to_del)

df_test = df_test.drop(columns=missingdata_to_del)

print(df_train.shape)

print(df_test.shape)
#Fill missing values with mean - mode values of numerical - categorical columns

df_train_num_cols = df_train.select_dtypes(exclude='object').columns

df_train_obj_cols = df_train.select_dtypes(exclude=np.number).columns



df_train[df_train_num_cols] = df_train[df_train_num_cols].fillna(df_train[df_train_num_cols].mean())

df_train[df_train_obj_cols] = df_train[df_train_obj_cols].fillna(df_train[df_train_obj_cols].mode().iloc[0])
#Check missing values

df_train.isnull().sum().sum()
#Rates the overall condition of the house with SalePrice

data_for_bar = pd.concat([df_train['SalePrice'], df_train['OverallCond']], axis=1)

fig,ax = plt.subplots(figsize=(10,10))

fig = sns.barplot(x='OverallCond', y='SalePrice', data=data_for_bar)

plt.show()
#Original construction date with SalePrice

plt.figure(figsize=(8,8))

sns.scatterplot(x='YearBuilt', y='SalePrice', data=df_train)

plt.show()
#Seperating both numerical & categorical variables

df_train_object = df_train.select_dtypes(exclude=np.number).columns

df_train_numeric = df_train.select_dtypes(exclude='object').columns

print(df_train_object)

print(df_train_numeric)
#Checking unique values in categorical columns

categ_unique = []

categ_unique_col=[]

for col in df_train_object:

    categ_unique_col.append(col)

    categ_unique.append(df_train[col].unique())

df_train_categ = pd.DataFrame({'Unique':list(categ_unique)}, index=categ_unique_col)

df_train_categ.sample(5)
#Seperating numerical columns with their type

train_squarefeet = ['LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea']

train_datetime = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']

train_ordinal = ['OverallQual','OverallCond']

train_continuous = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold']
#Plotting continuous training data as heatmap

var1_continuous = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','OpenPorchSF','SalePrice']

var2_continuous = ['KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','WoodDeckSF','SalePrice']

var3_continuous = ['EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','SalePrice']

var4_continuous = ['LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','SalePrice']

var5_continuous = ['1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','SalePrice']



corr_mat1 = df_train[var1_continuous].corr()

corr_mat2 = df_train[var2_continuous].corr()

corr_mat3 = df_train[var3_continuous].corr()

corr_mat4 = df_train[var4_continuous].corr()

corr_mat5 = df_train[var5_continuous].corr()

fig,ax = plt.subplots(5,1,figsize=(12,18))

plt.rcParams.update({'font.size':12})

sns.heatmap(corr_mat1, vmin=-0.5, vmax=0.8, annot=True, cmap=plt.cm.GnBu, ax=ax[0])

sns.heatmap(corr_mat2, vmin=-0.5, vmax=0.8, annot=True, cmap=plt.cm.GnBu, ax=ax[1])

sns.heatmap(corr_mat3, vmin=-0.5, vmax=0.8, annot=True, cmap=plt.cm.GnBu, ax=ax[2])

sns.heatmap(corr_mat4, vmin=-0.5, vmax=0.8, annot=True, cmap=plt.cm.GnBu, ax=ax[3])

sns.heatmap(corr_mat5, vmin=-0.5, vmax=0.8, annot=True, cmap=plt.cm.GnBu, ax=ax[4])

plt.tight_layout()
fig,ax = plt.subplots(2,2, figsize=(12,10))

sns.countplot(x='RoofStyle', data=df_train, ax=ax[0,0])

sns.countplot(x='Foundation', data=df_train, ax=ax[0,1])

sns.countplot(x='Electrical', data=df_train, ax=ax[1,0])

sns.countplot(x='GarageType', data=df_train, ax=ax[1,1])

plt.tight_layout()
target = df_train.SalePrice

df_train.drop(columns='SalePrice', inplace=True)
#converting categorical values to numeric, selecting low cardinal columns which are < 10

df_train_cardinal = [col for col in df_train.select_dtypes(exclude=np.number).columns if df_train[col].nunique() < 10]

df_train_high_cardinal = [col for col in df_train.select_dtypes(exclude=np.number).columns if df_train[col].nunique() > 10]

train_predictors = df_train[df_train_cardinal]

test_predictors = df_test[df_train_cardinal]

print(df_train_cardinal)

print(df_train_high_cardinal)
#Using one-hot-encoding to convert categorical features to numrical

hot_encoded_train = pd.get_dummies(train_predictors)

hot_encoded_test = pd.get_dummies(test_predictors)
#Algining both train & test dataset for matching columns after encoding

final_train, final_test = hot_encoded_train.align(hot_encoded_test, join='left',axis=1)

print(final_train.shape)

print(final_test.shape)
#Checking correleation coefficient for newly created categorical columns after encoding

label=[]

values=[]



for col in final_train.columns:

    label.append(col)

    values.append(np.corrcoef(final_train[col].values, target.values)[0,1])

    

df_1 = pd.DataFrame({'corr_labels':label, 'corr_values':values})

df_1 = df_1.sort_values(by='corr_values')



ind = np.arange(len(label))

fig,ax = plt.subplots(figsize=(20,50))

rect = ax.barh(ind, df_1.corr_values.values, color='y')

ax.set_yticks(ind)

ax.set_yticklabels(df_1.corr_labels.values)

plt.show()
print(f'Total numerical columns: {len(train_squarefeet+train_datetime+train_ordinal+train_continuous)}')

print('Total numerical columns in training set:', df_train_numeric.shape[0])
#Checking correlation for numerical columns

df_numeric = train_squarefeet + train_continuous



labels=[]

values=[]



for col in df_numeric:

    labels.append(col)

    values.append(np.corrcoef(df_train[col].values, target)[0,1])



corr_df = pd.DataFrame({'corr_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')



ind =  np.arange(len(labels))

fig,ax = plt.subplots(figsize=(10,12))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.corr_labels.values)

ax.set_xlabel('Correleation coefficient')

plt.show()