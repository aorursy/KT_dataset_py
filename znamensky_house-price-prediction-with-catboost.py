import pandas as pd



%matplotlib inline

import seaborn as sns

print("Setup Complete")



from scipy import stats

from scipy.stats import norm, skew #for some statistics



from sklearn import ensemble



from sklearn.model_selection import KFold,cross_val_score



import matplotlib.pyplot as plt

import numpy as np

import catboost

from catboost import CatBoostClassifier





import warnings

warnings.filterwarnings('ignore')
#read data

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',na_values='nan')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv',na_values='nan')

print("Train size:",df_train.shape)

print("Test size:",df_test.shape)



#Save the 'Id' column

train_ID =df_train['Id']

test_ID =df_test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)

print ("Size of train data after dropping Id: {}" .format(df_train.shape))

print ("Size of test data after dropping Id: {}" .format(df_test.shape))



df_train_num = df_train.select_dtypes(exclude=['object'])

df_train_num.columns
#Deleting outliers-test_Sale

#df_train = df_train.drop(df_train[(df_train['SalePrice']>700000)].index)



#Deleting outliers

train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
from scipy.stats import shapiro

stat, p = shapiro(df_train['SalePrice'])

print('Value for SalePrice column statistics=%.3f, p=%.3f' % (stat, p))
# Check normality for features

for col in df_train_num.columns:

    fig, ax = plt.subplots()

    stat, p = shapiro(train[col])

    ax.scatter(x =train[col], y = train['SalePrice'])

    plt.ylabel('SalePrice', fontsize=13)

    plt.xlabel((p,col), fontsize=13)

    plt.show()
p_df=[]

for col in df_train_num.columns:

    stat, p = shapiro(df_train[col])

    p_df.append(p)

    

#>0.05 means are normally distributed

# Only 3 columns have normal distribution

p_df_=pd.DataFrame(p_df,df_train_num.columns).sort_values(by=0,ascending=True)

p_df_=p_df_.loc[p_df_[0]>0.05]

p_df_
#Let's combine together train/test data for faster data processing



ntrain = train.shape[0]

ntest = df_test.shape[0]

#y_train = train.SalePrice.values

all_data = pd.concat((train,df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
all_data_rep=all_data.copy()

all_data_rep["PoolQC"] = all_data_rep["PoolQC"].fillna(0)

all_data_rep["MiscFeature"] = all_data_rep["MiscFeature"].fillna(0)

all_data_rep["Fence"] = all_data_rep["Fence"].fillna(0)



all_data_rep["FireplaceQu"] = all_data_rep["FireplaceQu"].fillna(0)



all_data_rep["LotFrontage"] = all_data_rep.groupby('Neighborhood')["LotFrontage"].transform(

    lambda x:x.fillna(x.median()))



grd_col=["GarageQual","GarageCond","GarageType","GarageFinish","GarageArea","GarageCars","GarageYrBlt"]

for col in grd_col:

    all_data_rep[col] = all_data_rep[col].fillna(0)

    

grd_col=["BsmtExposure","BsmtCond","BsmtQual","BsmtFinType1","BsmtFinType2","BsmtUnfSF","BsmtFinSF1","BsmtFinSF2"]

for col in grd_col:

    all_data_rep[col] = all_data_rep[col].fillna(0)

    

all_data_rep["TotalBsmtSF"] = all_data_rep["TotalBsmtSF"].fillna(0)





all_data_rep["MasVnrArea"] = all_data_rep["MasVnrArea"].fillna(0)

all_data_rep["MasVnrType"] = all_data_rep["MasVnrType"].fillna(0)



all_data_rep['MSZoning']=all_data_rep['MSZoning'].fillna(all_data_rep['MSZoning'].mode()[0])



all_data_rep["BsmtFullBath"] = all_data_rep["BsmtFullBath"].fillna(0)

all_data_rep["BsmtHalfBath"] = all_data_rep["BsmtHalfBath"].fillna(0)







all_data_rep["Functional"] =all_data_rep["Functional"].fillna("Typ")



all_data_rep["Alley"] = all_data_rep["Alley"].fillna(0)



mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in mode_col:

    all_data_rep[col] = all_data_rep[col].fillna(all_data_rep[col].mode()[0])



    

all_data_rep["MoSold"] = all_data_rep["MoSold"].fillna(0)
#Utilities still has null value, however we will drop this column later (it has only 1 value for majority-no result affect)

all_data_rep.isnull().sum().sort_values(ascending=False)[:3]
all_data_rep['TotalSF'] = all_data_rep['TotalBsmtSF'] + all_data_rep['1stFlrSF'] + all_data_rep['2ndFlrSF']
all_data_rep_enc_1=all_data_rep.copy()


all_data_rep_enc_1['MSSubClass']=all_data_rep_enc_1['MSSubClass'].astype(str)

all_data_rep_enc_1['YrSold']=all_data_rep_enc_1['YrSold'].astype(str)

all_data_rep_enc_1['MoSold']=all_data_rep_enc_1['MoSold'].astype(str)



qual_map = {0: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}



qual_col=['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond','ExterQual',

          'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual']

for col in qual_col:

    all_data_rep_enc_1[col] = all_data_rep_enc_1[col].map(qual_map)

#--------------------------------------------------------------

#furnit_map = {0: 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ':6 }



#furn_col=['BsmtFinType1', 'BsmtFinType2']

#for col in furn_col:

#    all_data_rep_enc_1[col] = all_data_rep_enc_1[col].map(furnit_map)



#functional_map={0: 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2':6,'Min1': 7, 'Typ':8}

#func_col=['Functional']

#for col in func_col:

#    all_data_rep_enc_1[col] = all_data_rep_enc_1[col].map(functional_map)



#fence_map={0: 0, 'MnWw': 1, 'GdWo': 2, 'GdWo': 3, 'MnPrv': 4, 'GdPrv': 5}

#fence_col=['Fence']

#for col in fence_col:

 #   all_data_rep_enc_1[col] = all_data_rep_enc_1[col].map(fence_map)

    

bsm_exp_map={0: 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

all_data_rep_enc_1['BsmtExposure'] = all_data_rep_enc_1['BsmtExposure'].map(bsm_exp_map)



#garag_fin_map={0: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}

#all_data_rep_enc_1['GarageFinish'] = all_data_rep_enc_1['GarageFinish'].map(garag_fin_map)
# Utilities we will drop later because it useless



all_data_rep_enc_1.isnull().sum().sort_values(ascending=False)[:3]
def check_skewness(col):

    sns.distplot(train[col] , fit=norm);

    fig = plt.figure()

    res = stats.probplot(train[col], plot=plt)

    # Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(train[col])

    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    

check_skewness('SalePrice')
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

train["SalePrice"] = np.log1p(train["SalePrice"])



check_skewness('SalePrice')
y_train=train["SalePrice"]
all_data_rep_enc_1.dtypes[all_data_rep_enc_1.dtypes=='object'].index
numeric_feats = all_data_rep_enc_1.dtypes[all_data_rep_enc_1.dtypes!= "object"].index



# Check the skew of all numerical features

skewed_feats = all_data_rep_enc_1[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head()
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.5

for feat in skewed_features:

    #all_data[feat] += 1

    all_data_rep_enc_1[feat] = boxcox1p(all_data_rep_enc_1[feat], lam)
all_data_rep_enc_1=all_data_rep_enc_1.drop(['Utilities'],axis=1)
cat_feat=all_data_rep_enc_1.dtypes[all_data_rep_enc_1.dtypes== "object"].index

cat_feat
X_train = all_data_rep_enc_1[:ntrain]

x_test = all_data_rep_enc_1[ntrain:]

print(X_train.shape)

print(x_test.shape)
# Models Packages

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn import feature_selection

from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
X_train_, X_valid, y_train_, y_valid = train_test_split(

    X_train, y_train, test_size=0.33, random_state=241)


cb_model = CatBoostRegressor(iterations=3500,learning_rate=0.03,od_type='Iter',od_wait=1500,

                             depth=6,random_strength=1,

                            l2_leaf_reg=10,

                             sampling_frequency='PerTree',

                             

                            )

zzz=cb_model.fit(X_train_, y_train_,

             eval_set=(X_valid,y_valid),

             cat_features=cat_feat,use_best_model=True,

             verbose=True)
from sklearn.model_selection import GridSearchCV



model = cb_model

grid_par=[{'iterations':list(range(3138,3139))}]

cv=KFold(n_splits=5,shuffle=True,random_state=241)



gs = GridSearchCV(model,scoring='neg_mean_squared_error',cv=cv,param_grid=grid_par)
gs.fit(X_train, y_train,cat_features=cat_feat,use_best_model=True,verbose=True)
gs.cv_results_
model_out= CatBoostRegressor(iterations=3138,learning_rate=0.03,

                             depth=6,random_strength=1,

                            l2_leaf_reg=10,

                             sampling_frequency='PerTree')

CAT=model_out.fit(X_train, y_train,

             cat_features=cat_feat,

             verbose=True)



test = x_test.copy()

cat_pred=CAT.predict(test)
finalMd = np.expm1(cat_pred)



finalMd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

sample_submission.iloc[:,1] = finalMd

sample_submission.to_csv("sample_submission.csv", index=False)