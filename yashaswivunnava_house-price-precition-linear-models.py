import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler,RobustScaler
import scipy
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_validate
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import GridSearchCV
import warnings
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import matplotlib.style as style
warnings.filterwarnings('ignore')
train_csv = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
test_csv = '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

def target_analysis(target):
    fig = plt.figure(constrained_layout=True, figsize=(14,10))
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Histogram')
    sns.distplot(target,norm_hist=True,ax=ax1)
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.set_title('Q-Q Plot')
    stats.probplot(target,plot=ax2)
    ax3 = fig.add_subplot(grid[:,2])
    ax3.set_title('Box Plot')
    sns.boxplot(target,orient='v',ax=ax3)
    print(f'skweness is { target.skew()}')
    plt.show()
target_analysis(df_train['SalePrice'])
target_analysis(np.log1p(df_train['SalePrice']))
num_cols_tr = df_train.select_dtypes('number').columns.tolist()
num_cols_te = df_test.select_dtypes('number').columns.tolist()

print(f'There are {len(num_cols_tr)} numeric columns in train data')
df_num = df_train[num_cols_tr]
df_test_num = df_test[num_cols_te]

print('Train/test numeric shapes')
print(df_num.shape)
print(df_test_num.shape)
df_num.dtypes
df_num = df_num.drop(columns=['Id'])
df_test_num = df_test_num.drop(columns=['Id'])
corr = df_num.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,cmap=sns.diverging_palette(20,220,n=200))
plt.show()
multicoll_pairs = ['GarageArea','GarageCars',
        'GarageYrBlt','YearBuilt','TotRmsAbvGrd','GrLivArea',
                   '1stFlrSF','TotalBsmtSF']

fig,axes = plt.subplots(4,2,figsize=(15,20))

def plot_two(feat,i,j):
    sns.regplot(x=df_num[feat], y=df_num['SalePrice'], ax=axes[i,j])
    sns.scatterplot(y=df_num['SalePrice'],x=df_num[feat],color=('orange'),ax=axes[i,j])   
    fig.tight_layout(pad=5.0)
    

for i,feat in enumerate(multicoll_pairs):
    j = i%2 #0 or 1
    plot_two(feat,i//2,j)
df_num.corr()['SalePrice'].sort_values(ascending=False)
df_num = df_num.drop(columns = ['1stFlrSF','GarageArea','TotRmsAbvGrd','GarageYrBlt'])
df_test_num = df_test_num.drop(columns = ['1stFlrSF','GarageArea','TotRmsAbvGrd','GarageYrBlt'])
fig,axes = plt.subplots(16,2,figsize=(15,60))

linear_num_cols = df_num.select_dtypes(include='number').columns.tolist() 
linear_num_cols.remove('SalePrice')

def plot_two(feat,i,j):
    sns.regplot(x=df_num[feat], y=df_num['SalePrice'], ax=axes[i,j])
    sns.scatterplot(y=df_num['SalePrice'],x=df_num[feat],color=('orange'),ax=axes[i,j])   
    fig.tight_layout(pad=5.0)
    

for i,feat in enumerate(linear_num_cols):
    j = i%2 #0 or 1
    plot_two(feat,i//2,j)
df_num = df_num.drop(columns=['OverallCond','LowQualFinSF', 'MiscVal'])
df_test_num = df_test_num.drop(columns=['OverallCond','LowQualFinSF', 'MiscVal'])
print('Train/test numeric shapes')
print(df_num.shape)
print(df_test_num.shape)
df_num = df_num[df_num['LotFrontage'] < 300]
df_num = df_num[df_num['BsmtFinSF1'] < 5000]
df_num = df_num[df_num['TotalBsmtSF'] < 6000]
df_num = df_num[df_num['GrLivArea'] < 4600]
df_num = df_num[df_num['SalePrice'] < 700000]
print(df_num.shape)
non_linear_cat_cols = ['YrSold','MoSold','PoolArea','BsmtFullBath',
            'BsmtHalfBath','HalfBath','BedroomAbvGr','Fireplaces']

df_num = df_num.drop(columns = non_linear_cat_cols)
df_test_num = df_test_num.drop(columns = non_linear_cat_cols)

for col in non_linear_cat_cols:
    df_train[col] = df_train[col].astype(object)
    df_test[col] = df_train[col].astype(object)
    
print(df_num.shape)
print(df_test_num.shape)
df_num.head(3)
# missing values in numeric features
def missing_cols(df):
    cols = df.columns[df.isna().any()].tolist()
    print(f'Columns | Percentage missing')
    for column in cols:
        percent = round((sum(df[column].isnull())/df.shape[0])*100,2)
        print(f'{column} : {percent}%')
missing_cols(df_num)
missing_cols(df_test_num)
# lets convert the target variable and keep
target = np.log1p(df_num['SalePrice'])
# helper functions
def categories_plot(df,col,xlabel='Values',size=(8,4)):
    y_train = df[col].value_counts().values
    x_train = df[col].value_counts().index.tolist()
    plt.figure(figsize=size)
    plt.title(col)
    sns.barplot(x_train,y_train)
    plt.xlabel(xlabel)
    plt.xticks(rotation=90, ha='right')
    plt.ylabel('count')
    plt.show()
num_idx = df_num.index.to_list()
cat_cols = df_train.select_dtypes(exclude=[np.number]).columns.tolist()

df_cat = df_train.loc[num_idx][cat_cols]
df_cat_test = df_test[cat_cols]
print('Train/test categoric shapes')
print(df_cat.shape)
print(df_cat_test.shape)
missing_cols(df_cat)
missing_cols(df_cat_test)
df_cat = df_cat.drop(columns=['Alley','MiscFeature','PoolQC','Fence'])
df_cat_test = df_cat_test.drop(columns=['Alley','MiscFeature','PoolQC','Fence'])
df_cat.describe()
categories_plot(df_cat,'MSZoning')
modified_cols = ['PoolArea','Street','MasVnrType','RoofMatl','Utilities']

for col in modified_cols:
    categories_plot(df_cat,col)
# some feature engineering on cat features

modified_cols = ['PoolArea','Street','MasVnrType','RoofMatl']


df_cat = df_cat.drop(columns = ['Utilities'])
df_cat_test = df_cat_test.drop(columns = ['Utilities'])

df_cat['PoolArea'] = df_cat['PoolArea'].apply(lambda x: 'Y' if x>1 else 'N') 
df_cat_test['PoolArea'] = df_cat_test['PoolArea'].apply(lambda x: 'Y' if x>1 else 'N')

df_cat['Street'] = df_cat['Street'].apply(lambda x: 'Pave' if x == 'Pave' else 'No Pave')
df_cat_test['Street'] = df_cat_test['Street'].apply(lambda x: 1 if x == 'Pave' else 0)

df_cat['MasVnrType'] = df_cat['MasVnrType'].apply(lambda x: 'N' if x == 'None' else 'Y')
df_cat_test['MasVnrType'] = df_cat_test['MasVnrType'].apply(lambda x: 'N' if x == 'None' else 'Y')

df_cat['RoofMatl'] = df_cat['RoofMatl'].apply(lambda x: 'CompShg' if x == 'CompShg' else 'Other')
df_cat_test['RoofMatl'] = df_cat_test['RoofMatl'].apply(lambda x: 'CompShg' if x == 'CompShg' else 'Other')
# saving coumn names
cat_cols = df_cat.columns.to_list()
num_cols = df_num.columns.to_list()

df_test_num = df_test_num.reset_index(drop=True)
df_num = df_num.reset_index(drop=True)
df_cat = df_cat.reset_index(drop=True)
df_test_cat = df_cat_test.reset_index(drop=True)

final_train = pd.concat([df_num,df_cat],axis=1)
final_test = pd.concat([df_test_num,df_test_cat],axis=1)

print('Final shapes:')
print(final_train.shape)
print(final_test.shape)
# apply box cox transform to features having skweness > 0.5
def sqrt_skew(df):
    
    sk_feats = df.apply(lambda x: stats.skew(x)).sort_values(ascending=False)
    high_skew = sk_feats[abs(sk_feats) > 0.5].index
    for feat in high_skew:
#         df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))
          df[feat] = np.sqrt(df[feat])
        
    return df
Y = target.values
X = final_train
x_test = final_test.copy()
x_train,x_cv,y_train,y_cv = train_test_split(X,Y,train_size=0.7,random_state=100)
print('Train cv data shape')
print(x_train.shape)
print(x_cv.shape)
print(x_test.shape)
# imputing numeric values
# Featurization of numeric data
# df_num = df_num.drop(columns=['SalePrice'])

num_cols_in = df_num.columns.to_list()
num_cols_in.remove('SalePrice')

imputer = SimpleImputer(strategy='median')
x_train_num = imputer.fit_transform(x_train[num_cols_in])
x_cv_num = imputer.transform(x_cv[num_cols_in])
x_test_num = imputer.transform(x_test[num_cols_in])
#Normalizing
scaler = RobustScaler()
x_train_num = scaler.fit_transform(x_train_num)
x_cv_num = scaler.transform(x_cv_num)
x_test_num = scaler.transform(x_test_num)

df_num = pd.DataFrame(x_train_num, columns=num_cols_in)
df_cv_num = pd.DataFrame(x_cv_num, columns=num_cols_in)
df_test_num = pd.DataFrame(x_test_num, columns=num_cols_in)
# cat_cols = df_cat.columns.to_list()

# # missing values in df
# imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
# df_cat = imputer.fit_transform(df_cat[cat_cols])
# df_cat_test = imputer.transform(df_cat_test[cat_cols])

# df_cat = pd.DataFrame(df_cat, columns=cat_cols)
# df_cat_test = pd.DataFrame(df_cat_test, columns=cat_cols)

x_train_cat = x_train[cat_cols]
x_cv_cat = x_cv[cat_cols]
x_test_cat = x_test[cat_cols]

for col in cat_cols:
    val = x_train_cat[col].mode()[0]
    x_train_cat[col] = x_train_cat[col].fillna(val)
    x_cv_cat[col] = x_cv_cat[col].fillna(val)
    x_test_cat[col] = x_test_cat[col].fillna(val)
df_cat_dummy = pd.get_dummies(x_train_cat, columns=cat_cols,drop_first=True)
df_cv_cat_dummy = pd.get_dummies(x_cv_cat, columns=cat_cols,drop_first=True)
df_test_cat_dummy = pd.get_dummies(x_test_cat, columns=cat_cols,drop_first=True)
print(df_cat_dummy.shape)
print(df_cv_cat_dummy.shape)
print(df_test_cat_dummy.shape)

df_cat, df_cat_cv = df_cat_dummy.align(df_cv_cat_dummy, join='left', axis=1) 
df_cat, df_cat_test = df_cat_dummy.align(df_test_cat_dummy, join='left', axis=1) 
df_cat_test = df_cat_test.fillna(0)
df_cat_cv = df_cat_cv.fillna(0)

print('dummy categorical data shapes after aligning with train data')
print(df_cat.shape)
print(df_cat_cv.shape)
print(df_cat_test.shape)
# reseting index
df_cat_dummy = df_cat.reset_index(drop=True)
df_cv_cat_dummy = df_cat_cv.reset_index(drop=True)
df_test_cat_dummy = df_cat_test.reset_index(drop=True)
df_num = df_num.reset_index(drop=True)
df_cv_num = df_cv_num.reset_index(drop=True)
df_test_num = df_test_num.reset_index(drop=True)

final_train = pd.concat([df_num,df_cat_dummy],axis=1)
final_cv = pd.concat([df_cv_num,df_cv_cat_dummy],axis=1)
final_test = pd.concat([df_test_num,df_test_cat_dummy],axis=1)

print('Final shapes:')
print(final_train.shape)
print(final_cv.shape)
print(final_test.shape)
x_train = final_train.copy()
x_cv = final_cv.copy()
x_test = final_test.copy()
linear = LinearRegression()
linear.fit(x_train,y_train)

y_pred_train = linear.predict(x_train)
y_pred_cv = linear.predict(x_cv)


print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(y_cv, y_pred_cv)))) 
print('R2 score = ' + str(r2_score(y_cv,y_pred_cv)))