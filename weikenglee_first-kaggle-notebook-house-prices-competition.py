#Basic Modules

from datetime import datetime

import pandas as pd

import numpy as np



#Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#PreProcessing

from sklearn.impute import SimpleImputer

from sklearn import preprocessing

import category_encoders as ce



#Model Building

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.linear_model import Lasso, Ridge, ElasticNet

from sklearn.ensemble import GradientBoostingRegressor



#Optional Modules

from IPython.display import display

pd.options.display.max_columns = None

import warnings

warnings.filterwarnings("ignore")
Sample = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv",index_col = 'Id')

Test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv",index_col = 'Id')
Sample.shape
Sample.head()
Tgt_Col = 'SalePrice'

Num_Col = Sample.select_dtypes(exclude='object').drop(Tgt_Col,axis=1).columns

Cat_Col = Sample.select_dtypes(include='object').columns



print("Numerical Columns : " , len(Num_Col))

print("Categorical Columns : " , len(Cat_Col))
sns.distplot(Sample[Tgt_Col])

plt.ticklabel_format(style='plain', axis='y')

plt.title("SalePrice's Distribution")

plt.show()



print('Skewness : ' , str(Sample[Tgt_Col].skew()))
sns.distplot(np.log(Sample[Tgt_Col]+1))

plt.ticklabel_format(style='plain', axis='y')

plt.title("SalePrice's Distribution")

plt.show()



print('Skewness : ' , str(np.log(Sample[Tgt_Col]+1).skew()))
Sample[Num_Col].describe().round(decimals=2)
fig = plt.figure(figsize=(12,18))

for idx,col in enumerate(Num_Col):

    fig.add_subplot(9,4,idx+1)

    sns.distplot(Sample[col].dropna(), kde_kws={'bw':0.1})

    plt.xlabel(col)

plt.tight_layout()

plt.show()
cor = Sample.corr()



import matplotlib.style as style

style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))



mask = np.zeros_like(cor, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(cor, cmap=sns.diverging_palette(8, 150, n=10),mask = mask, annot = True,vmin=-1,vmax=1);

plt.title("Heatmap of all the Features", fontsize = 30);
fig = plt.figure(figsize=(12,18))

for idx,col in enumerate(Num_Col):

    fig.add_subplot(9,4,idx+1)

    if abs(cor.iloc[-1,idx])<0.1:

        sns.scatterplot(Sample[col],Sample[Tgt_Col],color='red')

    elif abs(cor.iloc[-1,idx])>=0.5:

        sns.scatterplot(Sample[col],Sample[Tgt_Col],color='green')

    else:

        sns.scatterplot(Sample[col],Sample[Tgt_Col],color='blue')

    plt.title("Corr to SalePrice : " + (np.round(cor.iloc[-1,idx],decimals=2)).astype(str))

plt.tight_layout()

plt.show()
Sample[Num_Col].isna().sum().sort_values(ascending=False).head()
Sample[Cat_Col].describe()
for col in Sample[Cat_Col]:

    if Sample[col].isnull().sum() > 0 :

        print (col , " : ", Sample[col].isnull().sum() , Sample[col].unique())
Sample_copy = Sample.copy()



Sample_copy['MasVnrArea'] = Sample['MasVnrArea'].fillna(0)



Cat_Cols_Fill_NA = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType',

                      'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',

                      'GarageQual', 'GarageFinish', 'GarageType','GarageCond']



for cat in Cat_Cols_Fill_NA:

    Sample_copy[cat] = Sample_copy[cat].fillna("NA")
Sample_copy.isna().sum().sort_values(ascending = False).head()
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['LotFrontage']>200].index)

Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['LotArea']>100000].index)

Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['MasVnrArea']>1200].index)

Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['BsmtFinSF1']>4000].index)

Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['TotalBsmtSF']>4000].index)

Sample_copy = Sample_copy.drop(Sample_copy[(Sample_copy['GrLivArea']>4000) & (Sample_copy[Tgt_Col]<300000)].index)

Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['BsmtFinSF2']>1300].index)

Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['1stFlrSF']>4000].index)

Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['EnclosedPorch']>500].index)

Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['MiscVal']>5000].index)

Sample_copy = Sample_copy.drop(Sample_copy[(Sample_copy['LowQualFinSF']>600) & (Sample_copy[Tgt_Col]>400000)].index)
Sample_copy[Tgt_Col] = np.log(Sample_copy[Tgt_Col]+1)

Sample_copy = Sample_copy.rename(columns={'SalePrice': 'SalePriceLog'})

Tgt_features = 'SalePriceLog'
Sample_copy['MSSubClass'] = Sample_copy['MSSubClass'].astype(str)

Sample_copy['OverallQual'] = Sample_copy['OverallQual'].astype(str)

Sample_copy['OverallCond'] = Sample_copy['OverallCond'].astype(str)
Num_features = Sample_copy.select_dtypes(exclude='object').drop(Tgt_features,axis=1).columns

Cat_features = Sample_copy.select_dtypes(include='object').columns
cor = Sample_copy.corr()

cor_list = cor.abs().unstack()

cor_list[cor_list>0.75].sort_values(ascending=False)[34:].drop_duplicates()
Collinear = ['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF']
Low_Corr = []



for idx,col in enumerate(Num_features):

    if abs(cor.iloc[-1,idx])<=0.1:

        Low_Corr.append(col)
features_drop = ['SalePriceLog'] + Low_Corr + Collinear



X = Sample_copy.drop(features_drop, axis=1)

y = Sample_copy[Tgt_features]
numeric_features = X.select_dtypes(exclude='object').columns

categorical_features = X.select_dtypes(include='object').columns



skewed_feats = X[numeric_features].apply(lambda x: x.skew())

high_skew = skewed_feats[skewed_feats > 0.5]

skew_features = high_skew.index



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
RobustScaler = preprocessing.RobustScaler

PowerTransformer = preprocessing.PowerTransformer



model_list = {'RF_Model' : RandomForestRegressor(random_state=5),

              'XGB_Model' : XGBRegressor(objective ='reg:squarederror',n_estimators=1000, learning_rate=0.05),

              'Lasso_Model' : Lasso(alpha=0.0005), 

              'Ridge_Model' : Ridge(alpha=0.002), 

              'Elastic_Net_Model' : ElasticNet(alpha=0.02, random_state=5, l1_ratio=0.7), 

              'GBR_Model' : GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)}



def inv_y(transformed_y):

    return np.exp(transformed_y)
skew_transformer = Pipeline(steps=[('imputer', SimpleImputer()),

                                   ('scaler', PowerTransformer())])



numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer())])



categorical_transformer = Pipeline(steps=[

    ('imputer1', SimpleImputer(strategy='constant', fill_value='NA')),

    ('encoder', ce.one_hot.OneHotEncoder()),

    ('imputer2', SimpleImputer())])

    

preprocessor = ColumnTransformer(

    transformers=[('skw', skew_transformer, skew_features),

                  ('num', numeric_transformer, numeric_features),

                  ('cat', categorical_transformer, categorical_features)])
model_score = pd.Series()

    

for key in model_list.keys():

    pipe = Pipeline(steps=[('preprocessor', preprocessor),

                           ('scaler', RobustScaler()),

                           ('model', model_list[key])])

    

    model = pipe.fit(train_X, train_y)

    

    pred_y = model.predict(val_X)

    val_mae = mean_absolute_error(inv_y(pred_y), inv_y(val_y))

    model_score[key] = val_mae

    

top_2_model = model_score.nsmallest(n=2)

print(top_2_model)
from sklearn.model_selection import cross_val_score



imputed_X = preprocessor.fit_transform(X,y)

n_folds = 10



for model in top_2_model.index:

    scores = cross_val_score(model_list[model], imputed_X, y, scoring='neg_mean_squared_error', 

                             cv=n_folds)

    mae_scores = np.sqrt(-scores)



    print(model + ':')

    print('Mean RMSE = ' + str(mae_scores.mean().round(decimals=3)))

    print('Error std deviation = ' + str(mae_scores.std().round(decimals=3)) + '\n')
param_grid = [{'alpha': [0.001, 0.0005, 0.0001]}]

top_reg = Lasso()



grid_search = GridSearchCV(top_reg, param_grid, cv=5, 

                           scoring='neg_mean_squared_error')



grid_search.fit(imputed_X, y)



grid_search.best_params_
test_X = Test.copy()



test_X['MasVnrArea'] = test_X['MasVnrArea'].fillna(0)



cat_cols_fill_na = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType',

                      'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',

                      'GarageQual', 'GarageFinish', 'GarageType','GarageCond']



for cat in cat_cols_fill_na:

    test_X[cat] = test_X[cat].fillna("NA")

    

test_X['MSSubClass'] = test_X['MSSubClass'].astype(str)

test_X['OverallQual'] = test_X['OverallQual'].astype(str)

test_X['OverallCond'] = test_X['OverallCond'].astype(str)



if 'SalePriceLog' in features_drop:

    features_drop.remove('SalePriceLog')



test_X = test_X.drop(features_drop, axis=1)
final_model = Lasso(alpha=0.0005, random_state=5)



pipe = Pipeline(steps=[('preprocessor', preprocessor),

                       ('scaler', RobustScaler()),

                       ('model', final_model)])



model = pipe.fit(X, y)



test_preds = model.predict(test_X)



output = pd.DataFrame({'Id': Test.index,

                       'SalePrice': inv_y(test_preds)})



output.to_csv(str(datetime.now().strftime('%Y%m%d_%H%M%S')) + '.csv', index=False)