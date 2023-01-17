'''Importing Data Manipulattion Moduls'''

import numpy as np

import pandas as pd



'''Seaborn and Matplotlib Visualization'''

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('bmh')                    

sns.set_style({'axes.grid':False}) 

%matplotlib inline



'''plotly Visualization'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True)



'''Ignore deprecation and future, and user warnings.'''

import warnings as wrn

wrn.filterwarnings('ignore', category = DeprecationWarning) 

wrn.filterwarnings('ignore', category = FutureWarning) 

wrn.filterwarnings('ignore', category = UserWarning) 
'''Read in train and test data from csv files'''

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
'''Train and test data at a glance'''

train.head()
test.head()
'''Dimensions of train and test data'''

print('Dimensions of train data:', train.shape)

print('Dimensions of test data:', test.shape)
"""Let's check the columns names"""

train.columns.values
"""Let's merge the train and test data and inspect the data type"""

merged = pd.concat([train, test], axis=0, sort=True)

display(merged.dtypes.value_counts())

print('Dimensions of data:', merged.shape)
'''Extracting numerical variables first'''

num_merged = merged.select_dtypes(include = ['int64', 'float64'])

display(num_merged.head(3))

print('\n')

display(num_merged.columns.values)
'''Plot histogram of numerical variables to validate pandas intuition.'''

def draw_histograms(df, variables, n_rows, n_cols):

    fig=plt.figure()

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        df[var_name].hist(bins=40,ax=ax,color = 'green',alpha=0.5, figsize = (40, 200))

        ax.set_title(var_name, fontsize = 43)

        ax.tick_params(axis = 'both', which = 'major', labelsize = 35)

        ax.tick_params(axis = 'both', which = 'minor', labelsize = 35)

        ax.set_xlabel('')

    fig.tight_layout(rect = [0, 0.03, 1, 0.95])  # Improves appearance a bit.

    plt.show()

    

draw_histograms(num_merged, num_merged.columns, 19, 2)
'''Convert MSSubClass, OverallQual, OverallCond, MoSold, YrSold into categorical variables.'''

merged.loc[:,['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = merged.loc[:,['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')
'''Check out the data type after correction'''

merged.dtypes.value_counts()
'''Function to plot scatter plot'''

def scatter_plot(x, y, title, xaxis, yaxis, size, c_scale):

    trace = go.Scatter(x = x,

                        y = y,

                        mode = 'markers',

                        marker = dict(color = y, size=size, showscale = True, colorscale = c_scale))

    layout = go.Layout(hovermode = 'closest', title = title, xaxis = dict(title = xaxis), yaxis = dict(title = yaxis))

    fig = go.Figure(data = [trace], layout = layout)

    return iplot(fig)



'''Function to plot bar chart'''

def bar_plot(x, y, title, yaxis, c_scale):

    trace = go.Bar(x = x,

                   y = y,

                   marker = dict(color = y, colorscale = c_scale))

    layout = go.Layout(hovermode= 'closest', title = title, yaxis = dict(title = yaxis))

    fig = go.Figure(data = [trace], layout = layout)

    return iplot(fig)



'''Function to plot histogram'''

def histogram_plot(x, title, yaxis, color):

    trace = go.Histogram(x = x,

                        marker = dict(color = color))

    layout = go.Layout(hovermode = 'closest', title = title, yaxis = dict(title = yaxis))

    fig = go.Figure(data = [trace], layout = layout)

    return iplot(fig)
corr = train.corr()

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(corr, linewidths=.5, vmin=0, vmax=1, square=True)
k = 10 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
'''Sactter plot of GrLivArea vs SalePrice.'''

scatter_plot(train.GrLivArea, train.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
'''Drop observations where GrLivArea is greater than 4000 sq.ft'''

train.drop(train[train.GrLivArea>4000].index, inplace = True)

train.reset_index(drop = True, inplace = True)
'''Sactter plot of GrLivArea vs SalePrice.'''

scatter_plot(train.GrLivArea, train.SalePrice, 'GrLivArea vs SalePrice', 'GrLivArea', 'SalePrice', 10, 'Rainbow')
'''Scatter plot of TotalBsmtSF Vs SalePrice'''

scatter_plot(train.TotalBsmtSF, train.SalePrice, 'TotalBsmtSF Vs SalePrice', 'TotalBsmtSF', 'SalePrice', 10, 'Cividis')
'''Drop observations where TotlaBsmtSF is greater than 3000 sq.ft'''

train.drop(train[train.TotalBsmtSF>3000].index, inplace = True)

train.reset_index(drop = True, inplace = True)
'''Scatter plot of TotalBsmtSF Vs SalePrice'''

scatter_plot(train.TotalBsmtSF, train.SalePrice, 'TotalBsmtSF Vs SalePrice', 'TotalBsmtSF', 'SalePrice', 10, 'Cividis')
'''Scatter plot of YearBuilt Vs SalePrice'''

scatter_plot(train.YearBuilt, np.log1p(train.SalePrice), 'YearBuilt Vs SalePrice', 'YearBuilt', 'SalePrice', 10, 'viridis')
'''Drop observations where YearBulit is less than 1893 sq.ft'''

train.drop(train[train.YearBuilt<1900].index, inplace = True)

train.reset_index(drop = True, inplace = True)
'''Scatter plot of YearBuilt Vs SalePrice'''

scatter_plot(train.YearBuilt, np.log1p(train.SalePrice), 'YearBuilt Vs SalePrice', 'YearBuilt', 'SalePrice', 10, 'viridis')
'''Scatter plot of GarageCars Vs SalePrice'''

scatter_plot(train.GarageCars, np.log(train.SalePrice), 'GarageCars Vs SalePrice', 'GarageCars', 'SalePrice', 10, 'Electric')
'''Scatter plot of GarageCars Vs SalePrice'''

scatter_plot(train.OverallQual, np.log(train.SalePrice), 'OverallQual Vs SalePrice', 'OverallQual', 'SalePrice', 10, 'Bluered')
'''Scatter plot of FullBath Vs SalePrice'''

scatter_plot(train.FullBath, np.log(train.SalePrice), 'FullBath Vs SalePrice', 'FullBath', 'SalePrice', 10, 'RdBu')
'''separate our target variable first'''

y_train = train.SalePrice



'''Drop SalePrice from train data.'''

train.drop('SalePrice', axis = 1, inplace = True)



'''Now combine train and test data frame together.'''

df_merged = pd.concat([train, test], axis = 0)



'''Dimensions of new data frame'''

df_merged.shape
'''Again convert MSSubClass, OverallQual, OverallCond, MoSold, YrSold into categorical variables.'''

df_merged.loc[:,['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']] = df_merged.loc[:,['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']].astype('object')

df_merged.dtypes.value_counts()
'''columns with missing observation'''

missing_columns = df_merged.columns[df_merged.isnull().any()].values

'''Number of columns with missing obervation'''

total_missing_columns = np.count_nonzero(df_merged.isnull().sum())

print('We have ' , total_missing_columns ,  'features with missing values and those features (with missing values) are: \n\n' , missing_columns)
'''Simple visualization of missing variables'''

plt.figure(figsize=(20,8))

sns.heatmap(df_merged.isnull(), yticklabels=False, cbar=False, cmap = 'summer')
'''Get and plot only the features (with missing values) and their corresponding missing values.'''

missing_columns = len(df_merged) - df_merged.loc[:, np.sum(df_merged.isnull())>0].count()

x = missing_columns.index

y = missing_columns

title = 'Variables with Missing Values'

scatter_plot(x, y, title, 'Features Having Missing Observations','Missing Values', 20, 'Viridis')
missing_columns
'''Impute by None where NaN means something.'''

to_impute_by_none = df_merged.loc[:, ['PoolQC','MiscFeature','Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageCond','GarageFinish','GarageQual','BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1','MasVnrType']]

for i in to_impute_by_none.columns:

    df_merged[i].fillna('None', inplace = True)
'''These are categorical variables and will be imputed by mode.'''

to_impute_by_mode =  df_merged.loc[:, ['Electrical', 'MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional', 'SaleType']]

for i in to_impute_by_mode.columns:

    df_merged[i].fillna(df_merged[i].mode()[0], inplace = True)
'''The following variables are either discrete numerical or continuous numerical variables.So the will be imputed by median.'''

to_impute_by_median = df_merged.loc[:, ['BsmtFullBath','BsmtHalfBath', 'GarageCars', 'MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageArea']]

for i in to_impute_by_median.columns:

    df_merged[i].fillna(df_merged[i].median(), inplace = True)
'''We need to convert categorical variable into numerical to plot correlation heatmap. So convert categorical variables into numerical.'''

df = df_merged.drop(columns=['Id','LotFrontage'], axis=1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df = df.apply(le.fit_transform) # data is converted.

df.head(2)
 # Inserting Age in variable correlation.

df['LotFrontage'] = df_merged['LotFrontage']

# Move Age at index 0.

df = df.set_index('LotFrontage').reset_index()

df.head(2)
'''correlation of df'''

corr = df.corr()

display(corr['LotFrontage'].sort_values(ascending = False)[:5])

display(corr['LotFrontage'].sort_values(ascending = False)[-5:])
'''Impute LotFrontage with median of respective columns (i.e., BldgType)'''

df_merged['LotFrontage'] = df_merged.groupby(['BldgType'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
'''Is there any missing values left untreated??'''

print('Missing variables left untreated: ', df_merged.columns[df_merged.isna().any()].values)
'''Skewness and Kurtosis of SalePrice'''

print("Skewness: %f" % y_train.skew())

print("Kurtosis: %f" % y_train.kurt())
'''Plot the distribution of SalePrice with skewness.'''

histogram_plot(y_train, 'SalePrice without Transformation', 'Abs Frequency', 'deepskyblue')
'''Plot the distribution of SalePrice with skewness'''

y_train = np.log1p(y_train)

title = 'SalePrice after Transformation (skewness: {:0.4f})'.format(y_train.skew())

histogram_plot(y_train, title, 'Abs Frequency', ' darksalmon')
'''Now calculate the rest of the explanetory variables'''

skew_num = pd.DataFrame(data = df_merged.select_dtypes(include = ['int64', 'float64']).skew(), columns=['Skewness'])

skew_num_sorted = skew_num.sort_values(ascending = False, by = 'Skewness')

skew_num_sorted
''' plot the skewness for rest of the explanetory variables'''

bar_plot(skew_num_sorted.index, skew_num_sorted.Skewness, 'Skewness in Explanetory Variables', 'Skewness', 'Blackbody')
'''Extract numeric variables merged data.'''

df_merged_num = df_merged.select_dtypes(include = ['int64', 'float64'])
'''Make the tranformation of the explanetory variables'''

df_merged_skewed = np.log1p(df_merged_num[df_merged_num.skew()[df_merged_num.skew() > 0.5].index])





#Normal variables

df_merged_normal = df_merged_num[df_merged_num.skew()[df_merged_num.skew() < 0.5].index]

    

#Merging

df_merged_num_all = pd.concat([df_merged_skewed, df_merged_normal], axis = 1)
'''Update numerical variables with transformed variables.'''

df_merged_num.update(df_merged_num_all)
'''Standarize numeric features with RobustScaler'''

from sklearn.preprocessing import RobustScaler



'''Creating scaler object.'''

scaler = RobustScaler()



'''Fit scaler object on train data.'''

scaler.fit(df_merged_num)



'''Apply scaler object to both train and test data.'''

df_merged_num_scaled = scaler.transform(df_merged_num)
'''Retrive column names'''

df_merged_num_scaled = pd.DataFrame(data = df_merged_num_scaled, columns = df_merged_num.columns, index = df_merged_num.index)

# Pass the index of index df_merged_num, otherwise it will sum up the index.

"""Let's extract categorical variables first and convert them into category."""

df_merged_cat = df_merged.select_dtypes(include = ['object']).astype('category')



"""let's begin the tedious process of label encoding of ordinal variable"""

df_merged_cat.LotShape.replace(to_replace = ['IR3', 'IR2', 'IR1', 'Reg'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.LandContour.replace(to_replace = ['Low', 'Bnk', 'HLS', 'Lvl'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.Utilities.replace(to_replace = ['NoSeWa', 'AllPub'], value = [0, 1], inplace = True)

df_merged_cat.LandSlope.replace(to_replace = ['Sev', 'Mod', 'Gtl'], value = [0, 1, 2], inplace = True)

df_merged_cat.ExterQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.ExterCond.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.BsmtQual.replace(to_replace = ['None', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.BsmtCond.replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.BsmtExposure.replace(to_replace = ['None', 'No', 'Mn', 'Av', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.BsmtFinType1.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)

df_merged_cat.BsmtFinType2.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)

df_merged_cat.HeatingQC.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.Electrical.replace(to_replace = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], value = [0, 1, 2, 3, 4], inplace = True)

df_merged_cat.KitchenQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.Functional.replace(to_replace = ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)

df_merged_cat.FireplaceQu.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)

df_merged_cat.GarageFinish.replace(to_replace =  ['None', 'Unf', 'RFn', 'Fin'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.GarageQual.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)

df_merged_cat.GarageCond.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)

df_merged_cat.PavedDrive.replace(to_replace =  ['N', 'P', 'Y'], value = [0, 1, 2], inplace = True)

df_merged_cat.PoolQC.replace(to_replace =  ['None', 'Fa', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)

df_merged_cat.Fence.replace(to_replace =  ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], value = [0, 1, 2, 3, 4], inplace = True)
'''All the encodeded variables have int64 dtype except OverallQual and OverallCond. So convert them back into int64.'''

df_merged_cat.loc[:, ['OverallQual', 'OverallCond']] = df_merged_cat.loc[:, ['OverallQual', 'OverallCond']].astype('int64')



'''Extract label encoded variables'''

df_merged_label_encoded = df_merged_cat.select_dtypes(include = ['int64'])
'''Now selecting the nominal vaiables for one hot encording'''

df_merged_one_hot = df_merged_cat.select_dtypes(include=['category'])



"""Let's get the dummies variable"""

df_merged_one_hot = pd.get_dummies(df_merged_one_hot, drop_first=True)
"""Let's concat one hot encoded and label encoded variable together"""

df_merged_encoded = pd.concat([df_merged_one_hot, df_merged_label_encoded], axis=1)



'''Finally join processed categorical and numerical variables'''

df_merged_processed = pd.concat([df_merged_num_scaled, df_merged_encoded], axis=1)



'''Dimensions of new data frame'''

df_merged_processed.shape
'''Now retrive train and test data for modelling.'''

df_train_final = df_merged_processed.iloc[0:1438, :]

df_test_final = df_merged_processed.iloc[1438:, :]



'''And we have our target variable as y_train.'''

y_train = y_train
'''Updated train data'''

df_train_final.head()
'''Updated test data'''

df_train_final.head()
"""Let's have a final look at data dimension"""

print('Input matrix dimension:', df_train_final.shape)

print('Output vector dimension:', y_train.shape)

print('Test data dimension:', df_test_final.shape)
'''set a seed for reproducibility'''

seed = 44



'''Initialize all the regesssion models object we are interested in'''

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.cross_decomposition import PLSRegression

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor



''''We are interested in the following 14 regression models.

All initialized with default parameters except random_state and n_jobs.'''

lr = LinearRegression(n_jobs = -1)

lasso = Lasso(random_state = seed)

ridge = Ridge(random_state = seed)

elnt = ElasticNet(random_state = seed)

kr = KernelRidge()

dt = DecisionTreeRegressor(random_state = seed)

svr = SVR()

knn = KNeighborsRegressor(n_jobs= -1)

pls = PLSRegression()

rf = RandomForestRegressor(n_jobs = -1, random_state = seed)

et = ExtraTreesRegressor(n_jobs = -1, random_state = seed)

ab = AdaBoostRegressor(random_state = seed)

gb = GradientBoostingRegressor(random_state = seed)

xgb = XGBRegressor(n_jobs = -1, random_state = seed)

lgb = LGBMRegressor(n_jobs = -1, random_state = seed)
'''Training accuracy of our regression models. By default score method returns coefficient of determination (r_squared).'''

def train_r2(model):

    model.fit(df_train_final, y_train)

    return model.score(df_train_final, y_train)



'''Calculate and plot the training accuracy.'''

models = [lr, lasso, ridge, elnt, kr, dt, svr, knn, pls, rf, et, ab, gb, xgb, lgb]

training_score = []

for model in models:

    training_score.append(train_r2(model))



'''Plot dataframe of training accuracy.'''

train_score = pd.DataFrame(data = training_score, columns = ['Training_R2'])

train_score.index = ['LR', 'LASSO',  'RIDGE', 'ELNT', 'KR', 'DT', 'SVR', 'KNN', 'PLS', 'RF', 'ET', 'AB', 'GB', 'XGB', 'LGB']

train_score = (train_score*100).round(4)

scatter_plot(train_score.index, train_score['Training_R2'], 'Training Score (R_Squared)', 'Models', '% Training Score', 30, 'Rainbow')
'''Evaluate model on the hold set'''

def train_test_split(model):

    from sklearn.model_selection import train_test_split

    from sklearn.metrics import mean_squared_error

    X_train, X_test, Y_train, Y_test = train_test_split(df_train_final, y_train, test_size = 0.3, random_state = seed)

    model.fit(X_train, Y_train)

    prediction = model.predict(X_test)

    mse = mean_squared_error(prediction, Y_test)

    rmse = np.sqrt(mse) #non-negative square-root

    return rmse



'''Calculate train_test_split score of differnt models and plot them.'''

models = [lasso, ridge, elnt, kr, dt, svr, knn, pls, rf, et, ab, gb, xgb, lgb]

train_test_split_rmse = []

for model in models:

    train_test_split_rmse.append(train_test_split(model))

    

'''Plot data frame of train test rmse'''

train_test_score = pd.DataFrame(data = train_test_split_rmse, columns = ['Train_Test_RMSE'])

train_test_score.index = ['LASSO',  'RIDGE', 'ELNT', 'KR', 'DT', 'SVR', 'KNN', 'PLS', 'RF', 'ET', 'AB', 'GB', 'XGB', 'LGB']

scatter_plot(train_test_score.index, train_test_score['Train_Test_RMSE'], "Models' Test Score (RMSE) on Holdout(30%) Set", 'Models', 'RMSE', 30, 'plotly3')
'''Function to compute cross validation scores.'''

def cross_validation(model):

    from sklearn.model_selection import cross_val_score

    val_score = cross_val_score(model, df_train_final, y_train, cv=10, n_jobs= -1, scoring = 'neg_mean_squared_error')

    sq_val_score = np.sqrt(-1*val_score)

    r_val_score = np.round(sq_val_score, 5)

    return r_val_score.mean()



'''Calculate cross validation score of differnt models and plot them.'''

models = [lasso, ridge, elnt, kr, dt, svr, knn, pls, rf, et, ab, gb, xgb, lgb]

cross_val_scores = []

for model in models:

    cross_val_scores.append(cross_validation(model))



'''Plot data frame of cross validation scores.'''

x_val_score = pd.DataFrame(data = cross_val_scores, columns=['Cross_Val_Score(RMSE)'])

x_val_score.index = ['LASSO',  'RIDGE', 'ELNT', 'KR', 'DT', 'SVR', 'KNN', 'PLS', 'RF', 'ET', 'AB', 'GB', 'XGB', 'LGB']

scatter_plot(x_val_score.index, x_val_score['Cross_Val_Score(RMSE)'], "Models' 10-fold Cross Validation Scores (RMSE)", 'Models', 'RMSE', 30, 'cividis')
'''Create a function to tune hyperparameters of the selected models.'''

def tune_hyperparameters(model, param_grid):

    from sklearn.model_selection import GridSearchCV

    global best_params, best_score #if you want to know best parametes and best score

    

    '''Construct grid search object with 10 fold cross validation.'''

    grid = GridSearchCV(model, param_grid, cv = 10, verbose = 1, scoring = 'neg_mean_squared_error', n_jobs = -1)

    grid.fit(df_train_final, y_train)

    best_params = grid.best_params_ 

    best_score = np.round(np.sqrt(-1 * grid.best_score_), 5)

    return best_params, best_score
'''Difine hyperparameters of ridge'''

ridge_param_grid = {'alpha':[0.5, 2.5, 3.3, 5, 5.5, 7, 9, 9.5, 9.52, 9.64, 9.7, 9.8, 9.9, 10, 10.5,10.62,10.85, 20, 30],

                    'random_state':[seed]}

tune_hyperparameters(ridge, ridge_param_grid)

ridge_best_params, ridge_best_score = best_params, best_score

print('Ridge best params:{} & best_score:{:0.5f}' .format(ridge_best_params, ridge_best_score))
alpha = [0.0001, 0.0002, 0.00025, 0.0003, 0.00031, 0.00032, 0.00033, 0.00034, 0.00035, 0.00036, 0.00037, 0.00038, 

         0.0004, 0.00045, 0.0005, 0.00055, 0.0006, 0.0008,  0.001, 0.002, 0.005, 0.007, 0.008, 0.01]



lasso_params = {'alpha': alpha,

               'random_state':[seed]}



tune_hyperparameters(lasso, lasso_params)

lasso_best_params, lasso_best_score = best_params, best_score

print('Lasso best params:{} & best_score:{:0.5f}' .format(lasso_best_params, lasso_best_score))
KernelRidge_param_grid = {'alpha':[0.1, 0.15, 0.23, 0.25, 0.3,1],

                          'kernel': ['linear', 'polynomial'],

                          'degree': [2,3],

                          'coef0': [1.5,2,3]}

tune_hyperparameters(kr, KernelRidge_param_grid)

kr_best_params, kr_best_score = best_params, best_score

print('Kernel Ridge best params:{} & best_score: {:0.5f}'. format(kr_best_params, kr_best_score))
elastic_params_grid = {'alpha': [0.0001,0.0002, 0.0003, 0.01,0.1,2], 

                 'l1_ratio': [0.2, 0.85, 0.95,0.98,10],

                 'random_state':[seed]}

tune_hyperparameters(elnt, elastic_params_grid)

elastic_best_params, elastic_best_score = best_params, best_score

print('Elastic Net best params:{} & best_score:{:0.5f}' .format(elastic_best_params, elastic_best_score))
svr_params_grid = {'kernel':['linear', 'poly', 'rbf'],

                   'C':[2,4,5],

                   'gamma':[0.01,0.001,0.0001]}

tune_hyperparameters(svr, svr_params_grid)

svr_best_params, svr_best_score = best_params, best_score

print('SVR best params:{} & best_score:{:0.5f}' .format(svr_best_params, svr_best_score))
rf_params_grid = {'n_estimators':[1,5,50,100],

                   'max_depth':[1,2],

                   'min_samples_split':[3,4],

                   'min_samples_leaf':[2,4],

                   'random_state':[seed]}

tune_hyperparameters(rf, rf_params_grid)

rf_best_params, rf_best_score = best_params, best_score

print('RF best params:{} & best_score:{:0.5f}' .format(rf_best_params, rf_best_score))
xgb_params_grid = {'min_child_weight': [5, 10],

                   'gamma': [0.04, 0.1, 1.5],

                   'subsample': [0.6, 0.8, 1.0],

                   'colsample_bytree': [0.46, 1.0],

                   'max_depth': [3, 4]}

xgb_opt = XGBRegressor(learning_rate = 0.03, reg_alpha = 0.4640, reg_lambda = 0.8571, n_estimators = 1000, 

                       silent = 1, nthread = -1, random_state = 101)



tune_hyperparameters(xgb_opt, xgb_params_grid)

xgb_best_params, xgb_best_score = best_params, best_score

print('XGB best params:{} & best_score:{:0.5f}' .format(xgb_best_params, xgb_best_score))
'''Not Optimize Randomly choosen parameters'''

'''Hyperparameters of gb'''

gb_opt = GradientBoostingRegressor(n_estimators = 3000, learning_rate = 0.05,

                                   max_depth = 4, max_features = 'sqrt',

                                   min_samples_leaf = 15, min_samples_split = 10, 

                                   loss = 'huber', random_state = seed)

'''Hyperparameters of lgb'''

lgb_opt = LGBMRegressor(objective = 'regression', num_leaves = 5,

                              learning_rate=0.05, n_estimators = 660,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed = 9, bagging_seed = 9,

                              min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)

'''We can assume these 2 model best score is equal to cross validation scores.

Thought it might not be precise, but I will take it'''

gb_best_score = cross_validation(gb_opt)

lgb_best_score = cross_validation(lgb_opt)
"""Let's plot the models' rmse after optimization."""

optimized_scores = pd.DataFrame({'Optimized Scores':np.round([lasso_best_score, ridge_best_score, kr_best_score, 

                  elastic_best_score, svr_best_score, rf_best_score, xgb_best_score, gb_best_score, lgb_best_score], 5)})

optimized_scores.index = ['Lasso', 'Ridge', 'Kernel_ridge', 'E_net', 'SVM', 'RF', 'XGB', 'GB', 'LGB']

optimized_scores.sort_values(by = 'Optimized Scores')

scatter_plot(optimized_scores.index, optimized_scores['Optimized Scores'], "Models' Scores after Optimization", 'Models','Optimized Scores', 40, 'Rainbow')
'''Initialize 9 object models with best hyperparameters'''

lasso_opt = Lasso(**lasso_best_params)

ridge_opt = Ridge(**ridge_best_params)

kernel_ridge_opt = KernelRidge(**kr_best_params)

elastic_net_opt = ElasticNet(**elastic_best_params)

rf_opt = RandomForestRegressor(**rf_best_params)

svm_opt = SVR(**svr_best_params)

xgb_opt = XGBRegressor(**xgb_best_params)

gb_opt = gb_opt

lgb_opt = lgb_opt
'''Now train and predict with optimized models'''

def predict_with_optimized_models(model):

    model.fit(df_train_final, y_train)

    y_pred = np.expm1(model.predict(df_test_final))

    submission = pd.DataFrame()

    submission['Id']= test.Id

    submission['SalePrice'] = y_pred

    return submission



'''Make submission with optimized lasso, ridge, kernel_ridge, elastic_net and svm, xgb, gb, and lgb.'''

predict_with_optimized_models(lasso_opt).to_csv('lasso_optimized.csv', index = False)

predict_with_optimized_models(ridge_opt).to_csv('ridge_optimized.csv', index = False)

predict_with_optimized_models(kernel_ridge_opt).to_csv('kernel_ridge_optimized.csv', index = False)

predict_with_optimized_models(elastic_net_opt).to_csv('elastic_net_optimized.csv', index = False)

predict_with_optimized_models(rf_opt).to_csv('rf_opt_optimized.csv', index = False)

predict_with_optimized_models(svm_opt).to_csv('svm_opt_optimized.csv', index = False)

predict_with_optimized_models(xgb_opt).to_csv('xgb_optimized.csv', index = False)

predict_with_optimized_models(gb_opt).to_csv('gb_optimized.csv', index = False)

predict_with_optimized_models(lgb_opt).to_csv('lgb_optimized.csv', index = False)