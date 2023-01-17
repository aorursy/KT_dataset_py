# libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import sklearn.linear_model as linear_model
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from IPython.display import HTML, display, display_html
import numbers
warnings.filterwarnings('ignore')
%matplotlib inline
# formatting
CONSTANT_DECIMALPLACE = 3
CONSTANT_STRDECIMALPLACE = '.3f'
pd.set_option('float_format', '{:.3f}'.format)
def RoundValueStr(val):
    return '{0:.3f}'.format(round(val, CONSTANT_DECIMALPLACE))

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
sns.set(style='darkgrid')
# load data
df_train = pd.read_csv('../input/training-set/train.csv')
var_response = 'SalePrice'
# preview data
print(df_train.shape)

var_info = pd.DataFrame('', index=df_train.columns, columns=['index', 'values'])
counter = 0
cols_categorical = []
for c in df_train.columns:
    var_info.loc[c, 'index'] = counter
    tmp = df_train[c].unique()
    if (all(isinstance(x, numbers.Number) for x in tmp) and len(tmp) > 2):
        var_info.loc[c, 'values'] = str(min(tmp)) + ' - ' + str(max(tmp))
    else:
        var_info.loc[c, "values"] = tmp
        cols_categorical.append(c)
    counter += 1

var_info
# get descriptive statistics for SalePrice
print(df_train[var_response].describe())
plt.title('Distribution for ' + var_response)
sns.distplot(df_train[var_response]);
plt.show()
# define quantitative and qualitative columns
quantitative = [f for f in df_train.columns if df_train.dtypes[f] != 'object']
quantitative.remove(var_response)
qualitative = [f for f in df_train.columns if df_train.dtypes[f] == 'object']

# use pd.melt to unpivot all quantitative variables and values
# data are "unpivoted" to the row axis, leaving just two non-identifier columns, "variable" and "value"
f = pd.melt(df_train, value_vars=quantitative)
g = sns.FacetGrid(f, col='variable',  col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.distplot, 'value')
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Distributions for Quantitative Variables');
# none of the quantitative column variables have normal distribution
# some of them are good candidates for log transformation: TotalBsmtSF, KitchenAbvGr, LotFrontage, LotArea and others
for c in qualitative:
    df_train[c] = df_train[c].astype('category')
    if df_train [c].isnull().any():
        df_train[c] = df_train[c].cat.add_categories(['Missing'])
        df_train[c] = df_train[c].fillna('Missing')
        
def boxplot(x,y, **kwargs):
    sns.boxplot (x=x, y=y)
    x = plt.xticks(rotation=90)
    
f = pd.melt(df_train, id_vars=[var_response], value_vars=qualitative[0:20])
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False, size=4)
g = g.map(boxplot, 'value', var_response)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Boxplots for Qualitative Variables');

f = pd.melt(df_train, id_vars=[var_response], value_vars=qualitative[20:])
g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False, size=4)
g = g.map(boxplot, 'value', var_response)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Boxplots for Qualitative Variables')
# Change the categorical data to numerical data. Avoid using get_dummies at this stage, because it will generate too many variables.
# function to add an encoded column for each categorical variable
def encode(frame, feature):
    ordering = pd.DataFrame()
    # add a column of 'val' to dataframe. values in 'val' is the values in a specific categorical column
    ordering['val'] = frame[feature].unique()
    # add index
    ordering.index = ordering.val
    # spmean is based on the average SalePrice for each categorical value (group by each categorical value)
    ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    # sort according to the spmean and rank it
    ordering = ordering.sort_values('spmean')
    ordering['ordering'] = range(1, ordering.shape[0]+1)
    # convert it to a dictionary
    ordering = ordering['ordering'].to_dict()
    
    # the a new column _E to the orginal df_train. 
    for cat, o in ordering.items():
        frame.loc[frame[feature]==cat, feature + '_E'] = o
# the new columns provide quick influence estimations of categorical variable on SalePrice
# here, we just want to get a rough idea, so consider other variables as fixed
qual_encoded = []

for q in qualitative:
    encode(df_train, q)
    qual_encoded.append(q+'_E')
print(qual_encoded)
corr = df_train[quantitative+[var_response]].corr(method='spearman')
f,ax = plt.subplots(figsize=(30,30))
g = sns.heatmap(corr, linewidths=0.2, square=True, annot=True, vmax=1, cmap='RdBu_r')
ax.set_title('Correlation between Quantitative Variables', fontsize=25)
plt.show()
# function to detect multicollinear variables
def GetMulticolVars(corr_matrix):
    dictvars = dict()
    varstodrop = []
    threshold_multicol = 0.8
    for i in range(len(corr_matrix.columns)):
        currcol = corr_matrix.columns[i]
        if (currcol != var_response):
            currcol_corrwithresponse = corr_matrix.loc[currcol,var_response]
            for j in range(len(corr_matrix.index)):
                currrow = corr_matrix.index[j]
                if (currrow != var_response and currcol != currrow):
                    if (np.absolute(corr_matrix.iloc[i,j]) >= threshold_multicol):
                        currvartodrop = currrow
                        currvartonotdrop = currcol
                        currrow_corrwithresponse = corr_matrix.loc[currrow,var_response]
                        tmpkey = currcol + ' > ' + currrow
                        if (np.absolute(currrow_corrwithresponse) > np.absolute(currcol_corrwithresponse)):
                            tmpkey = currrow + ' > ' + currcol
                            currvartodrop = currcol
                        if (tmpkey not in dictvars):
                            dictvars[tmpkey] = corr_matrix.iloc[i,j]
                            print('Strength: ' + tmpkey + ' (' + RoundValueStr(corr_matrix.iloc[i,j]) + ')')
                            if (currvartonotdrop not in varstodrop and currvartodrop not in varstodrop):
                                varstodrop.append(currvartodrop)
    for v in varstodrop:
        print('Choice to drop: ' + v)
    return varstodrop
multicollinearcols_quantitative = GetMulticolVars(corr)
corr = df_train[qual_encoded+[var_response]].corr(method='spearman')
f,ax = plt.subplots(figsize=(30,30))
g = sns.heatmap(corr, linewidths=0.2, square=True, annot=True, vmax=1, cmap='RdBu_r')
ax.set_title('Correlation between Qualitative Variables', fontsize=25)
plt.show()
multicollinearcols_qualitative = GetMulticolVars(corr)
corr_quali = df_train[[var_response]+qual_encoded].corr().iloc[:,0] 
quali_remove=[]
for f in qual_encoded:
    if (np.absolute(corr_quali[f]) < 0.1):
        quali_remove.append(f)
quali_remove
corr = df_train[qual_encoded+quantitative+[var_response]].corr(method='spearman')
f,ax = plt.subplots(figsize=(50,50))
g = sns.heatmap(corr, linewidths=0.2, square=True, annot=True, vmax=1, cmap='RdBu_r')
ax.set_title('Correlation between All Variables', fontsize=25)
plt.show()
multicollinearcols = GetMulticolVars(corr)
f = pd.melt(df_train[quantitative+[var_response]], id_vars=[var_response], var_name="variable")
g = sns.FacetGrid(f, col='variable', col_wrap=5, sharex=False, sharey=False)
g = g.map(sns.regplot, "value", var_response);
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Regression Plots for Quantitative Variables');
# Drop unhelpful variables
# drop multicolinearity column
multicollinearcolstodrop = ['MiscFeature']
columns=[i for i in df_train.columns if i not in multicollinearcolstodrop]
df_train = df_train[columns]

# drop qualitative columns which have < |0.1| coefficients
quali_drop = ['Street', 'Utilities', 'Condition2', 'Functional', 'MiscFeature']
columns = [i for i in df_train.columns if i not in quali_drop]
df_train = df_train[columns]

# drop PoolArea, LowQualFinSF because they have no linear relationship with SalePrice
todrop = ['PoolArea', 'LowQualFinSF']
columns = [i for i in df_train.columns if i not in todrop]
df_train = df_train[columns]

# update test data accordingly
df_test = pd.read_csv('../input/test-set/test.csv')
test_columns = [j for j in df_test.columns if j not in multicollinearcolstodrop]
df_test = df_test[test_columns]

test_columns=[j for j in df_test.columns if j not in quali_drop]
df_test = df_test[test_columns]

test_columns =[j for j in df_test.columns if j not in todrop]
df_test = df_test[test_columns]
# Update variables based on domain knowledge
# Create new variable: YrSinceSold, and drop YrSold
df_train['YrSinceSold'] = 2017-df_train['YrSold']
df_train.drop('YrSold', axis=1, inplace=True)

df_test['YrSinceSold'] = 2017-df_test['YrSold']
df_test.drop('YrSold', axis=1, inplace=True)

# remove MoSold
df_train.drop('MoSold', axis=1, inplace=True)
df_test.drop('MoSold', axis=1, inplace=True)
# drop other redundant variables based domain knowledge (not recommended, as it makes things worse)
df_train.drop('Exterior2nd', axis=1, inplace=True)
df_test.drop('Exterior2nd', axis=1, inplace=True)

df_train.drop('BsmtFinType2', axis=1, inplace=True)
df_test.drop('BsmtFinType2', axis=1, inplace=True)

#df_train.drop('OverallQual', axis=1, inplace=True)
#df_test.drop('OverallQual', axis=1, inplace=True)

df_train.drop('ExterQual', axis=1, inplace=True)
df_test.drop('ExterQual', axis=1, inplace=True)

df_train.drop('BsmtQual', axis=1, inplace=True)
df_test.drop('BsmtQual', axis=1, inplace=True)
#  Handle missing values
# variables with high % of missing values should be drop
# Alley, Fence, should be dropped. Based on domain knowledge, they are not important factors when we are buying property
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/len(df_train)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_data[missing_data['Total'] > 0]
df_train.drop(["Alley",'Fence', 'PoolQC'], axis=1, inplace=True)
# update test data accordingly
df_test.drop(["Alley",'Fence', 'PoolQC'], axis=1, inplace=True)
# impute LotFrontage missing values with median for df_train
LFbyneighborhood = df_train["LotFrontage"].groupby(df_train["Neighborhood"])
for key, group in LFbyneighborhood:
    idx = (df_train["Neighborhood"]==key) & (df_train["LotFrontage"].isnull())
    df_train.loc[idx, "LotFrontage"] = group.median()
        
# impute LotFrontage missing values with median for df_test
LFbyneighborhood = df_test["LotFrontage"].groupby(df_test["Neighborhood"])
for key, group in LFbyneighborhood:
    idx = (df_test["Neighborhood"]==key) & (df_test["LotFrontage"].isnull())
    df_test.loc[idx, "LotFrontage"] = group.median()
    
    
# define quantitative and qualitative columns
quantitative= [f for f in df_train.columns if df_train.dtypes[f] != 'object']
quantitative.remove(var_response)

qualitative= [f for f in df_train.columns if df_train.dtypes[f] == 'object']
# fill NA with 0 in 'MasVnrArea'
df_train["MasVnrArea"].fillna(0, inplace=True)

# same for test data
df_test["MasVnrArea"].fillna(0, inplace=True)
# Manage categorical variables and null values
total = df_train[qualitative].isnull().sum().sort_values(ascending=False)
percent = (df_train[qualitative].isnull().sum()/len(df_train)).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_data[missing_data['Total'] > 0]
# convert to ordinal variables
quality_dict = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
# ExterQual has been removed
# df_train['ExterQual'] = df_train['ExterQual'].map(quality_dict).astype(int)

df_train['ExterCond'] = df_train['ExterCond'].map(quality_dict).astype(int)
# BsmtQual has been removed
# df_train['BsmtQual'] = df_train['BsmtQual'].map(quality_dict)
# df_train['BsmtQual'] = df_train['BsmtQual'].fillna(0).astype(int)

df_train['BsmtCond'] = df_train['BsmtCond'].map(quality_dict)
df_train['BsmtCond'] = df_train['BsmtCond'].fillna(0).astype(int)
df_train['KitchenQual'] = df_train['KitchenQual'].map(quality_dict).astype(int)

df_train['HeatingQC'] = df_train['HeatingQC'].map(quality_dict).astype(int)
df_train['GarageQual'] = df_train['GarageQual'].map(quality_dict)
df_train['GarageQual'] = df_train['GarageQual'].fillna(0).astype(int)
df_train['GarageCond'] = df_train['GarageCond'].map(quality_dict)
df_train['GarageCond'] = df_train['GarageCond'].fillna(0).astype(int)
df_train["BsmtExposure"] = df_train["BsmtExposure"].map({"Missing": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)
# df_train["BsmtExposure"].unique()
bsmt_fin_dict = {"Missing": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
df_train["BsmtFinType1"] = df_train["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
# df_train["BsmtFinType1"].unique()
# BsmtFinType2 has been removed
# df_train["BsmtFinType2"] = df_train["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

# Functional has been removed
# df_train['Functional'] = df_train["Functional"].map({None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

df_train["GarageFinish"] = df_train["GarageFinish"].map({"Missing": 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

df_train['Electrical'] = df_train['Electrical'].map({'Mix': 1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr': 5}).astype(int)

df_train['PavedDrive'] = df_train['PavedDrive'].map({'N': 1, 'P':2, 'Y':3}).astype(int)

df_train['LotShape'] = df_train['LotShape'].map({'IR3': 1, 'IR2':2, 'IR1':3, 'Reg': 4}).astype(int)

df_train['LandSlope'] = df_train['LandSlope'].map({'Sev': 1, 'Mod':2, 'Gtl':3}).astype(int)

# Utilities has been removed
# df_train['Utilities']=df_train['Utilities'].map({'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4})

df_train['FireplaceQu'] = df_train['FireplaceQu'].map({"Missing": 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)

# repeat for test data

# convert to ordinal variables
quality_dict = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
# ExterQual has been removed
# df_test['ExterQual'] = df_test['ExterQual'].map(quality_dict).astype(int)

df_test['ExterCond'] = df_test['ExterCond'].map(quality_dict).astype(int)

# BsmtQual has been removed
# df_test['BsmtQual'] = df_test['BsmtQual'].map(quality_dict)
# df_test['BsmtQual'] = df_test['BsmtQual'].fillna(0).astype(int)

df_test['BsmtCond'] = df_test['BsmtCond'].map(quality_dict)
df_test['BsmtCond'] = df_test['BsmtCond'].fillna(0).astype(int)

df_test['KitchenQual'] = df_test['KitchenQual'].map(quality_dict).astype(int)

df_test['HeatingQC'] = df_test['HeatingQC'].map(quality_dict).astype(int)

df_test['GarageQual'] = df_test['GarageQual'].map(quality_dict)
df_test['GarageQual'] = df_test['GarageQual'].fillna(0).astype(int)

df_test['GarageCond'] = df_test['GarageCond'].map(quality_dict)
df_test['GarageCond'] = df_test['GarageCond'].fillna(0).astype(int)
df_test["BsmtExposure"] = df_test["BsmtExposure"].map({None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)
# df_test["BsmtExposure"].unique()

bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
df_test["BsmtFinType1"] = df_test["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
# BsmtFinType2 has been removed
# df_test["BsmtFinType2"] = df_test["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

# Functional has been removed
# df_test['Functional'] = df_test["Functional"].map({None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

df_test["GarageFinish"] = df_test["GarageFinish"].map({None: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

df_test['Electrical'] = df_test['Electrical'].map({None: 0, 'Mix': 1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr': 5}).astype(int)

df_test['PavedDrive'] = df_test['PavedDrive'].map({'N': 1, 'P':2, 'Y':3}).astype(int)

df_test['LotShape'] = df_test['LotShape'].map({'IR3': 1, 'IR2':2, 'IR1':3, 'Reg': 4}).astype(int)

df_test['LandSlope'] = df_test['LandSlope'].map({'Sev': 1, 'Mod':2, 'Gtl':3}).astype(int)

# Utilities has been removed
# df_test['Utilities']=df_test['Utilities'].map({'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4})

df_test['FireplaceQu'] = df_test['FireplaceQu'].map({None: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}).astype(int)
# Deal with nominal variables
nominal_columns = [f for f in df_train.columns if df_train.dtypes[f]=='object']
nominal_columns = [nominal_columns+['MSSubClass']]
# define a function to to convert categorical features into ordinal numbers
# we can compare the results between this method and get_dummies. For most variables, get_dummies works better
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# df can be df_train or df_test
def factorize(df, column, fill_na=None):
    if fill_na is not None:
        df[column].fillna(fill_na, inplace=True)
    le.fit(df[column].unique())
    df[column] = le.transform(df[column])
    return df[column]

df_train['LandContour']=factorize(df_train, "LandContour")
df_train['GarageType']=df_train['GarageType'].map({"Missing": 0, 'Detechd': 1, 'CarPort': 2, 'BuiltIn':3, 'Basment': 4, 'Attchd': 5, '2Types': 6})
df_train['GarageType']=df_train['GarageType'].fillna(0).astype(int)

df_test['LandContour']=factorize(df_test, "LandContour")
df_test['GarageType']=df_test['GarageType'].map({None: 0, 'Detechd': 1, 'CarPort': 2, 'BuiltIn':3, 'Basment': 4, 'Attchd': 5, '2Types': 6})
df_test['GarageType']=df_test['GarageType'].fillna(0).astype(int)
df_train.columns[df_train.dtypes == "category"]
# df_train.columns

# [f for f in df_train.columns if df_train.dtypes[f]== "category"]
# nominal_columns=[f for f in df_train.columns if df_train.dtypes[f]== "category"]

# df_train['MSZoning']=factorize(df_train, "MSZoning", "RL")
# df_train['Street']=factorize(df_train, "Street")
#df_train['LotConfig']=factorize(df_train, "LotConfig")
#df_train['Condition1']=factorize(df_train, "Condition1")
#df_train['Condition2']=factorize(df_train, "Condition2")
#df_train['BldgType']=factorize(df_train, "BldgType")
#df_train['HouseStyle']=factorize(df_train, "HouseStyle")
#df_train['RoofStyle']=factorize(df_train, "RoofStyle")
#df_train['RoofMatl']=factorize(df_train, "RoofMatl")
#df_train['BldgType']=factorize(df_train, "BldgType")
#df_train['Exterior1st']=factorize(df_train, "Exterior1st")
#df_train['Heating']=factorize(df_train, 'Heating')
#df_train['CentralAir']=factorize(df_train, 'CentralAir')
#df_train['Foundation']=factorize(df_train, 'Foundation')
#df_train['SaleType']=factorize(df_train, 'SaleType')

# fill NA in GarageType with 0, and convert to ordinal variable


# same manipulation for test data
#df_test['MSZoning']=factorize(df_test, "MSZoning", "RL")
#df_test['Street']=factorize(df_test, "Street")
#df_test['LotConfig']=factorize(df_test, "LotConfig")
#df_test['Condition1']=factorize(df_test, "Condition1")
#df_test['Condition2']=factorize(df_test, "Condition2")
#df_test['BldgType']=factorize(df_test, "BldgType")
#df_test['HouseStyle']=factorize(df_test, "HouseStyle")
#df_test['RoofStyle']=factorize(df_test, "RoofStyle")
#df_test['RoofMatl']=factorize(df_test, "RoofMatl")
#df_test['BldgType']=factorize(df_test, "BldgType")
#df_test['Exterior1st']=factorize(df_test, "Exterior1st")
#df_test['Heating']=factorize(df_test, 'Heating')
#df_test['CentralAir']=factorize(df_test, 'CentralAir')
#df_test['Foundation']=factorize(df_test, 'Foundation')
#df_test['SaleType']=factorize(df_test, 'SaleType')

nominal_columns= df_train.columns[df_train.dtypes == "category"]
df_train_dummies=pd.get_dummies(df_train[nominal_columns], drop_first=True)
# ImStucc is not a value of 'Exterior1st' column in test data, therefore, drop this value to keep consistency
df_train_dummies.drop(['Exterior1st_ImStucc', 'SaleType_Oth', 'SaleType_VWD', 'Heating_GasA', 'RoofMatl_Membran'], axis=1, inplace=True)
df_train=pd.concat([df_train, df_train_dummies], axis=1)
df_train.drop(nominal_columns, axis=1, inplace=True)

# update the test data
df_test_dummies=pd.get_dummies(df_test[nominal_columns], drop_first=True)
# drop columns that are not in training data
df_test_dummies.drop(['Neighborhood_GrnHill', 'Neighborhood_Landmrk', 'MSZoning_C (all)', 
                      'Exterior1st_AsphShn', 'Exterior1st_CBlock',
                      'Exterior1st_PreCast', 'RoofMatl_Metal', 'RoofMatl_Roll'], axis=1, inplace=True)
df_test=pd.concat([df_test, df_test_dummies], axis=1)
df_test.drop(nominal_columns, axis=1, inplace=True)

var_response='SalePrice'
df_train[['YearBuilt', var_response]][df_train[var_response]>500000]
df_train[['YearRemod/Add', var_response]][df_train[var_response]>500000]

# functions for checking possible transformations
def Power2(val):
    return np.power(val, 2)

def Power3(val):
    return np.power(val, 3)

def CheckSkewness(currdf):
    df = currdf.copy()
    df2 = currdf.copy()
    functotry = [np.log, Power2, Power3, np.reciprocal, np.sqrt, np.cbrt, np.exp]
    funcnames = ["log", "power2", "power3", "reciprocal", "sqrt", "cuberoot", "exponent"]
    dictimprovs = dict()
    for c in df.columns:
        pre_skew = np.absolute(df[c].skew())
        improvs = dict()
        fcounter = 0
        for f in functotry:
            df[c] = f(df2[c])
            if (df[c].isnull().values.any() == False):
                post_skew = np.absolute(df[c].skew())
                if (pre_skew > post_skew):
                    diff_skew = pre_skew - post_skew
                    improvs[funcnames[fcounter]] = diff_skew
            fcounter += 1
        listup = sorted(improvs.items(), key=lambda x:x[1], reverse=True)
        if (len(listup) > 0):
            for tup in listup:
                funcname, improv_val = tup
                if (improv_val > 0.5 and not df[c].empty): # ignore if too insignificant
                    functouse = functotry[funcnames.index(funcname)]
                    df[c] = functouse(df2[c])
                    if (not df[c].empty and len(df[c].unique()) > 3):
                        try:
                            fig = plt.figure()
                            plt.title(c + ': Distribution plot after ' + funcname)
                            sns.distplot(df[c], fit=norm)
                            fig = plt.figure()
                            res = stats.probplot(df[c], plot=plt)
                            try:
                                strfunc, funcval = tup
                                print(c + ": " + strfunc + ": " + RoundValueStr(funcval))
                                dictimprovs[c] = funcname
                            except:
                                print(c + ": " + str(tup))
                            break
                        except BaseException as e:
                            continue
                            #print(c + ": exception: " + str(e))
                        
    return dictimprovs
colsfordummyvars = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', '2ndFlrSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']
init_df_train = pd.read_csv('../input/training-set/train.csv')
initcols_quant = [f for f in init_df_train.columns if init_df_train.dtypes[f] != 'object']
currcols_quant = [i for i in df_train.columns if (i in initcols_quant and i not in colsfordummyvars)]
dicttransformfunc = CheckSkewness(df_train[currcols_quant])
# transform selected training data
#df_train['LotArea'] = np.log(df_train['LotArea'])
#df_train['OverallCond'] = np.log(df_train['OverallCond'])
df_train['1stFlrSF'] = np.log(df_train['1stFlrSF'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#df_train['TotRmsAbvGrd'] = np.cbrt(df_train['TotRmsAbvGrd'])
#df_train['BsmtUnfSF'] = np.sqrt(df_train['BsmtUnfSF'])
# transform test data
#df_test['LotArea'] = np.log(df_test['LotArea'])
#df_train['OverallCond'] = np.log(df_train['OverallCond'])
df_test['1stFlrSF'] = np.log(df_test['1stFlrSF'])
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])
#df_test['TotRmsAbvGrd'] = np.cbrt(df_test['TotRmsAbvGrd'])
#df_test['BsmtUnfSF'] = np.sqrt(df_test['BsmtUnfSF'])
# after log transformation, the distribution of SalePrice is nearly normal
plt.title('Distribution plot for SalePrice')
sns.distplot(df_train[var_response], fit=norm)
fig = plt.figure()
res = stats.probplot(df_train[var_response], plot=plt)
# Regression plot after transformation
#transformed = ['LotArea', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'BsmtUnfSF']
transformed = ['1stFlrSF', 'GrLivArea']
f = pd.melt(df_train[transformed+[var_response]], id_vars=[var_response], var_name="variable")
g = sns.FacetGrid(f, col="variable", col_wrap=5, sharex=False, sharey=False)
g.map(sns.regplot, "value", var_response);
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Regression Plots for Transformed Variables');
# Create new dummy variables for quant variable which contains zeros
df_train['HasMasVnr']=(df_train['MasVnrArea']!=0)*1
df_train['HasBsmtFinSF1']=(df_train['BsmtFinSF1']!=0)*1
df_train['HasBsmtFinSF2']=(df_train['BsmtFinSF2']!=0)*1
df_train['Has2ndFlr']=(df_train['2ndFlrSF']!=0)*1
df_train['HasGarage']=(df_train['GarageArea']!=0)*1
df_train['HasWoodDeck']=(df_train['WoodDeckSF']!=0)*1
df_train['HasOpenPorch']=(df_train['OpenPorchSF']!=0)*1
df_train['HasEnclosedPorch']=(df_train['EnclosedPorch']!=0)*1
df_train['Has3SsnPorch']=(df_train['3SsnPorch']!=0)*1
df_train['HasScreenPorch']=(df_train['ScreenPorch']!=0)*1
df_train['HasMiscVal']=(df_train['MiscVal']!=0)*1

df_test['HasMasVnr']=(df_test['MasVnrArea']!=0)*1
df_test['HasBsmtFinSF1']=(df_test['BsmtFinSF1']!=0)*1
df_test['HasBsmtFinSF2']=(df_test['BsmtFinSF2']!=0)*1
df_test['Has2ndFlr']=(df_test['2ndFlrSF']!=0)*1
df_test['HasGarage']=(df_test['GarageArea']!=0)*1
df_test['HasWoodDeck']=(df_test['WoodDeckSF']!=0)*1
df_test['HasOpenPorch']=(df_test['OpenPorchSF']!=0)*1
df_test['HasEnclosedPorch']=(df_test['EnclosedPorch']!=0)*1
df_test['Has3SsnPorch']=(df_test['3SsnPorch']!=0)*1
df_test['HasScreenPorch']=(df_test['ScreenPorch']!=0)*1
df_test['HasMiscVal']=(df_test['MiscVal']!=0)*1

train_y=df_train[var_response]

cols=[i for i in df_test.columns if i != "Id"]
train_x=df_train[cols]

test=df_test[cols]


# fill NA value in LotFrontage column with mean
test['LotFrontage'].fillna(np.mean(test['LotFrontage']), inplace=True)

# fill Na in BsmtFullBath and BsmtHalfBath  with 0
test.fillna(0, inplace=True)
train_x.fillna(0, inplace=True)
# there is no need to standardise the data for tree method

from sklearn.cross_validation import train_test_split
train_x_tree= train_x.copy()
test_x_tree=test.copy()
X_train_tree, X_val_tree, y_train_tree, y_val_tree=train_test_split(train_x_tree,train_y, test_size=0.2, random_state=1)


mu=train_x.mean()
sigma=train_x.std()
train_x=(train_x-mu)/sigma

# standardise the test data
test_x=(test-mu)/sigma

# split data into training and validation set
from sklearn.cross_validation import train_test_split
X_train, X_val, y_train, y_val=train_test_split(train_x,train_y, test_size=0.2, random_state=1)





