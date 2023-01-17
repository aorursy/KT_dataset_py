import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



# Import Modules for Custom Transformers and Pipelines

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.utils import check_array

from sklearn.preprocessing import LabelEncoder

from scipy import sparse

from sklearn.preprocessing import (MinMaxScaler, StandardScaler, RobustScaler)





from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV)

from sklearn.model_selection import GridSearchCV





kfold = StratifiedKFold(n_splits=5)

rnd_st = 42





# Import Plotting-Modules

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

%matplotlib inline



import time

# Class to select DataFrames, since Scikit-Learn doesn't handles Pandas' DataFrames



class DFSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
# Definition of the CategoricalEncoder class, copied from PR #9151.

# This will be released in scikit-learn 0.20



class CategoricalEncoder(BaseEstimator, TransformerMixin):

    """Encode categorical features as a numeric array.

    The input to this transformer should be a matrix of integers or strings,

    denoting the values taken on by categorical (discrete) features.

    The features can be encoded using a one-hot aka one-of-K scheme

    (``encoding='onehot'``, the default) or converted to ordinal integers

    (``encoding='ordinal'``).

    This encoding is needed for feeding categorical data to many scikit-learn

    estimators, notably linear models and SVMs with the standard kernels.

    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

    Parameters

    ----------

    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'

        The type of encoding to use (default is 'onehot'):

        - 'onehot': encode the features using a one-hot aka one-of-K scheme

          (or also called 'dummy' encoding). This creates a binary column for

          each category and returns a sparse matrix.

        - 'onehot-dense': the same as 'onehot' but returns a dense array

          instead of a sparse matrix.

        - 'ordinal': encode the features as ordinal integers. This results in

          a single column of integers (0 to n_categories - 1) per feature.

    categories : 'auto' or a list of lists/arrays of values.

        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.

        - list : ``categories[i]`` holds the categories expected in the ith

          column. The passed categories are sorted before encoding the data

          (used categories can be found in the ``categories_`` attribute).

    dtype : number type, default np.float64

        Desired dtype of output.

    handle_unknown : 'error' (default) or 'ignore'

        Whether to raise an error or ignore if a unknown categorical feature is

        present during transform (default is to raise). When this is parameter

        is set to 'ignore' and an unknown category is encountered during

        transform, the resulting one-hot encoded columns for this feature

        will be all zeros.

        Ignoring unknown categories is not supported for

        ``encoding='ordinal'``.

    Attributes

    ----------

    categories_ : list of arrays

        The categories of each feature determined during fitting. When

        categories were specified manually, this holds the sorted categories

        (in order corresponding with output of `transform`).

    Examples

    --------

    Given a dataset with three features and two samples, we let the encoder

    find the maximum value per feature and transform the data to a binary

    one-hot encoding.

    >>> from sklearn.preprocessing import CategoricalEncoder

    >>> enc = CategoricalEncoder(handle_unknown='ignore')

    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])

    ... # doctest: +ELLIPSIS

    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,

              encoding='onehot', handle_unknown='ignore')

    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()

    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],

           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])

    See also

    --------

    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of

      integer ordinal features. The ``OneHotEncoder assumes`` that input

      features take on values in the range ``[0, max(feature)]`` instead of

      using the unique values.

    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of

      dictionary items (also handles string-valued features).

    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot

      encoding of dictionary items or strings.

    """



    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,

                 handle_unknown='error'):

        self.encoding = encoding

        self.categories = categories

        self.dtype = dtype

        self.handle_unknown = handle_unknown



    def fit(self, X, y=None):

        """Fit the CategoricalEncoder to X.

        Parameters

        ----------

        X : array-like, shape [n_samples, n_feature]

            The data to determine the categories of each feature.

        Returns

        -------

        self

        """



        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:

            template = ("encoding should be either 'onehot', 'onehot-dense' "

                        "or 'ordinal', got %s")

            raise ValueError(template % self.handle_unknown)



        if self.handle_unknown not in ['error', 'ignore']:

            template = ("handle_unknown should be either 'error' or "

                        "'ignore', got %s")

            raise ValueError(template % self.handle_unknown)



        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':

            raise ValueError("handle_unknown='ignore' is not supported for"

                             " encoding='ordinal'")



        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)

        n_samples, n_features = X.shape



        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]



        for i in range(n_features):

            le = self._label_encoders_[i]

            Xi = X[:, i]

            if self.categories == 'auto':

                le.fit(Xi)

            else:

                valid_mask = np.in1d(Xi, self.categories[i])

                if not np.all(valid_mask):

                    if self.handle_unknown == 'error':

                        diff = np.unique(Xi[~valid_mask])

                        msg = ("Found unknown categories {0} in column {1}"

                               " during fit".format(diff, i))

                        raise ValueError(msg)

                le.classes_ = np.array(np.sort(self.categories[i]))



        self.categories_ = [le.classes_ for le in self._label_encoders_]



        return self



    def transform(self, X):

        """Transform X using one-hot encoding.

        Parameters

        ----------

        X : array-like, shape [n_samples, n_features]

            The data to encode.

        Returns

        -------

        X_out : sparse matrix or a 2-d array

            Transformed input.

        """

        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)

        n_samples, n_features = X.shape

        X_int = np.zeros_like(X, dtype=np.int)

        X_mask = np.ones_like(X, dtype=np.bool)



        for i in range(n_features):

            valid_mask = np.in1d(X[:, i], self.categories_[i])



            if not np.all(valid_mask):

                if self.handle_unknown == 'error':

                    diff = np.unique(X[~valid_mask, i])

                    msg = ("Found unknown categories {0} in column {1}"

                           " during transform".format(diff, i))

                    raise ValueError(msg)

                else:

                    # Set the problematic rows to an acceptable value and

                    # continue `The rows are marked `X_mask` and will be

                    # removed later.

                    X_mask[:, i] = valid_mask

                    X[:, i][~valid_mask] = self.categories_[i][0]

            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])



        if self.encoding == 'ordinal':

            return X_int.astype(self.dtype, copy=False)



        mask = X_mask.ravel()

        n_values = [cats.shape[0] for cats in self.categories_]

        n_values = np.array([0] + n_values)

        indices = np.cumsum(n_values)



        column_indices = (X_int + indices[:-1]).ravel()[mask]

        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),

                                n_features)[mask]

        data = np.ones(n_samples * n_features)[mask]



        out = sparse.csc_matrix((data, (row_indices, column_indices)),

                                shape=(n_samples, indices[-1]),

                                dtype=self.dtype).tocsr()

        if self.encoding == 'onehot-dense':

            return out.toarray()

        else:

            return out
# Function to evaluate various Classifiers (Metrics and cross_val_score [CSV])

# Cross validate model with Kfold stratified cross val

# Props to Amit K Tiwary for the Snippet



def reg_cross_val_score_and_metrics(X, y, reg_dict, CVS_scoring, CVS_CV):

    # Train and Validation set split by model_selection

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rnd_st)

    metric_cols = ['reg_name', 'Score', 'CVS_Best', 'CVS_Mean', 'CVS_SD']

    reg_metrics = pd.DataFrame(columns = metric_cols)

    metric_dict = []

    

    # iterate over regressor   

    for reg_name, reg in reg_dict.items():

        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_val)

        Score = (reg.score(X_val, y_val))

        



        

        CVS_values = cross_validation.cross_val_score(estimator = reg, X = X, y = y, scoring = CVS_scoring, cv = CVS_CV)

        CVS_Best = (CVS_values.max())

        CVS_Mean = (CVS_values.mean())

        CVS_SD = (CVS_values.std())

        

        metric_values = [reg_name, Score, CVS_Best, CVS_Mean, CVS_SD]        

        metric_dict.append(dict(zip(metric_cols, metric_values)))

        

    reg_metrics = reg_metrics.append(metric_dict)

    # Change to float data type

    for column_name in reg_metrics.drop('reg_name', axis=1).columns:

        reg_metrics[column_name] = reg_metrics[column_name].astype('float')

    reg_metrics.sort_values('CVS_Mean', ascending=False, na_position='last', inplace=True)

    print(reg_metrics)

    

    reg_bp = sns.barplot(x='CVS_Mean', y='reg_name', data = reg_metrics, palette="viridis",orient = "h",**{'xerr':reg_metrics.CVS_SD})

    reg_bp.set_xlabel("Mean R2")

    reg_bp.set_ylabel("Regressors")

    reg_bp.set_title("Cross Validation Scores")


train = ("../input/train.csv")

test = ("../input/test.csv")



train_data = pd.read_csv(train)

test_data = pd.read_csv(test)
# Look at first 5 train data



train_data.head()
# Look at first 5 test data



test_data.head()
# Look at the shape



for i in train_data, test_data:

    print(i.shape)
# There are a few houses with more than 4000 sq ft (outliers)

train_data.drop(train_data[train_data["GrLivArea"] > 4000].index, inplace=True)
# Combine train and test data, reset index and drop old index

# 1456: is test_data with SalePrice NaN



comb_data = pd.concat((train_data, test_data)).reset_index(drop=True)
comb_data.SalePrice[1454:1458]
# IR2 and IR3 dont appear that often, just dinstinguish between Regular and not regular



comb_data["LotShape"] = comb_data.LotShape.replace({'IR1': "Irr", 'IR2': "Irr", 'IR3': "Irr"})

comb_data.LotShape.value_counts()
# Most properties are on Level. Just dinstinguish between Level and Not-Level

comb_data["LandContour"] = comb_data.LandContour.replace({'HLS': "NoLvl", 'Bnk': "NoLvl", 'Low': "NoLvl"})



comb_data.LandContour.value_counts()
# Most slopes are gentle. Just dinstinguish between gentle and not-gentle



comb_data["LandSlope"] = comb_data.LandSlope.replace({'Mod': "NoGtl", 'Sev': "NoGtl"})



comb_data.LandSlope.value_counts()
# Most properties use standard circuits breakers. Just dinstinguish between standard (SBrkr) and non Sbrkr



comb_data["Electrical"] = comb_data.Electrical.replace({'FuseA': "NoSbrkr", 'FuseF': "NoSbrkr", 'FuseP': "NoSbrkr", 'Mix': "NoSbrkr"})



comb_data.Electrical.value_counts()
# About 2/3rd have an attached garage. Treat others as Not Attchd



comb_data["GarageType"] = comb_data.GarageType.replace({'Detchd': "NotAttchd", 'BuiltIn': "NotAttchd", 'Basment': "NotAttchd", '2Types': "NotAttchd", 'CarPort': "NotAttchd"})



comb_data.GarageType.value_counts()
# Most have a paved drive, treat others as not paved



comb_data["PavedDrive"] = comb_data.PavedDrive.replace({'P': "N"})



comb_data.PavedDrive.value_counts()
# The only interesting "misc. feature" is if there is a shed or not



comb_data["MiscFeature"] = comb_data.MiscFeature.replace({'Gar2': "NoShed", 'Othr': "NoShed", 'TenC': "NoShed"})



comb_data.MiscFeature.value_counts()
# Was the house remodelled in the same year it was sold?  *1 to make Booleans to 0 and 1



comb_data["RecentRemodel"] = (comb_data["YearRemodAdd"] == comb_data["YrSold"]) *1

comb_data.RecentRemodel.value_counts()
# Was the sold in the same year it was built? New House - *1 to make Booleans to 0 and 1



comb_data["NewHouse"] = (comb_data["YearBuilt"] == comb_data["YrSold"]) *1

comb_data.NewHouse.value_counts()
# Add Features e.g. when there is are SquareFeet of a 2nd Floor it means 2nd Floor exists



comb_data["Has2ndFloor"] = (comb_data["2ndFlrSF"] != 0)*1

comb_data["HasMasVnr"] = (comb_data["MasVnrArea"] != 0) * 1

comb_data["HasWoodDeck"] = (comb_data["WoodDeckSF"] != 0) * 1

comb_data["HasOpenPorch"] = (comb_data["OpenPorchSF"] != 0) * 1

comb_data["HasEnclosedPorch"] = (comb_data["EnclosedPorch"] != 0) * 1

comb_data["Has3SsnPorch"] = (comb_data["3SsnPorch"] != 0) * 1

comb_data["HasScreenPorch"] = (comb_data["ScreenPorch"] != 0) * 1
# Month with most solds (HighSeason) could be significant



comb_data["HighSeason"] = comb_data["MoSold"].replace( {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
# Seperate SalePrice



comb_data_SalePrice = comb_data["SalePrice"].copy()



comb_data_Features = comb_data.drop("SalePrice", axis = 1)



comb_data_Features.shape, comb_data_SalePrice.shape
comb_data.SalePrice.isnull().sum()
# Check for Missing Values. Just print features with missing values



missing_v = []



for i in comb_data_Features.columns:

    if comb_data_Features[i].isnull().sum() == 0:

        pass

    else:

        missing_v.append([comb_data_Features[i].isnull().sum(), i])

        



def getKey(item):

    return item[0]



a = sorted(missing_v, key=getKey, reverse=True)

a



        

# We need to fix 34 Columns

# Create x and y for Plot



x_plot = np.asarray(a)[:,0].astype(int)

y_plot = np.asarray(a)[:,1]
#Plot features with missing vals



f, ax = plt.subplots(figsize=(12, 8))

sns.barplot(x = x_plot, y = y_plot)

plt.xlabel('Amount', fontsize=15)

plt.title('Missing Features', fontsize=15)

plt.show()
# Fix missing Values



# The following Features will be filled with "None", because their absence means, that they do exist

# e.g. PoolQC NaN means the House has no Pool





for col in ("PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish",

            "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", 

            "BsmtFinType2", "MasVnrType", "MSSubClass"):

    comb_data_Features[col] = comb_data_Features[col].fillna('None')

    

    

# doing the same for numerical features. Instead of None we replace it by 0



for col in ("GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",

            "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"):

    comb_data_Features[col] = comb_data_Features[col].fillna(0)





    

# Special fillings / removings



# We assume that the LotFrontage is similar for same Neigborhoods and fill these with respect to the Neighborhood

comb_data_Features["LotFrontage"] = comb_data_Features.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



#MSZoning will be filled with the most occurence which is "RL" for Residential Low Density



comb_data_Features['MSZoning'] = comb_data_Features['MSZoning'].fillna(comb_data_Features['MSZoning'].mode()[0])



# For Utilities all records are "AllPub", except for 1x "NoSeWa" and 2x NA. We will drop it.



comb_data_Features = comb_data_Features.drop('Utilities', axis=1)



# If Functionality is NaN it means typical functionality (typ)



comb_data_Features["Functional"] = comb_data_Features["Functional"].fillna("Typ")





# One NaN will be replaced by mode



comb_data_Features['Electrical'] = comb_data_Features['Electrical'].fillna(comb_data_Features['Electrical'].mode()[0])

comb_data_Features['KitchenQual'] = comb_data_Features['KitchenQual'].fillna(comb_data_Features['KitchenQual'].mode()[0])

comb_data_Features['Exterior1st'] = comb_data_Features['Exterior1st'].fillna(comb_data_Features['Exterior1st'].mode()[0])

comb_data_Features['Exterior2nd'] = comb_data_Features['Exterior2nd'].fillna(comb_data_Features['Exterior2nd'].mode()[0])

comb_data_Features['SaleType'] = comb_data_Features['SaleType'].fillna(comb_data_Features['SaleType'].mode()[0])



# Feature Engineering, adding one more Feature (Total Square Feet)



comb_data_Features['TotalSF'] = comb_data_Features['TotalBsmtSF'] + comb_data_Features['1stFlrSF'] + comb_data_Features['2ndFlrSF']

# looking at the skew of the data



from scipy.stats import norm, skew 



numeric_feats = ["LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea","BsmtFinSF1", 

               "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "TotalSF", "1stFlrSF", "2ndFlrSF", 

               "LowQualFinSF", "GrLivArea", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", 

               "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",

              "MiscVal", "MoSold", "YrSold", "TotalSF"]



# Check the skew of all numerical features

skewed_feats = comb_data_Features[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
#fixing the skew



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #comb_data[feat] += 1

    comb_data_Features[feat] = boxcox1p(comb_data_Features[feat], lam)

    

#comb_data[skewed_features] = np.log1p(comb_data[skewed_features])
# Set Pipelines for numerical and categorical Data



# Change objects into categorial features





num_attribs = ["LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea","BsmtFinSF1", 

               "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "TotalSF", "1stFlrSF", "2ndFlrSF", 

               "LowQualFinSF", "GrLivArea", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", 

               "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",

              "MiscVal", "MoSold", "YrSold"]    



cat_attribs_ordinal = ["LandSlope", "OverallQual", "OverallCond", "ExterQual", 

                       "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",

                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu",

                      "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence"]



cat_attribs_onehot = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",

                      "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType","HouseStyle",

                     "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Heating",

                      "CentralAir", "Electrical", "BsmtFullBath", "BsmtHalfBath", "FullBath", 

                       "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "GarageType",

                     "MiscFeature", "SaleType", "SaleCondition", "RecentRemodel", "NewHouse",

                      "Has2ndFloor", "HasMasVnr", "HasWoodDeck", "HasOpenPorch", "HasEnclosedPorch",

                      "Has3SsnPorch", "HasScreenPorch", "HighSeason"

                     ]



    

num_pipeline = Pipeline([

    ("DFSelector", DFSelector(num_attribs)),

    ("minmaxscaler", RobustScaler())

])



               

cat_pipeline_ordinal = Pipeline([

    ("DFSelector", DFSelector(cat_attribs_ordinal)),

    ("encoder", CategoricalEncoder(encoding = "ordinal")),

    ("minmaxscaler", MinMaxScaler()), #Otherwise some Classifiers (e. g. Poly SVC are too slow) 

])





cat_pipeline_onehot = Pipeline([

    ("DFSelector", DFSelector(cat_attribs_onehot)),

    ("encoder_onehot", CategoricalEncoder(encoding = "onehot"))

])





preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline_ordinal", cat_pipeline_ordinal),

        ("cat_pipeline_onehot", cat_pipeline_onehot)

    ]) 
comb_data_Features.shape
comb_data_Features_prepared = preprocess_pipeline.fit_transform(comb_data_Features)
# Split the prepared Train and Test Sets



X_train = comb_data_Features_prepared[:1456]

X_test = comb_data_Features_prepared[1456:]



y_train = comb_data_SalePrice[0:1456]
X_train.shape, y_train.shape, X_test.shape
#lookting at the sale price



sns.distplot(train_data.SalePrice , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_data.SalePrice)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
from scipy import stats

from scipy.stats import norm, skew #for some statistics



y_train = np.log1p(y_train)

(mu, sigma) = norm.fit(y_train)



#Check the new distribution 

sns.distplot(y_train , fit=norm);



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
y_train.head()
# import Regressors



from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet)



from sklearn.svm import SVR



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LassoLars

from sklearn.neighbors import KNeighborsRegressor

from sklearn import cross_validation

import xgboost as xgb



# Import Metric Modules

from sklearn.metrics import (accuracy_score, f1_score, log_loss, confusion_matrix)



# Import Model_Selection and Preprocessing Modules

from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV)

from sklearn.preprocessing import (Imputer, MinMaxScaler, StandardScaler)

kfold = StratifiedKFold(n_splits=5)
# Define set of Regressor

reg_dict = {

             "reg_lin" : LinearRegression(),

             "reg_ridge" : Ridge(alpha = 0.0006, random_state =rnd_st, max_iter = 50000),

             "reg_lasso" : Lasso(alpha = 0.0006, random_state =rnd_st, max_iter = 50000),

             "reg_elastic" : ElasticNet(alpha = 0.0006, l1_ratio = 0.5, random_state =rnd_st, max_iter = 50000),

    

             "reg_SVR_lin" : SVR(kernel="linear"),

             "reg_SVR_rbf" : SVR(kernel="rbf"),

             "reg_SVR_poly" : SVR(kernel="poly", degree=2),

    

             "reg_RF" : RandomForestRegressor(criterion="mse", n_jobs=-1, random_state=rnd_st), 

             "reg_DT" : DecisionTreeRegressor(criterion="mse",random_state=rnd_st) ,

             "reg_ExTree" : ExtraTreesRegressor(criterion="mse", n_jobs=-1, random_state=rnd_st),

             "reg_KNN" : KNeighborsRegressor(n_jobs=-1),

             "reg_AdaBoost" : AdaBoostRegressor(random_state=rnd_st),

             "reg_GrBoost" : GradientBoostingRegressor(random_state=rnd_st),

              "reg_XGBoost" : xgb.XGBRegressor(seed=rnd_st, nthread=-1)

           }
reg_cross_val_score_and_metrics(X=X_train, y=y_train, reg_dict=reg_dict, CVS_scoring = "r2", CVS_CV=5)
#Validation function

n_folds = 5



def valid(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)

    r2= (cross_val_score(model, X_train, y_train, scoring="r2", cv = kf))

    return(r2)
#Lasso

param= {



        'alpha': [0.0005, 0.0006],

        'max_iter' : [50000, 100000]

    }



lasso_reg = Lasso()

gs = GridSearchCV(lasso_reg, param_grid=param, cv=5, scoring='r2',

                                verbose=1, n_jobs=4)

gs_reg = gs.fit(X_train, y_train)

gs.best_score_ , gs.best_params_
lasso_reg = Lasso(alpha = 0.0005, max_iter = 50000)

lasso_reg.fit(X_train, y_train)



valid(lasso_reg).mean()

#ElasticNet

param= {



        'alpha': [0.0005],

        'max_iter' : [50000],

        'l1_ratio' : [0.9],

    

    }



elastic_reg = ElasticNet()

gs = GridSearchCV(elastic_reg, param_grid=param, cv=5, scoring='r2',

                                verbose=1, n_jobs=4)

gs_reg = gs.fit(X_train, y_train)

gs.best_score_ , gs.best_params_
elastic_reg = ElasticNet(alpha = 0.0005, max_iter = 50000, l1_ratio = 0.9)

elastic_reg.fit(X_train, y_train)



valid(elastic_reg).mean()
# GrBoost

param= {



        'n_estimators': [3000],

        'learning_rate' : [0.05],

        'max_depth' : [4],

        'max_features' : ["sqrt"],

        'min_samples_leaf' : [15],

        'min_samples_split' : [10],

    

    }



GrBoost_reg = GradientBoostingRegressor(loss="huber")

gs = GridSearchCV(GrBoost_reg, param_grid=param, cv=5, scoring='r2',

                                verbose=1, n_jobs=4)

gs_reg = gs.fit(X_train, y_train)

gs.best_score_ , gs.best_params_

GrBoost_reg = GradientBoostingRegressor(loss="huber", learning_rate=0.05, n_estimators=3000, max_depth=4, max_features="sqrt", min_samples_leaf=15, min_samples_split=10, random_state =42)

GrBoost_reg.fit(X_train, y_train)



valid(GrBoost_reg).mean()
XGBregr = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.00468, learning_rate=0.05, max_depth=3, min_child_weight=1.7818, n_estimators=2200, 

                           reg_alpha = 0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, nthread=-1, seed=42)

XGBregr.fit(X_train, y_train)



valid(XGBregr).mean()
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (lasso_reg, elastic_reg, GrBoost_reg, XGBregr))
valid(averaged_models).mean()
y_lasso = np.expm1((lasso_reg.predict(X_test)))

y_elastic = np.expm1((elastic_reg.predict(X_test)))

y_GrBoost = np.expm1((GrBoost_reg.predict(X_test)))

y_xgb = np.expm1((GrBoost_reg.predict(X_test)))

# 0.11778

blended = y_lasso * 0.0 + y_elastic * 0.4 + y_GrBoost * 0.2 + y_xgb * 0.4

blended
out = pd.DataFrame()

out['Id'] = test_data.iloc[:,[0]]

out['SalePrice'] = blended

out.to_csv('output_house_price.csv', index=False)