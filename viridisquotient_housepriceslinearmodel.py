import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set_style("whitegrid")
from scipy import stats
from scipy.stats import skew
from scipy.stats import norm
from scipy.stats.stats import pearsonr

%matplotlib inline
pd.options.display.max_columns = 999
plt.rcParams['figure.figsize'] = (16, 9)
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_train['trainingset'] = True
df_test['trainingset'] = False
df_full = pd.concat([df_train, df_test], sort=False)
df_full.reset_index(drop=True, inplace=True) # Reset to avoid duplicate indices

df_train.shape, df_test.shape
df_full.tail()
fig, ax = plt.subplots(2, figsize=(16, 16))
sns.distplot(df_train['SalePrice'], ax=ax[0], axlabel=False)
ax[0].set_title('Target')
sns.distplot(df_train['SalePrice'].apply(np.log1p), ax=ax[1], axlabel=False)
ax[1].set_title('Log of Target');
df_full['SalePrice_log'] = np.log1p(df_full['SalePrice'])
df_train['SalePrice_log'] = np.log1p(df_train['SalePrice'])
df_full['MSSubClass'] = df_full['MSSubClass'].astype('O')
df_full.columns.to_series().groupby(df_full.dtypes).groups
for feature in df_full.columns:
    if df_full[feature].isnull().sum() > 50:
        print(feature, df_full[feature].isnull().sum(), df_full[feature].dtype)
plt.scatter(df_train['GrLivArea'], df_train['SalePrice_log'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice_log');
df_full.loc[df_full['trainingset'] == True].sort_values(by='GrLivArea', ascending=False)[:2]
df_full = df_full.drop(df_full[df_full['Id'] == 1299].index)
df_full = df_full.drop(df_full[df_full['Id'] == 524].index)
def examine_catf(feature, df, target):
    if df[feature].dtype != 'int64':
        df[feature] = df[feature].astype('str')
    vc = df[feature].value_counts(dropna=False)
    print(vc)
    print()
    sns.boxplot(x=feature, y=target, data=df)
examine_catf('PoolQC', df_train, 'SalePrice_log')
df_full['hasPool'] = df_full['PoolQC'].notnull().astype(int)
df_full.drop('PoolQC', axis=1, inplace=True)
plt.scatter(df_train['3SsnPorch'], df_train['SalePrice_log']);
df_full['has3SsnPorch'] = df_full['3SsnPorch'].notnull().astype(int)
df_full.drop('3SsnPorch', axis=1, inplace=True)
numeric_feats = df_full.dtypes[df_full.dtypes != 'object'].index
skewed_feats = df_full[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats.drop(['SalePrice', 'hasPool'], inplace=True)
skewed_feats
skewed_feats = skewed_feats.index
df_full[skewed_feats] = np.log1p(df_full[skewed_feats])
num_cols = df_full.dtypes[df_full.dtypes != 'object'].index
num_cols = num_cols.drop('trainingset')
from sklearn.preprocessing import Imputer

imp = Imputer(strategy='median')
df_full.loc[df_full['trainingset'] == True, num_cols] = imp.fit_transform(df_full.loc[df_full['trainingset'] == True, num_cols])
df_full.loc[df_full['trainingset'] == False, num_cols] = imp.transform(df_full.loc[df_full['trainingset'] == False, num_cols])
sns.distplot(df_full['TotalBsmtSF'], fit=norm);
df_full['hasBsmt'] = 0
df_full.loc[df_full['TotalBsmtSF'] > 0, 'hasBsmt'] = 1
df_full.loc[df_full['hasBsmt'] == 1, 'TotalBsmtSF'] = np.log(df_full['TotalBsmtSF'])
sns.distplot(df_full['TotalBsmtSF'][df_full['TotalBsmtSF'] > 0], fit=norm);
y = df_full.loc[df_full['trainingset'] == True]['SalePrice_log'].copy()
df_full.drop(['Id', 'SalePrice_log', 'SalePrice'], axis=1, inplace=True)
training = df_full.loc[df_full['trainingset'] == True].copy().drop('trainingset', axis=1)
testing = df_full.loc[df_full['trainingset'] == False].copy().drop('trainingset', axis=1)
num_cols = training.dtypes[training.dtypes != 'object'].index
cat_cols = training.dtypes[training.dtypes == 'object'].index
training[cat_cols] = training[cat_cols].astype(str)
testing[cat_cols] = testing[cat_cols].astype(str)
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
class ConvertToStrings(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.astype(str)
# Definition of the CategoricalEncoder class, copied from PR #9151.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

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
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_cols)), 
    ('imputer', Imputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_cols)),
    ('strings', ConvertToStrings()),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense",
                                       handle_unknown='ignore'))
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])
X_train = full_pipeline.fit_transform(training)
X_test = full_pipeline.transform(testing)
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=10).fit(X_train, y)
lasso.alpha_
lasso = LassoCV(alphas=[0.0001, 0.0004, 0.0007, 0.001, 0.0013, 0.0016, 0.0019], cv=10).fit(X_train, y)
lasso.alpha_
lasso_rmse = np.mean(np.sqrt(-cross_val_score(lasso, X_train, y, scoring='neg_mean_squared_error', cv=10)))
lasso_rmse
lasso_pred = lasso.predict(X_test)
lasso_results = pd.DataFrame({'Id': df_test['Id'].values,
                              'SalePrice': np.expm1(lasso_pred)})
lasso_results.to_csv('lasso_results.csv', index=False)
