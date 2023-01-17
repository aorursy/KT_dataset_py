import pandas as pd

import numpy as np
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data.info()
train_data.describe()
y_train = train_data['Survived']
# Definition of the CategoricalEncoder class, copied from PR #9151.

# This is used until CategoricalEncoder is added to sklearn

import numpy as np

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

from sklearn.preprocessing import Imputer

from sklearn.ensemble import RandomForestClassifier



from sklearn.base import BaseEstimator, TransformerMixin



# Convert to numpy array since Sklearn doesn't handle it

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]



# Get most frequent value for field and set when null

class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],

                                       index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent)

    

# create age bucket field

# class AgeBucket(BaseEstimator, TransformerMixin):

#     def __init__(self, add_agebucket=True):

#         self.add_agebucket = add_agebucket

#     def fit(self, X, y=None):

#         return self

#     def transform(self, X, y=None):

#         X['AgeBucket'] = X['Age'] // 15 * 15

#         X.drop(['Age'], axis=1, inplace=True)

#         return X



# create class for family features (so we can run gridsearch against it)

class FamilyFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, add_familysize=True, add_isalone=True):

        self.add_familysize = add_familysize

        self.add_isalone = add_isalone

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        if self.add_familysize:

            X[3] = X[0] + X[1]

        if self.add_isalone:

            X[4] = X[0] + X[1] == 1

        return X





# Create numerical pipeline

num_pipeline = Pipeline([

    ('select_numeric', DataFrameSelector(['Age', 'SibSp', 'Parch', 'Fare'])),

    ('imputer', Imputer(strategy='median')),

    ('family_features', FamilyFeatures())

])



# Create categorical pipeline

cat_pipeline = Pipeline([

    ('select_cat', DataFrameSelector(['Pclass', 'Sex', 'Embarked'])),

    ('imputer', MostFrequentImputer()),

#     ('age_bucket', AgeBucket()),

    ('cat_encoder', CategoricalEncoder(encoding='onehot-dense'))

])



# Union data processing pipelines

preprocess_pipeline = FeatureUnion(transformer_list = [

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline)

])



# Create final pipeline with prediction built in

final_pipeline = Pipeline([

    ('prep', preprocess_pipeline),

    ('train_model', RandomForestClassifier()) # can test hyperparameters through gridsearch

])

# Review list of avaliable parameters that can be tuned

final_pipeline.get_params().keys()
from sklearn.model_selection import GridSearchCV



# Check if new features are worth it against other combinations

param_grid = [

    {'prep__num_pipeline__family_features__add_familysize': [True, False],

     'prep__num_pipeline__family_features__add_isalone': [True, False],

     'prep__num_pipeline__imputer__strategy':['median','mean','most_frequent'],

     'train_model__n_estimators': [10,15,20,25,30,35,40,45,50,55,60],

     'train_model__max_features': [2,3,4,5,6,7,8,9,10,11,12]}

    

]



grid_search_final = GridSearchCV(final_pipeline, param_grid, cv=5,

                                scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)



grid_search_final.fit(train_data, y_train);
grid_search_final.best_params_
from sklearn.metrics import accuracy_score



accuracy_score(grid_search_final.predict(train_data), y_train)
df_output = pd.concat([test_data['PassengerId'], 

                       pd.Series(grid_search_final.predict(test_data), name='Survived')], axis=1)
df_output.head()