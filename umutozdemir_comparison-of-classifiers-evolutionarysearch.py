# Import Numpy and Pandas 

import numpy as np

import pandas as pd



# Import os

import os



# Import Model_Selection and Preprocessing Modules

from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV)

from sklearn.preprocessing import (Imputer, MinMaxScaler, StandardScaler)

kfold = StratifiedKFold(n_splits=5)

rand_st = 42



# Import Modules for Custom Transformers and Pipelines

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import check_array

from sklearn.preprocessing import LabelEncoder

from scipy import sparse



# Import Metric Modules

from sklearn.metrics import (accuracy_score, f1_score, log_loss, confusion_matrix)



# Import EvolutionaryAlgorithm

from evolutionary_search import EvolutionaryAlgorithmSearchCV



# Import Classifiers

from sklearn.linear_model import LogisticRegression



from sklearn.tree import DecisionTreeClassifier



from sklearn.ensemble import (RandomForestClassifier,

                              RandomForestRegressor,

                              ExtraTreesClassifier,

                              AdaBoostClassifier,

                              GradientBoostingClassifier,

                              VotingClassifier)



from xgboost import XGBClassifier



from sklearn.svm import LinearSVC, SVC



from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF



from sklearn.neighbors import KNeighborsClassifier



from sklearn.naive_bayes import GaussianNB



from sklearn.neural_network import MLPClassifier



# Import Plotting-Modules

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

%matplotlib inline



# Import Time

import time
# Class to select DataFrames, since Scikit-Learn doesn't handles Pandas' DataFrames



class DFSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names]
# Most Frequent Imputer for categorical Data

# Inspired from stackoverflow.com/questions/25239958



class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X],index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent)
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



def clf_cross_val_score_and_metrics(X, y, clf_dict, CVS_scoring, CVS_CV):

    # Train and Validation set split by model_selection

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rand_st)

    metric_cols = ['clf_name', 'Score', 'Accu_Preds', 'F1_Score', 'CVS_Best', 'CVS_Mean', 'CVS_SD']

    clf_metrics = pd.DataFrame(columns = metric_cols)

    metric_dict = []

    

    # iterate over classifiers   

    for clf_name, clf in clf_dict.items():

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        Score = (clf.score(X_val, y_val))

        Accu_Preds = accuracy_score(y_val, y_pred, normalize=False)

        F1_Score = (f1_score(y_val, y_pred))

        

        CVS_values = cross_val_score(estimator = clf, X = X, y = y, scoring = CVS_scoring, cv = CVS_CV, n_jobs=-1)

        CVS_Best = (CVS_values.max())

        CVS_Mean = (CVS_values.mean())

        CVS_SD = (CVS_values.std())

        

        metric_values = [clf_name, Score, Accu_Preds, F1_Score, CVS_Best, CVS_Mean, CVS_SD]        

        metric_dict.append(dict(zip(metric_cols, metric_values)))

        

    clf_metrics = clf_metrics.append(metric_dict)

    # Change to float data type

    for column_name in clf_metrics.drop('clf_name', axis=1).columns:

        clf_metrics[column_name] = clf_metrics[column_name].astype('float')

    clf_metrics.sort_values('CVS_Mean', ascending=False, na_position='last', inplace=True)

    print(clf_metrics)

    

    clf_bp = sns.barplot(x='CVS_Mean', y='clf_name', data = clf_metrics, palette="viridis",orient = "h",**{'xerr':clf_metrics.CVS_SD})

    clf_bp.set_xlabel("Mean Accuracy")

    clf_bp.set_ylabel("Classifiers")

    clf_bp.set_title("Cross Validation Scores")
# Define function to plot learning curves

# Props to Amit K Tiwary for the Snippet



def plot_learning_curve(estimator, title, X_train, y_train, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
# Define train and test data



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/train.csv')
# Split train_data into Features and Labels



comb_data = pd.concat([train_data, test_data])



X_comb = comb_data.drop(['Survived'], axis = 1)



X_train = train_data.drop(['Survived'], axis = 1)

y_train = train_data['Survived']



X_test = test_data
X_comb.shape, X_train.shape, X_test.shape, y_train.shape
# Set Pipelines for numerical and categorical Data



num_attribs = ["Age", "SibSp", "Parch", "Fare"]    

    

num_pipeline = Pipeline([

    ("DFSelector", DFSelector(num_attribs)),

    ("imputer", Imputer(strategy = "median")),

    ("scaler", StandardScaler())

])





cat_attribs = ["Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked"]

               

cat_pipeline = Pipeline([

    ("DFSelector", DFSelector(cat_attribs)),

    ("imputer", MostFrequentImputer()),

    ("encoder", CategoricalEncoder(encoding = "ordinal")),

    ("minmaxscaler", MinMaxScaler() ) #Otherwise some Classifiers (e. g. Poly SVC are too slow)

])





preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline)

    ])
# Minimalistic preprocessing for the Classifiers



# Use combined Data (train and test) to make sure that EVERY categorical attribute is been seen. 

# Otherwise it might be there a categories, which were  just encoded for the train data, but not for test data.



X_comb_prep = preprocess_pipeline.fit_transform(X_comb)



X_comb_prep[0:891]
# Define set of classifiers

clf_dict = {"clf_Log_reg" : LogisticRegression(random_state=rand_st),

            "clf_Lin_SVC" : LinearSVC(random_state=rand_st), 

            "clf_Poly_SVC" : SVC(kernel="poly", degree=2, random_state=rand_st),

            "clf_Ker_SVC" : SVC(kernel="rbf", random_state=rand_st), 

            "clf_KNN" : KNeighborsClassifier(algorithm='auto'), 

            "clf_GNB" : GaussianNB(),

            "clf_Dec_tr" : DecisionTreeClassifier(random_state=rand_st),

            "clf_RF" : RandomForestClassifier(random_state=rand_st, n_jobs=-1),

            "clf_AdaBoost" : AdaBoostClassifier(algorithm='SAMME.R', random_state=rand_st),

            "clf_GrBoost" : GradientBoostingClassifier(random_state=rand_st),

            "clf_ExTree" : ExtraTreesClassifier(random_state=rand_st, n_jobs=-1),

            "clf_XGBoost" : XGBClassifier(seed=rand_st),

    #slow Classifiers

           "clf_MLP" : MLPClassifier(learning_rate_init=0.05, hidden_layer_sizes = 24, shuffle=True, random_state=rand_st)          

           }
# Intermediary results before Feature Engineering



clf_cross_val_score_and_metrics(X=X_comb_prep[0:891], y=y_train, clf_dict=clf_dict, CVS_scoring = "accuracy", CVS_CV=kfold)
# Make a fresh copy for feature engineered X

X_fe = comb_data.copy()
# Change objects to categories



cat_attribs = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]



for col in cat_attribs:

    X_fe[col] = X_fe[col].astype("category")

    

X_fe.dtypes
# Describe the numerical and categorical Data



print(X_fe.describe(), '\n', comb_data.describe(include = ["object"]))
# Missing Values



X_fe.isnull().sum()
sns.countplot(X_fe['Pclass'], hue = X_fe['Survived'])
X_fe.head()
# Extract the Title

# Alternative: train['Title'] = X_fe['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])



X_fe["Title"] = X_fe["Name"].str.extract(' ([A-Za-z]+\.)')



X_fe.head()
# Unique values for Title

X_fe["Title"].value_counts()
# Replace equal Title and create grouped Titles



X_fe["Title"] = X_fe["Title"].replace("Mlle.", "Miss.")

X_fe["Title"] = X_fe["Title"].replace("Ms.", "Miss.")

X_fe["Title"] = X_fe["Title"].replace("Mme.", "Mrs.")



X_fe["Title"] = X_fe["Title"].replace(["Col.", "Major.", "Sir.", "Lady.", "Don.", "Dona.", "Capt.", "Countess.", "Jonkheer."], "GT")

X_fe['Survived'].groupby(X_fe['Title']).mean()
X_fe['Survived'].groupby(X_fe['Sex']).mean()
X_fe['Ticket_Letter'] = X_fe['Ticket'].apply(lambda x: str(x)[0] )



X_fe['Survived'].groupby(X_fe['Ticket_Letter']).mean()
# Add Namelength as Feature

# Alternative: X_fe['NameLength'] = X_fe['Name'].apply(lambda x: len(x))



X_fe["NameLength"] = X_fe["Name"].str.len()
X_fe['Survived'].groupby(pd.qcut(X_fe['NameLength'], 5)).mean()   # 5 Bins
X_fe.head()
# Seperate Cabin Prefix and its Number as own feature. We assume, that Cabin Prefix are different Decks.

# It might be, that there were less survival chance for low decks and high room numbers, if as example

# tail of the ship sank first and high room numbers are correspondig to the tail.



# First replace NaN by "N0" for "Nodeck". We assume that N0 has the lowest Fareprices.



X_fe["Cabin"].replace(np.nan, "N0.0", inplace=True)



# Extract Prefix



X_fe["Cabin_Prefix"] = X_fe["Cabin"].str.extract('(^.)')



# Extract Cabin Numbers



X_fe["Cabin_Number"] = X_fe["Cabin"].str.extract('(\d+)')



X_fe.head()

X_fe.drop(["PassengerId", "Name", "Cabin", 'Ticket'], axis = 1, inplace = True)
X_fe.head()
# Set Pipelines for numerical and categorical Data



num_attribs = ["Age", "SibSp", "Parch", "Fare", "NameLength", "Cabin_Number"]    

cat_attribs_ordinal = ["Pclass", "Sex", "Embarked", "Cabin_Prefix", "Ticket_Letter"]

cat_attribs_onehot = ["Title"]

    

num_pipeline = Pipeline([

    ("DFSelector", DFSelector(num_attribs)),

    ("imputer", Imputer(strategy = "median")),

    ("scaler", StandardScaler())

])



               

cat_pipeline_ordinal = Pipeline([

    ("DFSelector", DFSelector(cat_attribs_ordinal)),

    ("imputer", MostFrequentImputer()),

    ("encoder", CategoricalEncoder(encoding = "ordinal")),

    ("minmaxscaler", MinMaxScaler()), #Otherwise some Classifiers (e. g. Poly SVC are too slow) 

])





cat_pipeline_onehot = Pipeline([

    ("DFSelector", DFSelector(cat_attribs_onehot)),

    ("encoder_onehot", CategoricalEncoder(encoding = "onehot-dense"))

])





preprocess_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline_ordinal", cat_pipeline_ordinal),

        ("cat_pipeline_onehot", cat_pipeline_onehot)

    ])
X_fe_prepared = preprocess_pipeline.fit_transform(X_fe)
# Define set of classifiers

clf_dict_2 = {"clf_Log_reg" : LogisticRegression(random_state=rand_st),

              "clf_Lin_SVC" : LinearSVC(random_state=rand_st), 

              "clf_Poly_SVC" : SVC(kernel="poly", degree=2, random_state=rand_st),

              "clf_Ker_SVC" : SVC(kernel="rbf", random_state=rand_st), 

              "clf_KNN" : KNeighborsClassifier(algorithm='auto'), 

              "clf_GNB" : GaussianNB(),

              "clf_Dec_tr" : DecisionTreeClassifier(random_state=rand_st),

              "clf_RF" : RandomForestClassifier(random_state=rand_st, n_jobs=-1),

              "clf_AdaBoost" : AdaBoostClassifier(algorithm='SAMME.R', random_state=rand_st),

              "clf_GrBoost" : GradientBoostingClassifier(random_state=rand_st),

              "clf_ExTree" : ExtraTreesClassifier(random_state=rand_st, n_jobs=-1),

              "clf_XGBoost" : XGBClassifier(seed=rand_st),

    #slow Classifiers

              "clf_MLP" : MLPClassifier(learning_rate_init=0.05, hidden_layer_sizes = 10, shuffle=True, random_state=rand_st)          

           }
# Show intermediary results after feature engineering and with default hyperparameters



clf_cross_val_score_and_metrics(X_fe_prepared[0:891], y_train, clf_dict_2, CVS_scoring = "accuracy", CVS_CV = kfold)

# Define Classifier Params



clf_XGBoost_df = XGBClassifier(seed= rand_st , nthread = 4)

clf_XGBoost_p0 = [{}]

   

clf_XGBoost_gs = XGBClassifier(seed= rand_st , nthread = 4)

clf_XGBoost_pg = [{

    'max_depth': np.logspace(0.3,4,num = 10 ,base=10,dtype='int'), #[1, 5, 13, 34, 87, 226, 584, 1505, 3880, 10000]

    'learning_rate': np.logspace(-0.99, 1, num=10, base=10), # [0.102, 0.170, 0.283, 0.471, 0.784,1.304, 2.171, 3.612, 6.010, 10.0]

    'n_estimators' : np.logspace(0.1,3,num = 10 ,base=10,dtype='int'), #[1, 2, 5, 11, 24, 51, 107, 226, 476, 1000]

    'gamma' : np.logspace(-0.99, 1, num=10, base=10), # [0.102, 0.170, 0.283, 0.471, 0.784,1.304, 2.171, 3.612, 6.010, 10.0]

    'min_child_weight' : np.logspace(-0.1, 1, num=5, base=10, dtype=int), # [0, 1, 2, 5, 10]

    'max_delta_step' : np.logspace(-0.1, 1, num=5, base=10, dtype=int), # [0, 1, 2, 5, 10]

    'subsample' : [0, 0.5, 1], # [0, 0.5, 1]

    

    }]

  



clf_ExTree_df = ExtraTreesClassifier(random_state=rand_st)

clf_ExTree_p0 = [{}]



clf_ExTree_gs = ExtraTreesClassifier(random_state=rand_st)

clf_ExTree_pg = [{

    'max_depth': np.logspace(0.3,4,num = 10 ,base=10,dtype='int'), #[1, 5, 13, 34, 87, 226, 584, 1505, 3880, 10000]

    'n_estimators' : np.logspace(0.1,3,num = 10 ,base=10,dtype='int'), #[1, 2, 5, 11, 24, 51, 107, 226, 476, 1000]

    'min_samples_split' : np.logspace(0.4, 1, num=5, base=10, dtype=int), #[2, 3, 5, 7, 10]

    'min_samples_leaf' : np.logspace(0.1,1,num = 4 ,base=9,dtype='int'), #[1, 2, 4, 9]

    'max_features' : ['auto', None]

              }]

    

    

clf_KER_SVC_df = SVC(random_state=rand_st)

clf_KER_SVC_p0 = [{'kernel': ['rbf']}]

    

clf_KER_SVC_gs = SVC(random_state=rand_st)

clf_KER_SVC_pg = [{

    'kernel': ['rbf'],

    'C'     : np.logspace(0.1,1,num = 4 ,base=9,dtype='int'), #[1, 2, 4, 9]

    'gamma' : np.logspace(-0.99, 1, num=10, base=10), # [0.102, 0.170, 0.283, 0.471, 0.784,1.304, 2.171, 3.612, 6.010, 10.0]



}]







clf_RF_df = RandomForestClassifier(random_state=rand_st, n_jobs=-1)

clf_RF_p0 = [{}]





clf_RF_gs = RandomForestClassifier(random_state=rand_st, n_jobs=-1)

clf_RF_pg = [{

    'max_depth': np.logspace(0.3,4,num = 10 ,base=10,dtype='int'), #[1, 5, 13, 34, 87, 226, 584, 1505, 3880, 10000]

    'n_estimators' : np.logspace(0.1,3,num = 10 ,base=10,dtype='int'), #[1, 2, 5, 11, 24, 51, 107, 226, 476, 1000]

    'min_samples_split' : np.logspace(0.4, 1, num=5, base=10, dtype=int), #[2, 3, 5, 7, 10]

    'min_samples_leaf' : np.logspace(0.1,1,num = 4 ,base=9,dtype='int'), #[1, 2, 4, 9]

    'max_features' : ['auto', None]

              }]

    



clf_GrBoost_df = GradientBoostingClassifier(random_state=rand_st)

clf_GrBoost_p0 = [{}]



clf_GrBoost_gs = GradientBoostingClassifier(random_state=rand_st)

clf_GrBoost_pg = [{

    'max_depth': np.logspace(0.3,4,num = 10 ,base=10,dtype='int'), #[1, 5, 13, 34, 87, 226, 584, 1505, 3880, 10000]

    'n_estimators' : np.logspace(0.1,3,num = 10 ,base=10,dtype='int'), #[1, 2, 5, 11, 24, 51, 107, 226, 476, 1000]

    'min_samples_split' : np.logspace(0.4, 1, num=5, base=10, dtype=int), #[2, 3, 5, 7, 10]

    'min_samples_leaf' : np.logspace(0.1,1,num = 4 ,base=9,dtype='int'), #[1, 2, 4, 9]

    'max_features' : ['auto', None]

              }]







clf_models_gs = [clf_XGBoost_df, clf_XGBoost_gs, clf_ExTree_df, clf_ExTree_gs, clf_KER_SVC_df, clf_KER_SVC_gs, clf_RF_df, clf_RF_gs, clf_GrBoost_df, clf_GrBoost_gs  ]

clf_models_gs_name = ['clf_XGBoost_df', 'clf_XGBoost_gs', 'clf_ExTree_df', 'clf_ExTree_gs', 'clf_KER_SVC_df', 'clf_KER_SVC_gs', 'clf_RF_df', 'clf_RF_gs', 'clf_GrBoost_df', 'clf_GrBoost_gs'  ]

clf_params_gs = [clf_XGBoost_p0, clf_XGBoost_pg, clf_ExTree_p0, clf_ExTree_pg, clf_KER_SVC_p0, clf_KER_SVC_pg, clf_RF_p0, clf_RF_pg, clf_GrBoost_p0, clf_GrBoost_pg ]



gs_metric_cols = ['clf_name', 'Best_Score','Mean_Test_Score', 'Mean_Test_SD', 'Best_Estimator', 'Best_Params']

gs_metrics = pd.DataFrame(columns = gs_metric_cols)



# Define function to conduct extensive GridSearch and return valuable parameters / score in a dataframe

def clf_GridSearchCV_results(gs_metrics, X_train, y_train, GS_scoring, GS_CV):

    

    gs_metric_dict = []

    # iterate over classifiers and param grids 

    for clf_gs_name, clf_gs, params_gs in zip(clf_models_gs_name, clf_models_gs, clf_params_gs):

        clf_gs = EvolutionaryAlgorithmSearchCV(clf_gs,params = params_gs, cv=GS_CV, scoring=GS_scoring, 

                                               n_jobs= 4, 

                                               verbose = 1, 

                                               population_size=50,

                                               gene_mutation_prob=0.10,

                                               gene_crossover_prob=0.5,

                                               tournament_size=3,

                                               generations_number=5, )

        

        clf_gs.fit(X_train,y_train)

        

        clf_name = clf_gs

        Best_Score = clf_gs.best_score_

        #Mean_Train_Score = np.mean(clf_gs.cv_results_['mean_train_score']) #Not available with Evolutionary Search

        Mean_Test_Score = np.mean(clf_gs.cv_results_['mean_test_score'])

        Mean_Test_SD = np.mean(clf_gs.cv_results_['std_test_score'])

        Best_Estimator = clf_gs.best_estimator_

        Best_Params = clf_gs.best_params_

        

        gs_metric_values = [clf_gs_name, Best_Score, Mean_Test_Score, Mean_Test_SD, Best_Estimator, Best_Params]        

        gs_metric_dict.append(dict(zip(gs_metric_cols, gs_metric_values)))

        

    gs_metrics = gs_metrics.append(gs_metric_dict)

    return gs_metrics
# Run the EvolutionarySearch, Note: The name "GridSearch" is misleading.



#gs_metrics = clf_GridSearchCV_results(gs_metrics, X_train=X_fe_prepared[0:891], y_train=y_train, GS_scoring = "accuracy", GS_CV=5)

gs_metrics
Best_Estimator_RF = gs_metrics.iloc[7,4]

Best_Estimator_ExTree = gs_metrics.iloc[3,4]

Best_Estimator_GrBoost = gs_metrics.iloc[9,4]
plot_learning_curve(Best_Estimator_RF,"RandomForest learning curves",X_fe_prepared[0:891],y_train,cv=kfold)

plot_learning_curve(Best_Estimator_ExTree,"ExtraTrees learning curves",X_fe_prepared[0:891],y_train,cv=kfold)

plot_learning_curve(Best_Estimator_GrBoost,"GrBoost learning curves",X_fe_prepared[0:891],y_train,cv=kfold)
# num_attribs = ["Age", "SibSp", "Parch", "Fare", "NameLength", "Cabin_Number"]    

# cat_attribs_ordinal = ["Pclass", "Sex", "Embarked", "Cabin_Prefix", "Ticket_Letter"]

# cat_attribs_onehot = ["Title"]



attribs = num_attribs + cat_attribs_ordinal + cat_attribs_onehot



Imp_RF = np.asarray(sorted(list(zip(Best_Estimator_RF.feature_importances_, attribs )),reverse = True))

Imp_ExTree = np.asarray(sorted(list(zip(Best_Estimator_ExTree.feature_importances_, attribs )),reverse = True))

Imp_GrBoost = np.asarray(sorted(list(zip(Best_Estimator_GrBoost.feature_importances_, attribs )),reverse = True))
def impFeatPlot (ImpF, axis):

    val = ImpF[:,0].astype(np.float)

    att = ImpF[:,1]

    

    sns.barplot(val, att, palette = "viridis", ax = axis )

    



fig = plt.figure(figsize = (25,5))

fig.suptitle("Important Features", fontsize=16)

ax1 = fig.add_subplot(131)

ax1.set_title("RandomForest")

ax2 = fig.add_subplot(132)

ax2.set_title("ExTree")

ax3 = fig.add_subplot(133)

ax3.set_title("GrBoost")



impFeatPlot(Imp_RF, ax1)

impFeatPlot(Imp_ExTree, ax2)

impFeatPlot(Imp_GrBoost, ax3)
#Preparation same as Traindata



X_test["Title"] = X_test["Name"].str.extract(' ([A-Za-z]+\.)')



# Replace equal Title and create grouped Titles



X_test["Title"] = X_test["Title"].replace("Mlle.", "Miss.")

X_test["Title"] = X_test["Title"].replace("Ms.", "Miss.")

X_test["Title"] = X_test["Title"].replace("Mme.", "Mrs.")



X_test["Title"] = X_test["Title"].replace(["Col.", "Major.", "Sir.", "Lady.", "Don.", "Dona.", "Capt.", "Countess.", "Jonkheer."], "GT")



X_test['Ticket_Letter'] = X_test['Ticket'].apply(lambda x: str(x)[0] )



X_test["NameLength"] = X_test["Name"].str.len()



X_test["Cabin"].replace(np.nan, "N0.0", inplace=True)



# Extract Prefix



X_test["Cabin_Prefix"] = X_test["Cabin"].str.extract('(^.)')



# Extract Cabin Numbers



X_test["Cabin_Number"] = X_test["Cabin"].str.extract('(\d+)')



X_test.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis = 1, inplace = True)
X_test.head()
# Defining the Voting Classifier out of Top 3 Estimators





X_train, X_val, y_train, y_val = train_test_split(X_fe_prepared[0:891], y_train, test_size=0.2, random_state=rand_st)



voting_clf = VotingClassifier(

    estimators=[('RF', Best_Estimator_RF), ('ExTree', Best_Estimator_ExTree), ('GrBoost', Best_Estimator_GrBoost)],

    voting = 'hard', n_jobs = -1)





for clf in (Best_Estimator_RF, Best_Estimator_ExTree, Best_Estimator_GrBoost, voting_clf):

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    print(clf.__class__.__name__, accuracy_score(y_val, y_pred))


