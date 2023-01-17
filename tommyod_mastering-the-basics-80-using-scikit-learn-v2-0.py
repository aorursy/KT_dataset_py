# Python library imports
import numpy as np # All numerical libraries in Python built on NumPy
import pandas as pd # Pandas provides DataFrames for data wrangling
import matplotlib.pyplot as plt # The de facto plotting library in Python
import itertools # For iterations
import string # For strings
import re # For regular expression (regex)
import os # Operating system functions

plt.style.use('Solarize_Light2') # Set a non-default visual aesthetic for plots
%matplotlib inline
pd.set_option("display.max_columns", 2**10) # View more columns than pandas default
# Import the train and test data as pandas DataFrame instances
date_cols = ['project_submitted_datetime'] # Convert to datetime format automatically
train = pd.read_csv(os.path.join(r'../input', 'train.csv'), low_memory=False, parse_dates=date_cols)
test = pd.read_csv(os.path.join(r'../input', 'test.csv'), low_memory=False, parse_dates=date_cols)
# Keep track of the original data source by adding a 'source' column
train['source'] = 'train'
test['source'] = 'test'
test_train = pd.concat((test, train))
print(f'The shape of the data is {test_train.shape}.') # Showing off Python f-strings
# Checking for missing values
print("="*60)
print("Detecting NaN values in data:")
print("="*60)
print(test_train.isnull().sum(axis=0)[test_train.isnull().sum(axis=0) > 0])
most_common = test_train.teacher_prefix.value_counts().idxmax() # Compute argument maximizing value counts
test_train.teacher_prefix = test_train.teacher_prefix.fillna(most_common)
numerical_cols = []
dummy_categorical_cols = []
text_cols = []
# Find the rows where essays 3 and 4 are not null
mask_four_essays = ~(test_train.project_essay_3.isnull() & test_train.project_essay_4.isnull())

# Assign them to columns 1 and 2 by concatenation
test_train[mask_four_essays] = (test_train[mask_four_essays]
                 .assign(project_essay_1 = lambda df: df.project_essay_1 + df.project_essay_2)
                 .assign(project_essay_2 = lambda df: df.project_essay_3 + df.project_essay_4))

# Drop columns related to essay 3 and 4
test_train = test_train.drop(columns=['project_essay_3', 'project_essay_4'])
# Load resources
resources = pd.read_csv(os.path.join(r'../input', 'resources.csv'), low_memory=False)
print(f'The shape of the data is {resources.shape}.')
# Checking for missing values
print("="*30)
print("Detecting NaN values in data:")
print("="*30)
print(resources.isnull().sum(axis=0)[resources.isnull().sum(axis=0) > 0])
# Fill NAs
resources = resources.fillna('X')
# Previewing data
resources.head(3)
def concatenate(series, sep=' '):
    return sep.join(series) # Preferred to lambda, since this function has a name

# Create a lot of possible numerical features, each starting with 'p_'
resource_stats = (resources
.assign(p_desc_len = lambda df: df.description.str.len()) # Length of description text
.assign(p_total_price = lambda df: df.quantity * df.price) # Total price per item
.groupby('id') # Grouping by teacher ID and aggregating the following columns (note we pass function as a list): 
.agg({'description': [pd.Series.nunique, concatenate], # Number of unique items asked for
'quantity': [np.sum], # Total number of items asked for
'price': [np.sum, np.mean], # Prices per item added and averaged (quantity not included)
'p_desc_len': [np.mean, np.min, np.max], # Average description length
'p_total_price': [np.mean, np.min, np.max]})
)
# Checking for missing values
print("="*30)
print("Detecting NaN values in data:")
print("="*30)
print(resource_stats.isnull().sum(axis=0)[resource_stats.isnull().sum(axis=0) > 0])
# Collaps to flat index
resource_stats.columns = ['_'.join([col, func]) for col, func in resource_stats.columns.values]
numerical_cols += list(resource_stats.columns.values)
# Merge the resources statistics into the test and train sets
test_train = test_train.merge(resource_stats, how='left', left_on='id', right_index=True)
test_train.sample(1).T.tail(6)
# Get the month from the datetime, convert it to a string, and add it as a new column
test_train['month'] = test_train.project_submitted_datetime.dt.month.apply(str)

# Does submitting during the morning hours help?
test_train['daytime'] = pd.Series(np.where( 
    ((7 <= test_train.project_submitted_datetime.dt.hour) & 
     (test_train.project_submitted_datetime.dt.hour <= 10)), 1, 0)).apply(str)
# Simple dummy variables, i.e. every entry has one value
dummy_colnames = ['teacher_prefix', 'month', 'school_state', 'daytime']
dummies = pd.get_dummies(test_train.loc[:, dummy_colnames])
dummy_categorical_cols += dummies.columns.tolist()

# Concatenate along the columns
test_train = pd.concat((test_train, dummies), axis=1)
# The following libray is for representing (printing) long strings and lists
import reprlib

def set_of_categories(col_name):
    """Retrieve a set of category names from a column"""
    list_train = test_train[col_name].tolist()
    list_test = test_train[col_name].tolist()
    return set(', '.join(list_train + list_test).split(', '))

unique_categories = set_of_categories('project_subject_categories')
unique_subcategories = set_of_categories('project_subject_subcategories')
unique_cats_total = list(unique_categories.union(unique_subcategories))
dummy_categorical_cols += unique_cats_total
print('Categories:', reprlib.repr(unique_cats_total))
project_cat_colnames = ['project_subject_categories', 'project_subject_subcategories']

df_cats = test_train.loc[:, project_cat_colnames]

# Create a new column for each category: put 1 if it's mentioned, 0 if not
for category in unique_categories:
    df_cats[category] = np.where(df_cats.project_subject_categories.str.contains(category), 1, 0)
for category in unique_subcategories:
    df_cats[category] = np.where(df_cats.project_subject_subcategories.str.contains(category), 1, 0)
    
df_cats = df_cats.drop(columns=project_cat_colnames)
df_cats.head(1).T.head(5)
test_train = pd.concat((test_train, df_cats), axis=1)
print(f'The dataset now has ~{len(test_train.columns)} features.')
test_train.head(1)
# Extracting text columns
text_cols += ['project_essay_1', 'project_essay_2', 'project_resource_summary', 'description_concatenate']
from textblob import TextBlob

df_subset = test_train[test_train.source == 'train']

# Create a plot
plt.figure(figsize=(14, 10))

plt_num = 1
for column in text_cols:
    
    # Get data corresponding to the columns which we will use
    data = df_subset.loc[:, ('project_is_approved', column)]
        
    for feature in ['polarity', 'subjectivity']:
    
        # Create a new subplot, set the title
        plt.subplot(4, 2, plt_num)
        plt.title(f'{column} - {feature.capitalize()}', fontsize=12)
        plt_num += 1

        # Function to get features from text using TextBlob
        feature_from_txt = lambda x : getattr(TextBlob(x).sentiment, feature)

        # Sample some data, apply the feature extraction function
        approved_mask = (data.project_is_approved == 1)
        approved = data[approved_mask].sample(1000).assign(feat=lambda df: df[column].apply(feature_from_txt))
        not_approved = data[~approved_mask].sample(1000).assign(feat=lambda df: df[column].apply(feature_from_txt))
        
        # Plot the subplot
        bandwidth = 0.225
        ax = approved.feat.plot.kde(bw_method=bandwidth, label='Approved')
        ax = not_approved.feat.plot.kde(ax=ax, bw_method=bandwidth, label='Not approved')
        plt.xlim([-.5, 1])
        plt.legend(loc='best')
        
# Show the full figure
plt.tight_layout()
plt.show()
%%time

subj = lambda x: TextBlob(x).sentiment.subjectivity
polar = lambda x: TextBlob(x).sentiment.polarity

test_train['description_subjectivity'] = test_train['description_concatenate'].apply(subj)
test_train['description_polarity'] = test_train['description_concatenate'].apply(polar)

test_train['project_resource_summary_subjectivity'] = test_train['project_resource_summary'].apply(subj)
test_train['project_resource_summary_polarity'] = test_train['project_resource_summary'].apply(polar)

test_train['project_essay_2_polarity'] = test_train['project_essay_2'].apply(polar)
test_train['project_essay_1_polarity'] = test_train['project_essay_1'].apply(polar)
text_cols += ['project_title']
print(text_cols)
%%time

# This code is a bit slow: approximately ~50 seconds on my computer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

for text_col in text_cols: # Title, essays, summary, title
    
    # Get a sparse SciPy matrix of words counts
    X = vectorizer.fit_transform(test_train[text_col].values)
    
    col_new_name = text_col.replace('project_', '')
    
    # Compute some basic statistics and add to dataset
    unique_words = (X > 0).sum(axis=1) # Sum of words appearing more than zero times
    num_words = (X).sum(axis=1) # Sum of occurences of words
    test_train[col_new_name + '_unique_words'] = unique_words
    test_train[col_new_name + '_num_words'] = num_words
    test_train[col_new_name + '_vocab'] = np.exp(unique_words / (num_words + 10e-10))
from csv import QUOTE_ALL

# We're going to quote the fields using ", so remove it from the text just in case
for text_col in text_cols:
    test_train[text_col] = test_train[text_col].str.replace('"', ' ')
   
# Save the data
# test_train.to_csv('preprocessed_test_train.csv', quoting=QUOTE_ALL)
# Load the data from disk
# test_train = pd.read_csv('preprocessed_test_train.csv', index_col=0, quoting=QUOTE_ALL)
numeric_cols = [
 'teacher_number_of_previously_posted_projects',
 'description_nunique',
 'quantity_sum',
 'price_sum',
 'price_mean',
 'p_desc_len_mean',
 'p_desc_len_amin',
 'p_desc_len_amax',
 'p_total_price_mean',
 'p_total_price_amin',
 'p_total_price_amax',
 'project_resource_summary_subjectivity',
 'project_resource_summary_polarity',
 'project_essay_2_polarity',
 'project_essay_1_polarity',
 'essay_1_unique_words',
 'essay_1_num_words',
 'essay_1_vocab',
 'essay_2_unique_words',
 'essay_2_num_words',
 'essay_2_vocab',
 'resource_summary_unique_words',
 'resource_summary_num_words',
 'resource_summary_vocab',
 'description_concatenate_unique_words',
 'description_concatenate_num_words',
 'description_concatenate_vocab',
 'title_unique_words',
 'title_num_words',
 'title_vocab']
# Imports for the pipeline
from tempfile import mkdtemp
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.sparse import coo_matrix, hstack
from sklearn.preprocessing import RobustScaler
class ColSplitter(BaseEstimator, TransformerMixin):
    """Estimator to split the columns of a pandas dataframe."""
    
    def __init__(self, cols, ravel=False):
        self.cols = cols
        self.ravel = ravel # If it's a text features, we ravel for TF-IDF

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return (x.loc[:, self.cols].values.ravel() if self.ravel
                else x.loc[:, self.cols].values)
    
class LogTransform(BaseEstimator, TransformerMixin):
    """Take linear combination of f(x) = x and g(x) = ln(x)"""
    
    def __init__(self, alpha):
        """alpha = 1 -> log
        alpha = 0 -> linear"""
        self.alpha = alpha
        
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return self.alpha * np.log(np.log(x + 2.001) + 1.001) + (1 - self.alpha) * x
    
class ParallelPipe(BaseEstimator, TransformerMixin):
    """Put similar pipes in parallel. This has two effects:
    (1) When transforming, the final result is concatenated (Feature Union)
    (2) Setting hyperparameters on this pipe will set it on every sub-pipe."""
    
    def __init__(self, pipes, *args, **kwargs):
        self.pipes = pipes
        
    def fit(self, x, y=None):
        [p.fit(x) for p in self.pipes]
        return self

    def transform(self, x):
        try:
            return hstack([p.transform(x) for p in self.pipes])
        except:
            return np.concatenate(tuple([p.transform(x) for p in self.pipes]), axis=1)
    
    def _get_param_names(self, *args, **kwargs):
        return ['pipes']
    
    def set_params(self, *args, **kwargs):
        [p.set_params(*args, **kwargs) for p in self.pipes]
        return None
    
# --------------------------------------------------
# ----- (1) Numerical pipeline ---------------------
# --------------------------------------------------
num_colsplitter = ColSplitter(cols=numeric_cols)
logtransform = LogTransform(alpha=1)
scaler = RobustScaler(quantile_range=(5.0, 95.0))
numerical_pipe = Pipeline([('num_colsplitter', num_colsplitter), 
                           ('logtransform', logtransform),
                           ('scaler', scaler)])

# --------------------------------------------------
# ----- (2) Categorical pipeline -------------------
# --------------------------------------------------
dummy_colsplitter = ColSplitter(cols=dummy_categorical_cols)
categorical_pipe = Pipeline([('dummy_colsplitter', dummy_colsplitter)])

# --------------------------------------------------
# ----- (3) Text pipeline --------------------------
# --------------------------------------------------
text_subpipes = []
for text_col in text_cols:
    text_colsplitter = ColSplitter(cols=text_col, ravel=True)
    
    tf_idf = TfidfVectorizer(sublinear_tf=False, norm='l2', stop_words=None, 
                             ngram_range=(1, 1), max_features=None)
    
    text_col_pipe = Pipeline([('text_colsplitter', text_colsplitter),
                     ('tf_idf', tf_idf)])
    text_subpipes.append(text_col_pipe)

text_pipe = ParallelPipe(text_subpipes)

# --------------------------------------------------
# ----- (4) Final pipeline - Logistic regression ---
# --------------------------------------------------
#cachedir = mkdtemp() # Creates a temporary directory
estimator = LogisticRegression(penalty="l2", C=0.21428)

pipeline_logreg = Pipeline([('union', FeatureUnion(transformer_list=[
    ('numerical_pipe', numerical_pipe),
    ('categorical_pipe', categorical_pipe),
    ('text_pipe', text_pipe)
])), 
                     ('estimator', estimator)])
# Testing that all numeric values are finite
assert np.all(np.isfinite(test_train[numeric_cols].values))
from sklearn.model_selection import GridSearchCV

# Dictionary with parameters names to try during search
# We tried a lot of parameters, you may uncomment the code an experiment
param_grid = {"estimator__C": np.linspace(0.24285-0.1, 0.24285+0.1, num=6)
             # "union__numerical_pipe__logtransform__alpha": [0.8, 1],
             # "union__text_pipe__tf_idf__stop_words": [None, 'english']
             }

# run randomized search
grid_search = GridSearchCV(pipeline_logreg, param_grid=param_grid,
                                    scoring='roc_auc',
                                    n_jobs=1,
                                    verbose=1,
                                    cv=3)
# Grab 2 subsets of the data
n = 25000
subset_A = test_train.loc[lambda df: (df.project_is_approved == 1)].sample(n)
subset_B = test_train.loc[lambda df: (df.project_is_approved == 0)].sample(n)
test_train_subset = pd.concat((subset_A, subset_B))
test_X = test_train_subset
test_y = test_X.project_is_approved
import warnings
perform_grid_search = True

if perform_grid_search:
    with warnings.catch_warnings():
        # UserWarning: Persisting input arguments took 0.70s to run.
        warnings.simplefilter("ignore", category=UserWarning)
        grid_search.fit(test_X, test_y.values.ravel())
    best_estimator = grid_search.best_estimator_
    print('Best score:', grid_search.best_score_)
    print('Best params:\n', grid_search.best_params_)
else:
    # If we do not run grid search, we use the pipeline as initialized
    best_estimator = pipeline_logreg
# Grab 100% of the train data, run fitting using the best estimator
full_train = test_train.loc[lambda df: df.source == 'train']
y_pred = best_estimator.fit(full_train, full_train.project_is_approved.ravel())
# Predict on test data
test = test_train[test_train.source == 'test']
test.loc[:, ('project_is_approved', )] = best_estimator.predict_proba(test)[:, 1]
test[['id','project_is_approved']].shape
test[['id','project_is_approved']].to_csv('submission.csv', index=False)