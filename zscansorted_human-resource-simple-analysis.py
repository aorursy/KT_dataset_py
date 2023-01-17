# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
_df = pd.read_csv('../input/HR_comma_sep.csv')
_df.describe()
print(_df.columns)
# create encoding by using the index of sorted unique value
sales_uniq = _df.sales.unique()
sales_uniq.sort()
print(sales_uniq)

# create encoding by using the index of sorted unique value
salary_uniq = _df.salary.unique()
salary_uniq.sort()
print(salary_uniq)

# encode categorical variable into number
_df.sales_code = _df.sales.map(lambda sls: np.where(sales_uniq == sls)[0][0])
_df.salary_code = _df.salary.map(lambda sal: np.where(salary_uniq == sal)[0][0])
# describe the dataset for who left the company
_df_left = _df[_df.left==1]
_df_left.describe()
# describe the dataset for who don't left the company
_df_not_left = _df[_df.left!=1]
_df_not_left.describe()
# Create function to Explore The Data Easily
def _summary_plots(_df, _div, nrows, ncols, plot_func):
    columns = [x for x in _df.columns if x not in ['left', 'salary', 'sales']]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9,10))
    fig.suptitle('DIVISION: {}'.format(_div))
    for col_idx in range(len(columns)):
        plot_func(data=_df, x='left', y=columns[col_idx], ax=axs[col_idx/nrows][col_idx%ncols])
_df.sales.unique()
division = _df.sales.unique()
for _div in _df.sales.unique():
    _summary_plots(_df[_df.sales==_div], _div, 3, 3, sns.boxplot)
# plotting  the 
from numpy import count_nonzero
sns.barplot(
    data=_df,
    x='sales',
    y='promotion_last_5years',
    estimator=count_nonzero,
    hue='left'
)
# plotting the correlations of variables

df_corr = _df.corr()

sns.heatmap(df_corr, vmax=.8, square=True)
# see the summary by removing the bias variables 
sns.set()
sns.pairplot(
    _df, 
    hue='left'
)
# splitting train (80%) and test set (20%)
msk = np.random.rand(len(_df)) < 0.8
_df_train = _df[msk]
_df_test = _df[~msk]
# helper functions to quickly evaluate the model
def train_model(model, df_x, df_y):
    return model.fit(
        df_x,
        df_y
    )

def get_y_predicted(model, df_x, df_y, test_x):
    res_model = train_model(
        model, 
        df_x, 
        df_y
    )
    return res_model.predict(
        test_x
    )

def get_model_avg_accuracy(model, df_x, df_y, evaldf_x, evaldf_y):
    res_model = train_model(
        model, 
        df_x, 
        df_y
    )
    return res_model.score(
        evaldf_x,
        evaldf_y
    )
# exclude class column and non number columns
logreg = linear_model.LogisticRegression()
excluded_columns = ['left', 'sales', 'salary']
attributes = [
    x for x in _df_train.columns
    if x not in excluded_columns
]
to_train_score = get_model_avg_accuracy(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_train[attributes],
    _df_train[['left']]
)

to_test_score = get_model_avg_accuracy(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes],
    _df_test[['left']]
)
df_coef = pd.DataFrame(
    list(zip(attributes, logreg.coef_[0])),
    columns=['attribute_name', 'coef']
)
print('feeding to train score: {}'.format(to_train_score))
print('feeding to test score: {}'.format(to_test_score))
print()
print(df_coef)
# exclude some columns
logreg = linear_model.LogisticRegression()
excluded_columns = [
    'left', 'sales', 'salary', 'number_project', 'Work_accident'
]
attributes = [
    x for x in _df_train.columns
    if x not in excluded_columns
]
to_train_score = get_model_avg_accuracy(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_train[attributes],
    _df_train[['left']]
)

to_test_score = get_model_avg_accuracy(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes],
    _df_test[['left']]
)

df_coef = pd.DataFrame(
    list(zip(attributes, logreg.coef_[0])),
    columns=['attribute_name', 'coef']
)

print('feeding to train score: {}'.format(to_train_score))
print('feeding to test score: {}'.format(to_test_score))
print()
print(df_coef)
# include only satisfaction_level 

logreg = linear_model.LogisticRegression()
attributes = ['satisfaction_level']
to_train_score = get_model_avg_accuracy(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_train[attributes],
    _df_train[['left']]
)

to_test_score = get_model_avg_accuracy(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes],
    _df_test[['left']]
)

df_coef = pd.DataFrame(
    list(zip(attributes, logreg.coef_[0])),
    columns=['attribute_name', 'coef']
)
print('feeding to train score: {}'.format(to_train_score))
print('feeding to test score: {}'.format(to_test_score))
print()
print(df_coef)
# check additional metrics while excluding 'number_project' and 'work_accident'
logreg = linear_model.LogisticRegression()
excluded_columns = [
    'left', 'sales', 'salary', 'number_project', 'Work_accident'
]
attributes = [
    x for x in _df_train.columns
    if x not in excluded_columns
]
y_predicted = get_y_predicted(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes]
)

metrics_col = ['precision', 'recall', 'fbeta_score', 'support']

metrics_val = precision_recall_fscore_support(
    _df_test[['left']],
    y_predicted,
    average='macro'
)

_metrics_df = pd.DataFrame(
    list(zip(metrics_col, metrics_val)),
    columns=['metrics_name', 'metrics_value']
)

print(_metrics_df)
# check additional metrics for 'satisfaction_level' only attribute
logreg = linear_model.LogisticRegression()
attributes = ['satisfaction_level']
y_predicted = get_y_predicted(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes]
)

metrics_col = ['precision', 'recall', 'fbeta_score', 'support']

metrics_val = precision_recall_fscore_support(
    _df_test[['left']],
    y_predicted,
    average='macro'
)

_metrics_df = pd.DataFrame(
    list(zip(metrics_col, metrics_val)),
    columns=['metrics_name', 'metrics_value']
)

print(_metrics_df)
# exclude class column and non number columns
logreg = RandomForestClassifier()
excluded_columns = ['left', 'sales', 'salary']
attributes = [
    x for x in _df_train.columns
    if x not in excluded_columns
]
to_train_score = get_model_avg_accuracy(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_train[attributes],
    _df_train[['left']]
)

to_test_score = get_model_avg_accuracy(
    logreg,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes],
    _df_test[['left']]
)
df_coef = pd.DataFrame(
    list(zip(attributes, model.feature_importances_)),
    columns=['attribute_name', 'feature_importance']
)
print('feeding to train score: {}'.format(to_train_score))
print('feeding to test score: {}'.format(to_test_score))
print()
print(df_coef)
# exclude some columns
model = RandomForestClassifier()
excluded_columns = [
    'left', 'sales', 'salary', 'number_project', 'Work_accident'
]
attributes = [
    x for x in _df_train.columns
    if x not in excluded_columns
]
to_train_score = get_model_avg_accuracy(
    model,
    _df_train[attributes],
    _df_train[['left']],
    _df_train[attributes],
    _df_train[['left']]
)

to_test_score = get_model_avg_accuracy(
    model,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes],
    _df_test[['left']]
)

df_coef = pd.DataFrame(
    list(zip(attributes, model.feature_importances_)),
    columns=['attribute_name', 'feature_importance']
)
print('feeding to train score: {}'.format(to_train_score))
print('feeding to test score: {}'.format(to_test_score))
print()
print(df_coef)
# include only satisfaction_level 

model = RandomForestClassifier()
attributes = ['satisfaction_level']
to_train_score = get_model_avg_accuracy(
    model,
    _df_train[attributes],
    _df_train[['left']],
    _df_train[attributes],
    _df_train[['left']]
)

to_test_score = get_model_avg_accuracy(
    model,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes],
    _df_test[['left']]
)

print('feeding to train score: {}'.format(to_train_score))
print('feeding to test score: {}'.format(to_test_score))
model = RandomForestClassifier()

excluded_columns = [
    'left', 'sales', 'salary', 'number_project', 'Work_accident'
]
attributes = [
    x for x in _df_train.columns
    if x not in excluded_columns
]

y_predicted = get_y_predicted(
    model,
    _df_train[attributes],
    _df_train[['left']],
    _df_test[attributes]
)

metrics_col = ['precision', 'recall', 'fbeta_score', 'support']

metrics_val = precision_recall_fscore_support(
    _df_test[['left']],
    y_predicted,
    average='macro'
)

_metrics_df = pd.DataFrame(
    list(zip(metrics_col, metrics_val)),
    columns=['metrics_name', 'metrics_value']
)

print(_metrics_df)