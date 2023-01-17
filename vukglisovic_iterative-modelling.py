import os

import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.feature_selection import RFE

from sklearn import metrics

import eli5

from tqdm import tqdm

from copy import deepcopy

from IPython.display import display

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn-whitegrid')



np.random.seed(1234)



# Any results you write to the current directory are saved as output.

print(os.listdir("../input"))
class PipelineWithCoef(Pipeline):

    """This class only adds the coef_ attribute to the original Pipeline

    class. This allows us to use Pipelines in RFE.

    """

    def __init__(self, steps, memory=None):

        super(PipelineWithCoef, self).__init__(steps, memory)



    def fit(self, X, y=None, **fit_params):

        """Calls last elements .coef_ method. Based on the sourcecode for decision_function(X).

        Link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pipeline.py

        """

        super(PipelineWithCoef, self).fit(X, y, **fit_params)

        self.coef_ = self.steps[-1][1].coef_

        return self
target_column = 'target'

converters = {target_column: lambda v: int(float(v))}  # the file contains string values 1.0 and 0.0, but int(1.0) and int(0.0) raise errors

train_df = pd.read_csv('../input/train.csv', index_col=0, converters=converters)

feature_columns = train_df.columns.drop(target_column)

print(train_df.shape)

train_df.head()



train_df.target.value_counts()
(train_df.isnull().sum() > 0).any()
fig, ax = plt.subplots(figsize=(19,14))

corr = train_df.corr()

sns.heatmap(corr, ax=ax);
target_correlations = corr[target_column].drop(target_column, axis=0)

top_target_correlations = target_correlations.abs().sort_values(ascending=False)[:50]

top_corr_feature_columns = list(top_target_correlations.index)

top_target_correlations[:20]
#Scaling Numerical columns

std = StandardScaler()

X_scaled = std.fit_transform(train_df[feature_columns])

X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
lr = LogisticRegression(solver='liblinear', random_state=42)

param_grid = {'class_weight' : ['balanced', None],

              'penalty' : ['l2','l1'],

              'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid = GridSearchCV(estimator=lr, param_grid=param_grid, iid=False, cv=3, scoring='roc_auc', n_jobs=-1)

grid.fit(X_scaled[feature_columns], train_df[target_column])

print("Grid best score was: {}".format(grid.best_score_))



# select the best parameters from the grid search above

lr = LogisticRegression(solver='liblinear', **grid.best_params_)

lr.fit(X_scaled[feature_columns], train_df[target_column])



rfe_lr = RFE(lr, 25, step=1)

rfe_lr.fit(X_scaled, train_df[target_column])
test_df = pd.read_csv('../input/test.csv', index_col=0)

X_test_scaled = std.transform(test_df[feature_columns])

X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)



submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = rfe_lr.predict_proba(X_test_scaled)

submission.to_csv('rfe_lr_submission.csv', index=False)
def test_model(model, parameter_grid, train_feature_values, train_target_values, retrain_with_top_features=True, importance_threshold=0.0005):

    """Function for training and testing a model. This method accepts a parameter grid

    with which you can do hyperparameter tuning. It is also possible to retrain the model

    by automatically selecting the top features based on eli5.sklearn.PermutationImportance.



    Args:

        model (sklearn.pipeline.Pipeline): a scikit learn Pipeline

        parameter_grid (dict): hyperparameter tuning specifications for the grid search

        train_feature_values (pd.DataFrame): training features values

        train_target_values (pd.Series): training target values

        retrain_with_top_features (bool): if True, then eli5.sklearn.PermutationImportance

            will be used to determine what features should be used to retrain the model.

            Otherwise, the feature importances will still be calculated, but no new model

            will be trained.

        importance_threshold (float): importance threshold that must be reached from the

            PermutationImportance methodology before a feature is included in the

            retraining procedure.



    Return:

        sklearn.model_selection._search.GridSearchCV: a fitted model with the best

            hyperparameters chosen

    """

    metric = metrics.make_scorer(metrics.roc_auc_score, greater_is_better=True, needs_proba=True)



    def cv_train_model(model, parameter_grid, train_feature_values, train_target_values):

        """Receives an unfitted model as a parameter and does a GridSearch with the

        training data.

        

        Args:

            model (sklearn.pipeline.Pipeline):

            parameter_grid (dict):

            train_feature_values (pd.DataFrame):

            train_target_values (pd.Series):

            

        Returns:

            sklearn.model_selection._search.GridSearchCV

        """

        # cv: for integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.

        folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1234)  # repeats n_splits splits n_repeats times

        folds_iterable = folds.split(train_feature_values, train_target_values)

        grid_search = GridSearchCV(model, param_grid=parameter_grid, scoring=metric, iid=False, refit=True, cv=folds_iterable, return_train_score=True, n_jobs=-1)

        grid_search_model = grid_search.fit(train_feature_values, train_target_values)



        cv_results = pd.DataFrame.from_dict(grid_search_model.cv_results_)

        cv_results.set_index('rank_test_score', drop=True, inplace=True)

        cv_results.sort_index(inplace=True)

        cv_result_columns = [column for column in cv_results.columns if column.startswith('param_')] + ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']

        display(cv_results[cv_result_columns].head(5))

        return grid_search_model



    print("Training first model...")

    grid_search_model = cv_train_model(model, parameter_grid, train_feature_values, train_target_values)



    print("Checking feature importances...")

    perm = eli5.sklearn.PermutationImportance(grid_search_model, scoring=metric, n_iter=10, random_state=1234).fit(train_feature_values, train_target_values)

    display(eli5.show_weights(perm, top=25))

    features_importance_df = eli5.formatters.as_dataframe.explain_weights_df(perm)

    top_feature_columns = [train_feature_values.columns[int(feature_column[1:])]

                           for feature_column

                           in features_importance_df[features_importance_df.weight > importance_threshold].feature]

    if len(top_feature_columns) == 0:

        print("No top feature columns found. Exiting function.")

        return grid_search_model, list(train_feature_values.columns)

    top_feature_columns = top_feature_columns[:50]  # never take more than 50 top feature columns



    if retrain_with_top_features and len(top_feature_columns) < len(train_feature_values.columns):

        print("Retraining with top {} features.".format(len(top_feature_columns)))

        grid_search_model = cv_train_model(model, parameter_grid, train_feature_values[top_feature_columns], train_target_values)

    elif retrain_with_top_features:

        print("All features are important. Therefore not retraining.")

        top_feature_columns = list(train_feature_values.columns)

    else:

        print("Not retraining with top features.")

        top_feature_columns = list(train_feature_values.columns)



    return grid_search_model, top_feature_columns
# Create a dictionary to keep track of all scores

all_scores = dict()
sc_lr = Pipeline(steps=[('sc', StandardScaler(with_mean=True, with_std=True)),

                        ('logistic_regression', LogisticRegression(solver='liblinear', max_iter=100, random_state=42))])

param_grid = {'logistic_regression__class_weight': ['balanced', None],

              'logistic_regression__penalty': ['l1', 'l2'],

              'logistic_regression__C': [0.01, 0.1, 0.5, 1., 1.5, 10.]}



sc_lr_model, sc_lr_feature_columns = test_model(sc_lr, param_grid, train_df[feature_columns], train_df[target_column])

all_scores['sc_lr'] = sc_lr_model.best_score_
sc_pca_lr = Pipeline(steps=[('sc', StandardScaler(with_mean=True, with_std=True)),

                            ('pca', PCA()),

                            ('logistic_regression', LogisticRegression(solver='liblinear', max_iter=1000))])

param_grid = {'pca__n_components': [5, 10, 15],

              'logistic_regression__class_weight': ['balanced', None],

              'logistic_regression__penalty': ['l1', 'l2'],

              'logistic_regression__C': [0.1, 1., 10.]}



sc_pca_lr_model, sc_pca_lr_feature_columns = test_model(sc_pca_lr, param_grid, train_df[feature_columns], train_df[target_column], importance_threshold=0.001)

all_scores['sc_pca_lr'] = sc_pca_lr_model.best_score_
sc_lda_lr = Pipeline(steps=[('sc', StandardScaler(with_mean=True, with_std=True)),

                            ('lda', LinearDiscriminantAnalysis()),

                            ('logistic_regression', LogisticRegression(solver='liblinear', max_iter=1000))])

param_grid = {'lda__n_components': [5, 10, 15],

              'logistic_regression__class_weight': ['balanced', None],

              'logistic_regression__penalty': ['l1', 'l2'],

              'logistic_regression__C': [0.1, 1., 10.]}



sc_lda_lr_model, sc_lda_lr_feature_columns = test_model(sc_lda_lr, param_grid, train_df[feature_columns], train_df[target_column])

all_scores['sc_lda_lr'] = sc_lda_lr_model.best_score_
svc = Pipeline(steps=[('svc', SVC(kernel='linear', gamma='auto', probability=True))])

param_grid = {'svc__C': [0.01, 0.1, 0.5, 1., 1.5, 10., 100.],

              'svc__class_weight': ['balanced', None]}



svc_model, svc_feature_columns = test_model(svc, param_grid, train_df[top_corr_feature_columns], train_df[target_column])

all_scores['svc'] = svc_model.best_score_
def test_model_with_rfe(best_model, train_feature_values, train_target_values, min_nr_features=10, max_nr_features=40):

    """Creates a model using recursive feature elimination (RFE) with the best parameters

    found with GridSearchCV. This method tries out all variants with the number of features

    to select between min_nr_features and max_nr_features (both inclusive) for RFE.

    

    Args:

        best_model (sklearn.model_selection.GridSearchCV):

        train_feature_values (pd.DataFrame):

        train_target_values (pd.Series):

        min_nr_features (int):

        max_nr_features (int):

        

    Return:

        RFE

    """

    deepcopied_best_model = deepcopy(best_model.best_estimator_)

    nr_feaures_to_select_range = list(range(min_nr_features, max_nr_features + 1))

    all_scores = []

    all_estimators = []

    for n_features_to_select in tqdm(nr_feaures_to_select_range):

        # recreate the original Pipeline

        best_pipeline = PipelineWithCoef(steps=deepcopied_best_model.steps)

        best_pipeline.fit(train_feature_values, train_target_values)

        estimator = RFE(best_pipeline, n_features_to_select=n_features_to_select)

        # train the model

        estimator.fit(train_feature_values, train_target_values)

        folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1234)  # repeats n_splits splits n_repeats times

        folds_iterable = folds.split(train_feature_values, train_target_values)

        scores = cross_val_score(estimator, train_feature_values, train_target_values, scoring='roc_auc', cv=folds_iterable)

        all_scores.append(scores.mean())

        all_estimators.append(estimator)

        

    def plot_roc_auc_scores(nr_features_list, scores_list):

        """Plots the ROC-AUC scores obtained from the number of features to select

        for RFE.

        

        Args:

            nr_features_list (list):

            scores_list (list):

        """

        fig, ax = plt.subplots(figsize=(20, 4))

        ax.plot(nr_features_list, scores_list)

        for s in ax.spines:

            ax.spines[s].set_visible(False)

        plt.xticks(fontsize=12)

        plt.yticks(fontsize=12)

        ax.set_title('ROC-AUC Score', fontsize=16)

        ax.set_xlabel('Nr Features Selected', fontsize=14);

    

    plot_roc_auc_scores(nr_feaures_to_select_range, all_scores)

    

    top_index = np.argmax(all_scores)

    print("Number of features selected for best score: {}".format(nr_feaures_to_select_range[top_index]))

    best_estimator = all_estimators[top_index]

    

    return best_estimator
rfe_sc_lr_model = test_model_with_rfe(sc_lr_model, train_df[feature_columns], train_df[target_column], max_nr_features=30)
test_df = pd.read_csv('../input/test.csv', index_col=0)

print(test_df.shape)

test_df.head()
def create_submission_file(model, test_data, prefix_output_file):

    submission_df = pd.DataFrame(data={'id': test_data.index,

                                       'target': model.predict_proba(test_data)[:,1]})

    submission_df.to_csv(prefix_output_file + '_submission.csv', index=False)
all_scores
# select a few models from first modelling section to create outputs

create_submission_file(sc_lr_model, test_df[sc_lr_feature_columns], 'sc_lr')

create_submission_file(svc_model, test_df[svc_feature_columns], 'svc')
# select the RFE model from recursive feature elimination to create outputs

create_submission_file(rfe_sc_lr_model, test_df[feature_columns], 'rfe_sc_lr')