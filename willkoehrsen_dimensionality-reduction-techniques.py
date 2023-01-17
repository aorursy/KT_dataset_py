import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
# Clearing up memory
import gc

# Featuretools for automated feature engineering
import featuretools as ft
import featuretools.variable_types as vtypes

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer

# Modeling
import lightgbm as lgb

# Evaluation of the model
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
feature_matrix = pd.read_csv('../input/costa-rican-poverty-derived-data/ft_2000.csv')
feature_matrix.shape
feature_matrix['SUM(ind.rez_esc / escolari)'] = feature_matrix['SUM(ind.rez_esc / escolari)'].astype(np.float32)
feature_matrix['SUM(ind.age / escolari)'] = feature_matrix['SUM(ind.age / escolari)'].astype(np.float32)
for col in feature_matrix:
    if feature_matrix[col].dtype == 'object':
        if col != 'idhogar':
            feature_matrix[col] = feature_matrix[col].astype(np.float32)
feature_matrix.columns[np.where(feature_matrix.dtypes == 'object')]
missing_threshold = 0.95
correlation_threshold = 0.99


train = feature_matrix[feature_matrix['Target'].notnull()]
test = feature_matrix[feature_matrix['Target'].isnull()]

train_ids = list(train.pop('idhogar'))
test_ids = list(test.pop('idhogar'))

feature_matrix = feature_matrix.replace({np.inf: np.nan, -np.inf:np.nan})
n_features_start = feature_matrix.shape[1]
print('Original shape: ', feature_matrix.shape)

# Find missing and percentage
missing = pd.DataFrame(feature_matrix.isnull().sum())
missing['fraction'] = missing[0] / feature_matrix.shape[0]
missing.sort_values('fraction', ascending = False, inplace = True)

# Missing above threshold
missing_cols = list(missing[missing['fraction'] > missing_threshold].index)
n_missing_cols = len(missing_cols)

# Remove missing columns
feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
print('{} missing columns with threshold: {}.'.format(n_missing_cols, missing_threshold))

# Zero variance
unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
n_zero_variance_cols = len(zero_variance_cols)

# Remove zero variance columns
feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
print('{} zero variance columns.'.format(n_zero_variance_cols))

# Correlations
corr_matrix = feature_matrix.corr()

# Extract the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

# Select the features with correlations above the threshold
# Need to use the absolute value
to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

n_collinear = len(to_drop)

feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
print('{} collinear columns removed with correlation above {}.'.format(n_collinear,  correlation_threshold))

total_removed = n_missing_cols + n_zero_variance_cols + n_collinear

print('Total columns removed: ', total_removed)
print('Shape after feature selection: {}.'.format(feature_matrix.shape))

# Remove columns derived from the Target
drop_cols = []
for col in feature_matrix:
    if col == 'Target':
        pass
    else:
        if 'Target' in col:
            drop_cols.append(col)

feature_matrix = feature_matrix[[x for x in feature_matrix if x not in drop_cols]]    

# Extract out training and testing data
train = feature_matrix[feature_matrix['Target'].notnull()]
test = feature_matrix[feature_matrix['Target'].isnull()]

train_ids = list(train.pop('idhogar'))
test_ids = list(test.pop('idhogar'))

train_labels = np.array(train.pop('Target')).reshape((-1, ))
test = test.drop(columns = 'Target')

train = train.replace({np.inf: np.nan, -np.inf: np.nan})
test = test.replace({np.inf: np.nan, -np.inf: np.nan})
from sklearn.impute import SimpleImputer

feature_list = list(train.columns)

imputer = SimpleImputer(strategy = 'median')
train = imputer.fit_transform(train)
test = imputer.transform(test)

train_df = pd.DataFrame(train, columns = feature_list)
test_df = pd.DataFrame(test, columns = feature_list)
train.shape
train_df = train_df.astype(np.float32)
test_df = test_df.astype(np.float32)
from timeit import default_timer as timer

n_components = 3
umap = UMAP(n_components=n_components)
pca = PCA(n_components=n_components)
ica = FastICA(n_components=n_components)
tsne = TSNE(n_components=n_components)
for method, name in zip([umap, pca, ica], ['umap', 'pca', 'ica']):
    
    if name == 'umap':
        start = timer()
        reduction = method.fit_transform(train, train_labels)
        test_reduction = method.transform(test)
        end = timer()
    
    else:
        start = timer()
        reduction = method.fit_transform(train)
        test_reduction = method.transform(test)
        end = timer()
        
    print(f'Method: {name} {round(end - start, 2)} seconds elapsed.')
    train_df['%s_c1' % name] = reduction[:, 0]
    train_df['%s_c2' % name] = reduction[:, 1]
    train_df['%s_c3' % name] = reduction[:, 2]
    
    test_df['%s_c1' % name] = test_reduction[:, 0]
    test_df['%s_c2' % name] = test_reduction[:, 1]
    test_df['%s_c3' % name] = test_reduction[:, 2]
train_df['label'] = train_labels
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
cmap = plt.get_cmap('tab10', 4)

for method, name in zip([umap, pca, ica], ['umap', 'pca', 'ica']):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(train_df['%s_c1' % name], train_df['%s_c2'  % name], train_df['%s_c3'  % name], c = train_df['label'].astype(int), cmap = cmap)
    plt.title(f'{name.capitalize()}')
    fig.colorbar(p, aspect = 4, ticks = [1, 2, 3, 4])
    
test_comp = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')
submission_base = test_comp.loc[:, ['idhogar', 'Id']]
def macro_f1_score(labels, predictions):
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True
def model_gbm(features, labels, test_features, test_ids, nfolds = 5, return_preds = False):
    """Model using the GBM and cross validation.
       Trains with early stopping on each fold.
       Hyperparameters probably need to be tuned."""
    
    feature_names = list(features.columns)
    
    # Model with hyperparameters selected from previous work
    model = lgb.LGBMClassifier(boosting_type = 'gbdt', n_estimators = 10000, max_depth = -1,
                               learning_rate = 0.025, metric = 'None', min_child_samples = 30,
                               reg_alpha = 0.35, reg_lambda = 0.6, num_leaves = 15, 
                               colsample_bytree = 0.85, objective = 'multiclass', 
                               class_weight = 'balanced', 
                               n_jobs = -1)
    
    # Using stratified kfold cross validation
    strkfold = StratifiedKFold(n_splits = nfolds, shuffle = True)
    predictions = pd.DataFrame()
    importances = np.zeros(len(feature_names))
    
    # Convert to arrays for indexing
    features = np.array(features)
    test_features = np.array(test_features)
    labels = np.array(labels).reshape((-1 ))
    
    valid_scores = []
    
    # Iterate through the folds
    for i, (train_indices, valid_indices) in enumerate(strkfold.split(features, labels)):
        # Dataframe for 
        fold_predictions = pd.DataFrame()
        
        # Training and validation data
        X_train = features[train_indices]
        X_valid = features[valid_indices]
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]
        
        # Train with early stopping
        model.fit(X_train, y_train, early_stopping_rounds = 100, 
                  eval_metric = macro_f1_score,
                  eval_set = [(X_train, y_train), (X_valid, y_valid)],
                  eval_names = ['train', 'valid'],
                  verbose = 200)
        
        # Record the validation fold score
        valid_scores.append(model.best_score_['valid']['macro_f1'])
        
        # Make predictions from the fold
        fold_probabilitites = model.predict_proba(test_features)
        
        # Record each prediction for each class as a column
        for j in range(4):
            fold_predictions[(j + 1)] = fold_probabilitites[:, j]
            
        fold_predictions['idhogar'] = test_ids
        fold_predictions['fold'] = (i+1)
        predictions = predictions.append(fold_predictions)
        
        importances += model.feature_importances_ / nfolds    

    feature_importances = pd.DataFrame({'feature': feature_names,
                                        'importance': importances})
    valid_scores = np.array(valid_scores)
    print(f'{nfolds} cross validation score: {round(valid_scores.mean(), 5)} with std: {round(valid_scores.std(), 5)}.')
    
    # If we want to examine predictions don't average over folds
    if return_preds:
        predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)
        predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)
        return predictions, feature_importances
    
    # Average the predictions over folds
    predictions = predictions.groupby('idhogar', as_index = False).mean()
    
    # Find the class and associated probability
    predictions['Target'] = predictions[[1, 2, 3, 4]].idxmax(axis = 1)
    predictions['confidence'] = predictions[[1, 2, 3, 4]].max(axis = 1)
    predictions = predictions.drop(columns = ['fold'])
    
    # Merge with the base to have one prediction for each individual
    submission = submission_base.merge(predictions[['idhogar', 'Target']], 
                                       on = 'idhogar', how = 'left').drop(columns = ['idhogar'])
        
    submission['Target'] = submission['Target'].fillna(4).astype(np.int8)
    
    # return the submission and feature importances
    return submission, feature_importances, valid_scores
train_df.head()
for col in train_df:
    if 'Target' in col:
        print(col)
predictions, fi = model_gbm(train_df.drop(columns = 'label'), train_labels, 
                                   test_df, test_ids, return_preds = True)
fi.sort_values('importance').dropna().tail()
submission, fi, scores = model_gbm(train_df.drop(columns = 'label'), train_labels, 
                                   test_df, test_ids, return_preds = False)

submission.to_csv('dimension_reduction.csv', index = False)
fi.sort_values('importance').dropna().tail(25)
scores.mean()
scores.std()
for method, name in zip([umap, pca, ica], ['umap', 'pca', 'ica']):
    start = timer()
    reduction = method.fit_transform(train)
    test_reduction = method.transform(test)
    end = timer()
    print(f'Method: {name} {round(end - start, 2)} seconds elapsed.')
    train_df['%s_c1' % name] = reduction[:, 0]
    train_df['%s_c2' % name] = reduction[:, 1]
    train_df['%s_c3' % name] = reduction[:, 2]
    
    test_df['%s_c1' % name] = test_reduction[:, 0]
    test_df['%s_c2' % name] = test_reduction[:, 1]
    test_df['%s_c3' % name] = test_reduction[:, 2]
cmap = plt.get_cmap('tab10', 4)

for method, name in zip([umap, pca, ica], ['umap', 'pca', 'ica']):
    fig = plt.figure(figsize = (8, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(train_df['%s_c1' % name], train_df['%s_c2'  % name], train_df['%s_c3'  % name], c = train_df['label'].astype(int), cmap = cmap)
    plt.title(f'{name.capitalize()}')
    fig.colorbar(p, aspect = 4, ticks = [1, 2, 3, 4])
submission, fi, scores = model_gbm(train_df.drop(columns = 'label'), train_labels, 
                                   test_df, test_ids, return_preds = False)

submission.to_csv('dimension_reduction_nolabels.csv', index = False)
fi.sort_values('importance').dropna().tail(25)
