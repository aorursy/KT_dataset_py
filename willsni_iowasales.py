from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, gaussian_process, neighbors, svm, model_selection
from xgboost import XGBRegressor 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
# %% Helpful Functions
from sklearn.base import TransformerMixin



class CustomImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with median of column.

        """
    def fit(self, X, excluded_features=[], y=None):

        fill_series = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].median() for c in X], index=X.columns)
        
        # Remove excluded features (i.e. ones that you want to not impute values for).
        fill_series.drop(index=excluded_features, inplace=True)
        
        self.fill = fill_series
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
def count_missing_data(df):
    """Obviously doesn't know if something should in fact be NA as opposed to a missing value."""
    print(df.isnull().sum())
    
def merge_duplicate_features(df, dup_features):
    """
    This was useful for the housing dataset because there were features like Condition1 and Condition2 with 
    identical options but entered in separately. When doing one-hot encoding, we can combine them into a joint
    """
    for [c1,c2, joint_name] in dup_features:
        
        c1_endings = [col.replace(c1+"_","") for col in df.columns if c1+"_" in col]
        c2_endings = [col.replace(c2+"_","") for col in df.columns if c2+"_" in col]
        
        # Get unique ending values by converting to dict and back
        all_endings = list(dict.fromkeys(c1_endings+c2_endings))
        
        for end in all_endings:
            
            c1_with_end = c1+"_"+end
            c2_with_end = c2+"_"+end
            joint_with_end = joint_name+"_"+end
            
            if c1_with_end in df.columns and c2_with_end in df.columns:
                c1_feature = df[c1_with_end]
                c2_feature = df[c2_with_end]
                joint_feature = np.logical_or(c1_feature,c2_feature)
                df[joint_with_end] = joint_feature
                df.drop(columns=[c1_with_end, c2_with_end], inplace=True)
            elif c1_with_end in df.columns and c2_with_end not in df.columns:
                c1_feature = df[c1_with_end]
                joint_feature = c1_feature
                df[joint_with_end] = joint_feature
                df.drop(columns=[c1_with_end], inplace=True)
            elif c1_with_end not in df.columns and c2_with_end in df.columns:
                c2_feature = df[c2_with_end]
                joint_feature = c2_feature
                df[joint_with_end] = joint_feature
                df.drop(columns=c2_with_end, inplace=True)
            else:
                print("This should never occur; check code.") 

def compare_MLA(X, X_target, vote_est):
    """
    Make a quick table showing model performance without hyperparameter tuning.
    """
    MLA_list = [alg[1] for alg in vote_est]

    cv_split = model_selection.KFold(n_splits = 10)

    #create table to compare MLA metrics
    MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
    MLA_compare = pd.DataFrame(columns = MLA_columns)

    #create table to compare MLA predictions
    #MLA_predict = X_target.copy()

    #index through MLA and save performance to table
    row_index = 0
    for alg in MLA_list:

        #set name and parameters
        MLA_name = alg.__class__.__name__
        print(MLA_name)
        MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
        MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
        
        #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        cv_results = model_selection.cross_validate(alg, X, X_target, cv  = cv_split, scoring='neg_mean_squared_error', return_train_score = True, n_jobs=-1)

        MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
        MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = np.sqrt(-cv_results['train_score']).mean()
        MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = np.sqrt(-cv_results['test_score']).mean()   
        #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
        MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
        

        #save MLA predictions - see section 6 for usage
        #alg.fit(X, X_target)
        #MLA_predict[MLA_name] = alg.predict(X)
        
        row_index+=1

        
    #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
    #MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
    print(MLA_compare[ ['MLA Name', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean']])
    return MLA_compare
    #MLA_predict    

def plot_feature_importances(reg, train_data, y, num_vars):
    reg.fit(train_data,y)
    feature_importance = reg.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[-num_vars:]
    pos = np.arange(sorted_idx.shape[0]) + .5
    most_important_features = train_data.columns[sorted_idx]
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, most_important_features)
    plt.xlabel('Relative Importance')
    plt.title('Top {} Variable Importances'.format(num_vars))
    plt.show()
    return most_important_features

os.chdir('/kaggle/input')



# Make df show full size
# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 15)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)

# Gather data
train_data = pd.read_csv(Path("house-prices-advanced-regression-techniques/train.csv"))
test_data = pd.read_csv(Path("house-prices-advanced-regression-techniques/test.csv"))
test_copy = test_data.copy()

# Extract y
y = train_data['SalePrice']
y = np.log1p(y) 

# Remove unwanted features
unwanted_features = ['Id']
train_data.drop(columns=unwanted_features, inplace=True)
train_data.drop(columns="SalePrice", inplace=True)
test_data.drop(columns=unwanted_features, inplace=True)

# Remove outliers - chosen from features that are in the top 5 most importance to a gradient boosting regressor and have a small number of clear outliers.
train_data = train_data[train_data['BsmtFinSF1'] <= 5000]
train_data = train_data[train_data['TotalBsmtSF'] <= 4000]
train_data = train_data[train_data['GrLivArea'] <= 4000]
train_data = train_data[train_data['LotArea'] <= 1000000]
y = y[train_data.index]
train_data.reset_index(drop=True, inplace=True)

# Impute median missing values for both numerical and categorical features, except certain excluded features (that have NA as a real category, not NaN)
excluded_features = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
CI = CustomImputer()
train_data = CI.fit_transform(train_data, excluded_features)
test_data = CI.transform(test_data)

# Define categorical variables
more_cat_vars = ['MSSubClass']
cat_vars = [f for f in train_data.columns if train_data[f].dtype == np.dtype('O')]
cat_vars.extend(more_cat_vars)
num_vars = [var for var in train_data.columns if var not in cat_vars]

# Transform to dummy variables
train_data = pd.get_dummies(train_data, columns=cat_vars, dummy_na=True) # dummy_na needed!!!
test_data = pd.get_dummies(test_data, columns=cat_vars, dummy_na=True)

# Merge duplicate features
dup_features = [ ['Condition1', 'Condition2', 'ConditionJoint'], ['Exterior1st', 'Exterior2nd', 'ExteriorJoint'], ['BsmtFinType1', 'BsmtFinType2', 'BsmtFinTypeJoint']]
merge_duplicate_features(train_data, dup_features)
merge_duplicate_features(test_data, dup_features)

# Make sure columns of train_data and test_data match. Can have problems if one dataset does not contain an instance of a feature while the other does. One-hot encoding thus causes mismatches in the encoding. 
missing_train_features = [f for f in test_data.columns if f not in train_data.columns]
for f in missing_train_features: train_data[f] = 0
missing_test_features = [f for f in train_data.columns if f not in test_data.columns]
for f in missing_test_features: test_data[f] = 0 
test_data = test_data[train_data.columns] # Rearrange order of columns to match

# Make boxplots to show outliers of most important variables. 
# most_important_features = plot_feature_importances(ensemble.GradientBoostingRegressor(), train_data, y, 10)
# for f in most_important_features:
#     train_data[f].plot.box(vert=False)
#     plt.show()


# Rescale the data
scaler = StandardScaler()
train_data[train_data.columns] = scaler.fit_transform(train_data)
test_data[test_data.columns] = scaler.transform(test_data)


# Pick estimators
vote_est = [
    
    # Ensemble
    ('ada', ensemble.AdaBoostRegressor()),
    ('br', ensemble.BaggingRegressor()),
    ('etr',ensemble.ExtraTreesRegressor()),
    ('gbr', ensemble.GradientBoostingRegressor()),
    ('rfr', ensemble.RandomForestRegressor()),

    #Gaussian Processes
    ('gpr', gaussian_process.GaussianProcessRegressor()),
    
    #Nearest Neighbor
    ('knn', neighbors.KNeighborsRegressor()),
    
    #SVM
    ('svr', svm.SVR()),

    #xgboost
   ('xgb', XGBRegressor())
]

# Get sense of estimator accuracy
MLA_compare = compare_MLA(train_data, y, vote_est)

grid_n_estimator = [10, 33, 100, 333, 1000, 3333, 10000]
grid_ratio = [0.1, 0.25, 0.5, 0.75, 1.0]
grid_learn = [.01, 0.03, 0.1, 0.33, 1.0]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, 0.03, 0.05, 0.10]
grid_criterion = ['mse']
grid_bool = [False]

grid_param = [
            [{
            #AdaBoostRegressor - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
            'n_estimators': grid_n_estimator, #default=50
            'learning_rate': grid_learn #default=1
            }],
       
    
            [{
            #BaggingRegressor - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor
            'n_estimators': grid_n_estimator, #default=10
            'max_samples': grid_ratio #default=1.0
             }],

    
            [{
            #ExtraTreesRegressor 
            'n_estimators': grid_n_estimator, 
            'criterion': grid_criterion, 
            'max_depth': grid_max_depth
             }],


            [{
            #GradientBoostingRegressor - http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor
            #'loss': ['deviance', 'exponential'], #default=’deviance’
            'learning_rate': grid_learn,
            'n_estimators': grid_n_estimator,
            #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”
            'max_depth': grid_max_depth    
             }],

    
            [{
            #RandomForestRegressor
            'n_estimators': grid_n_estimator, #default=100
            'criterion': grid_criterion,
            'max_depth': grid_max_depth, 
            'oob_score': grid_bool
             }],
    
            [{    
            #GaussianProcessRegressor
            'alpha': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
            }],
    
            [{
            #KNeighborsRegressor 
            'n_neighbors': [1,2,3,4,5,6,7], #default: 5
            'weights': ['uniform', 'distance'], #default = ‘uniform’
            'algorithm': ['ball_tree', 'kd_tree', 'brute']
            }],
            
    
            [{
            #SVR - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            #http://blog.hackerearth.com/simple-tutorial-svm-parameter-tuning-python-r
            #'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C': [0.01, 0.03, 0.1, 0.33, 1, 3, 10, 33, 100], #default=1.0
            'gamma': ['auto', 'scale'],
            'epsilon': [1e-4, 1e-3, 1e-2, 1e-1]
             }],

    
            [{
            #XGBRegressor - http://xgboost.readthedocs.io/en/latest/parameter.html
            'learning_rate': grid_learn, #default: .3
            'max_depth': [1,2,4,6,8,10] #default 2
             }]   
        ]



start_total = time.perf_counter() 
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size=0.2)
for reg, param in zip (vote_est, grid_param): #https://docs.python.org/3/library/functions.html#zip

    #if reg[0] in [alg[0] for alg in vote_est[:5]]: continue
    #print(reg[1]) #vote_est is a list of tuples, index 0 is the name and index 1 is the algorithm
    #print(param)
    
    
    start = time.perf_counter()        
    best_search = model_selection.GridSearchCV(estimator = reg[1], param_grid = param, cv = cv_split, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    best_search.fit(train_data, y)
    cv_score = np.sqrt(-best_search.best_score_)

    run = time.perf_counter() - start

    best_param = best_search.best_params_
    print('\nThe best parameter for {} is \n{} with a runtime of {:.2f} seconds.\n This gave a cv score of {}\n'.format(reg[1].__class__.__name__, best_param, run, cv_score))
    reg[1].set_params(**best_param) 


run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))
os.chdir("/kaggle/input/model-params-for-regression/")

print('-'*10)
# Save model parameters
pkl_filename = "model_params.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(vote_est, file)

# # Load model parameters
pkl_filename = "model_params.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# See accuracy after hyperparameter tuning
MLA_compare_tuned = compare_MLA(train_data, y, vote_est)

os.chdir("/kaggle/working")

# Make submission
vote_est[3][1].fit(train_data, y)
predictions = vote_est[3][1].predict(test_data)
predictions = np.expm1(predictions)
output = pd.DataFrame({'Id': test_copy['Id'], 'SalePrice': predictions})
csv_name = 'GradientBoosting6_submission.csv'
competition = "house-prices-advanced-regression-techniques"
output.to_csv(csv_name, index=False)
print("Your submission was successfully saved!")

# kaggle.api.competition_submit(csv_name, csv_name, competition)
# kaggle.api.competitions_submissions_list(competition)[0]

print('-'*10)
    
# Make submission
voter_idx = [1,2,3,4,8]
voters = [vote_est[i] for i in voter_idx]
stack = ensemble.StackingRegressor(estimators=voters, cv=10, n_jobs=-1, final_estimator=XGBRegressor())
stack.fit(train_data, y)
predictions = stack.predict(test_data)
predictions = np.expm1(predictions)
output = pd.DataFrame({'Id': test_copy['Id'], 'SalePrice': predictions})
csv_name = 'Stacked1_submission.csv'
competition = "house-prices-advanced-regression-techniques"
output.to_csv(csv_name, index=False)
print("Your submission was successfully saved!")

# kaggle.api.competition_submit(csv_name, csv_name, competition)
# kaggle.api.competitions_submissions_list(competition)[0]