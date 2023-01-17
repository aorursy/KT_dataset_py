# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from matplotlib import rcParams
rcParams['figure.figsize'] = 11, 8
def fill_nan(table):
    for col in table.columns:
        table[col] = table[col].fillna(table[col].median())
    return table   
data = pd.read_csv('../input/credit_scoring_sample.csv')
data.head()
data.dtypes
ax = data['SeriousDlqin2yrs'].hist(orientation='horizontal', color='red')
ax.set_xlabel("number_of_observations")
ax.set_ylabel("unique_value")
ax.set_title("Target distribution")

print('Distribution of the target:')
data['SeriousDlqin2yrs'].value_counts()/data.shape[0]
independent_columns_names = [x for x in data if x != 'SeriousDlqin2yrs']
independent_columns_names
table = fill_nan(data)
X = table[independent_columns_names]
y = table['SeriousDlqin2yrs']
table.shape
def get_bootstrap_samples(data, n_samples):
    """Generate bootstrap samples using the bootstrap method."""
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples

def stat_intervals(stat, alpha):
    """Produce an interval estimate."""
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

age_0 = table[table['SeriousDlqin2yrs'] == 0].age.values
age_1 = table[table['SeriousDlqin2yrs'] == 1].age.values

# Set the seed for reproducibility of the results
np.random.seed(0)

# Generate the samples using bootstrapping and calculate the mean for each of them
age_0_mean_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(age_0, 1000)]
age_1_mean_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(age_1, 1000)]

# Print the resulting interval estimates
print("average age for the customers who is good client: mean interval", stat_intervals(age_0_mean_scores, 0.1))
print("average age for the customers who delayed payment: mean interval", stat_intervals(age_1_mean_scores, 0.1))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
lr = LogisticRegression(random_state=5, class_weight='balanced')
parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
locally_best_lr = GridSearchCV(lr, parameters, scoring='roc_auc', n_jobs=-1, cv=skf)
locally_best_lr.fit(X, y)
locally_best_lr.best_params_, locally_best_lr.best_score_
best_roc_auc = locally_best_lr.best_score_
best_roc_auc
locally_best_lr.cv_results_['std_test_score'][1]
from sklearn.preprocessing import StandardScaler
logit = LogisticRegression(C=0.001, random_state=5, class_weight='balanced')
scale = StandardScaler()
X_scaled = scale.fit_transform(X)
logit.fit(X_scaled, y)
logit.coef_.reshape((7))
feat_imp = pd.DataFrame({'feature' : independent_columns_names, 'coef' : logit.coef_.flatten()})
feat_imp.sort_values(by='coef', ascending=False)
independent_columns_names
dept_ratio_coef = logit.coef_.flatten()[2]
softmax = np.exp(logit.coef_.flatten()) / np.sum(np.exp(logit.coef_.flatten()))
print(softmax[2])
logit_not_scaled = LogisticRegression(C=0.001, random_state=5, class_weight='balanced')
logit_not_scaled.fit(X, y)

feat_imp = pd.DataFrame({'feature' : independent_columns_names, 'coef' : logit_not_scaled.coef_.flatten()})
feat_imp.sort_values(by='coef', ascending=False)
np.exp(logit_not_scaled.coef_[0][0]*20)
np.exp(logit_not_scaled.coef_[0][0])

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, 
                            class_weight='balanced')
parameters = {'max_features': [1, 2, 4], 'min_samples_leaf': [3, 5, 7, 9], 'max_depth': [5,10,15]}
rf_grid = GridSearchCV(rf, parameters, scoring='roc_auc', n_jobs=-1, cv=skf)
rf_grid.fit(X, y)
rf_grid.best_score_
best_roc_auc
feat_imp_rf = pd.DataFrame({'feature' : independent_columns_names, 'coef' : rf_grid.best_estimator_.feature_importances_})
feat_imp_rf.sort_values(by='coef', ascending=True)
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

parameters = {'max_features': [2, 3, 4], 'max_samples': [0.5, 0.7, 0.9], 
              'base_estimator__C': [0.0001, 0.001, 0.01, 1, 10, 100]}
bc = BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced'), n_estimators=100, random_state=42)
bag_c = RandomizedSearchCV(bc, parameters, n_iter=20, scoring='roc_auc', cv=skf, random_state=1, n_jobs=-1)
bag = bag_c.fit(X, y)
bag.best_score_
bag.best_estimator_
