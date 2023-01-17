# Import packages

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

import numpy as np 

import pandas as pd 
raw_data = pd.read_csv('../input/Dataset_spine.csv')



# View the column names given in the last column

raw_data.iloc[:, -1].head(20)
# Remove the last column that only contains meta data for us

data = raw_data.iloc[:, :-1]



# View the data prior to renaming the columns for comparison

print(data.head())
# Rename the columns in place using a dictionary and the information found in the 13th column

meta_column_names = {"Col1" : "pelvic_incidence", "Col2" : "pelvic_tilt",

                     "Col3" : "lumbar_lordosis_angle","Col4" : "sacral_slope", 

                     "Col5" : "pelvic_radius","Col6" : "degree_spondylolisthesis", 

                     "Col7" : "pelvic_slope","Col8" : "direct_tilt", 

                     "Col9" : "thoracic_slope", "Col10" :"cervical_tilt", 

                     "Col11" : "sacrum_angle", "Col12" : "scoliosis_slope"}



# Rename the columns using the above dictionary, and replace the existing data object with inplace = True

data.rename(columns = meta_column_names, inplace = True)
# Features

X = data.iloc[:, :-1]

# Response

y = data.iloc[:, -1]



print(X.shape)

print(y.shape)
data.head()
correlations = X.corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(correlations, square=True, cbar=True, annot=True, vmax=.9);
sns.pairplot(X.iloc[:, :6], height = 1.75)

plt.show()
fg, ax = plt.subplots(ncols=4, nrows=3, sharex=False, sharey=False, figsize=(15,12))

fg.tight_layout()

# Add h spacing and space at the bottom for the legend

plt.subplots_adjust(hspace=0.25, bottom = 0.1)



# If we reshape the 2D numpy array that is returned from subplots for the axes then we can loop through without worrying about dimensions

for index, axes in enumerate(ax.reshape(-1)):

    normal = sns.distplot(data[data.Class_att == 'Normal'].iloc[:, index], ax=axes, color='#2980b9', label = 'Normal')

    abnormal = sns.distplot(data[data.Class_att == 'Abnormal'].iloc[:, index], ax=axes, color='#e74c3c', label = 'Abnormal')

    

# For this to work you have to have named the lines AND given labels 

# Get the handles and lines from the final axes - we can do this in this case because the lines are the same for every subplot

h, l = axes.get_legend_handles_labels()

fg.legend(h, l, loc='lower center', ncol = 2)

plt.show()
sns.swarmplot(y='degree_spondylolisthesis', x='Class_att', data=data)

plt.title('degree_spondylolisthesis difference between classes\n')

plt.xlabel('')

plt.ylabel('')

plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled)
type(X_scaled)
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

scaled_df['Class_att'] = y

scaled_df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.25, random_state = 1000)
proportion_train = round(len(y_train[y_train == 'Abnormal'])/len(y_train)*100, 2)

proportion_test = round(len(y_test[y_test == 'Abnormal'])/len(y_test)*100, 2)

print('Train case proportion: {}%'.format(proportion_train))

print('Test case proportion:  {}%'.format(proportion_test))

print('Difference between test and train sets: {}%'.format(round(np.abs(proportion_train - proportion_test), 2)))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.25, random_state = 1000, stratify = y)
proportion_train = round(len(y_train[y_train == 'Abnormal'])/len(y_train)*100, 2)

proportion_test = round(len(y_test[y_test == 'Abnormal'])/len(y_test)*100, 2)

print('Train case proportion: {}%'.format(proportion_train))

print('Test case proportion:  {}%'.format(proportion_test))

print('Difference between test and train sets: {}%'.format(round(np.abs(proportion_train - proportion_test), 2)))
# Import decision trees and accuracy metric 

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
# We are going to be doing this a lot, so better to define a function

def print_accuracy(accuracy_score, score_text=False):

    """

    Take an accuracy score between 0 and 1 and print output to screen cleanly 

    """

    clean_accuracy = accuracy_score*100.0

    if score_text:

        clean_text = score_text.strip() + ' '

        print('{}{:.2f}%'.format(clean_text, clean_accuracy))

    else:

        print('{:.2f}%'.format(clean_accuracy))
tree_model = DecisionTreeClassifier(random_state = 1000).fit(X_train, y_train)



# Predict on the test set and get the accuracy using the known values 

tree_pred = tree_model.predict(X_test)



print_accuracy(accuracy_score(y_test, tree_pred), 'Decision tree accuracy:')
from sklearn.model_selection import GridSearchCV
# Define our parameter grid that will be looped through to test our hyperparameters

param_grid = {'max_depth': [i for i in range(1, 11)], 

              'max_features': [i for i in range(1, 8)], 

              'min_samples_leaf': [i for i in range(1, 11)]}



grid = GridSearchCV(DecisionTreeClassifier(random_state = 1000), param_grid, cv=10, return_train_score = True)
grid.fit(X_train, y_train)
print("Best Score: {}%".format(round(grid.best_score_*100.0, 2)))

print("Best params: {}".format(grid.best_params_))
print_accuracy(accuracy_score(y_test, grid.best_estimator_.predict(X_test)), 'Decision Tree Classifier:')
keys = []

shapes = []

examples = []

for key in list(grid.cv_results_.keys()):

    keys.append(key)

    shapes.append(len(grid.cv_results_.get(key)))

    examples.append(grid.cv_results_.get(key)[0])



df_results_info = pd.DataFrame({'size': shapes, 'example': examples}, index = keys)

df_results_info
plt.hist(grid.cv_results_.get('mean_fit_time'), bins = 70);plt.show()
mean_fit_times = grid.cv_results_.get('mean_fit_time')

mean_test_scores = grid.cv_results_.get('mean_test_score')

max_score = np.argmax(mean_test_scores)



plt.scatter(mean_fit_times, mean_test_scores)

plt.xlim(min(mean_fit_times)*0.95, max(mean_fit_times*1.05))

plt.scatter(mean_fit_times[max_score], mean_test_scores[max_score])

plt.show()
max_depth = grid.cv_results_.get('param_max_depth').astype(int)

max_features = grid.cv_results_.get('param_max_features').astype(int)

min_samples_leaf = grid.cv_results_.get('param_min_samples_leaf').astype(int)

mean_fit_times = grid.cv_results_.get('mean_fit_time')

mean_test_scores = grid.cv_results_.get('mean_test_score')



params_df = pd.DataFrame({'max_depth' : max_depth, 

                          'max_features' : max_features, 

                          'min_samples_leaf' : min_samples_leaf,

                          'mean_fit_time' : mean_fit_times,

                          'mean_test_score' : mean_test_scores})
fig, ax = plt.subplots(nrows= 1, ncols = 3, figsize = (15,5), sharey=True, sharex=True)

plt.xlim(min(mean_fit_times)*0.95, max(mean_fit_times*1.05))



features = ['max_depth', 'max_features', 'min_samples_leaf']

axes = ax.flatten()



for idx in range(3):

    sns.scatterplot(y = 'mean_test_score', x = 'mean_fit_time', hue = features[idx], data = params_df, ax = axes[idx], palette='Greens_r')

    

plt.show()
from sklearn.linear_model import LogisticRegression
param_grid = {'penalty': ['l1', 'l2']}

grid = GridSearchCV(LogisticRegression(random_state = 1000, solver = 'liblinear'), param_grid, cv=10, return_train_score=True)
grid.fit(X_train, y_train)
print("Best Score: {}".format(grid.best_score_))

print("Best params: {}".format(grid.best_params_))
print_accuracy(accuracy_score(y_test, grid.best_estimator_.predict(X_test)), 'Best logisitic regression grid score:')
# Solver suppresses the warnings: more information about which solver for which problems can be found in the docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

log_reg_l1 = LogisticRegression(penalty='l1', solver = 'liblinear', random_state=1000)

log_model_l1 = log_reg_l1.fit(X_train, y_train)

y_pred_l1 = log_model_l1.predict(X_test)

print(log_model_l1.coef_)



print_accuracy(accuracy_score(y_test, y_pred_l1), 'L1 Regularised test accuracy:')
# Having a look at the l2 regularisation, which isn't capable of fully removing features

log_reg_l2 = LogisticRegression(penalty='l2', solver = 'liblinear', random_state=1000)

log_model_l2 = log_reg_l2.fit(X_train, y_train)

y_pred_l2 = log_model_l2.predict(X_test)

print(log_model_l2.coef_)



print_accuracy(accuracy_score(y_test, y_pred_l2), 'L2 Regularised test accuracy:')
grid.best_estimator_.coef_
log_model_l2.coef_
log_model_l2.coef_ == grid.best_estimator_.coef_
test_splits = [i for i in list(grid.cv_results_.keys()) if i.endswith('test_score') and i.startswith('split')]

test_split_results = {}

penalty = list(grid.cv_results_.get('param_penalty'))



for split in test_splits:

    test_split_results[split] = list(grid.cv_results_.get(split))

    

df_test_splits = pd.DataFrame(test_split_results, index = penalty)

df_test_splits.columns = [i.replace('_test_score', '') for i in df_test_splits.columns]
df_test_splits
df_test_splits.loc['l1'] >= df_test_splits.loc['l2']
list(grid.cv_results_.get('mean_test_score'))
long_test_splits = pd.DataFrame(df_test_splits.unstack().reset_index())

long_test_splits.columns = ['split', 'penalty', 'score']

long_test_splits
sns.boxplot(x='penalty', y="score", data=long_test_splits)

plt.xlabel('Penalty')

plt.ylabel('Test score')

plt.show()
# Import support vector machines

from sklearn.svm import SVC
poly_model = SVC(kernel = 'poly', gamma = 'auto', random_state = 1000).fit(X_train, y_train)

rbf_model = SVC(kernel = 'rbf', gamma = 'auto', random_state = 1000).fit(X_train, y_train)



poly_pred = poly_model.predict(X_test)

print_accuracy(accuracy_score(y_test, poly_pred), 'Polynomial kernel accuracy:')



rbf_pred = rbf_model.predict(X_test)

print_accuracy(accuracy_score(y_test, rbf_pred), 'RBF kernel accuracy:')
param_grid = {'gamma': np.arange(0, 0.25, 0.01)}

grid = GridSearchCV(SVC(random_state = 1000, kernel = 'rbf'), param_grid, cv=5, return_train_score=True)

grid.fit(X_train, y_train)
print("Best Score: {}".format(grid.best_score_))

print("Best params: {}".format(grid.best_params_))

print(1/12)
rbf_model_2 = SVC(kernel = 'rbf', gamma = 0.03, random_state = 1000).fit(X_train, y_train)

rbf_pred_2 = rbf_model_2.predict(X_test)



print_accuracy(accuracy_score(y_test, rbf_pred), 'Original RBF kernel accuracy:')

print_accuracy(accuracy_score(y_test, rbf_pred_2), 'RBF kernel accuracy with gamma of 0.06:')
gammas = list(grid.cv_results_.get('param_gamma'))

mean_test_scores = list(grid.cv_results_.get('mean_test_score'))

mean_train_times = list(grid.cv_results_.get('mean_fit_time'))
fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True, sharey = False)

axs = axes.flatten()



plt.sca(axs[0])

sns.lineplot(gammas, mean_test_scores, color = '#44bd32')

plt.axvline(0.06, color = '#fbc531')

plt.ylabel('Mean Test Score')

plt.title('Mean Test Score during CV as Gamma increases')



plt.sca(axs[1])

sns.lineplot(x=gammas, y=mean_train_times, color="#8c7ae6")

plt.ylabel('Mean Training Time')

plt.axvline(0.06, color = '#fbc531')

plt.xlabel('Gamma value')

plt.show()