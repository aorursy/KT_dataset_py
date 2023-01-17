# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# Visualizations!
import matplotlib.pyplot as plt
import seaborn as sns

# Plot formatting and default style
plt.rcParams['font.size'] = 18
plt.style.use('fivethirtyeight')
%matplotlib inline
# Read in data and sort
random = pd.read_csv('../input/home-credit-model-tuning/random_search_simple.csv').sort_values('score', ascending = False).reset_index()
opt = pd.read_csv('../input/home-credit-model-tuning/bayesian_trials_simple.csv').sort_values('score', ascending = False).reset_index()

print('Best score from random search:         {:.5f} found on iteration: {}.'.format(random.loc[0, 'score'], random.loc[0, 'iteration']))
print('Best score from bayesian optimization: {:.5f} found on iteration: {}.'.format(opt.loc[0, 'score'], opt.loc[0, 'iteration']))
import pprint
import ast

keys = []
for key, value in ast.literal_eval(random.loc[0, 'hyperparameters']).items():
    print(f'{key}: {value}')
    keys.append(key)
for key in keys:
    print('{}: {}'.format(key, ast.literal_eval(opt.loc[0, 'hyperparameters'])[key]))
# Kdeplot of model scores
plt.figure(figsize = (10, 6))
sns.kdeplot(opt['score'], label = 'Bayesian Opt')
sns.kdeplot(random['score'], label = 'Random Search')
plt.xlabel('Score (5 Fold Validation ROC AUC)'); plt.ylabel('Density');
plt.title('Random Search and Bayesian Optimization Results');
random['set'] = 'random'
scores = random[['score', 'iteration', 'set']]

opt['set'] = 'opt'
scores = scores.append(opt[['set', 'iteration', 'score']], sort = True)
scores.head()
plt.figure(figsize = (12, 6))

plt.subplot(121)
plt.hist(random['score'], bins = 20, color = 'blue', edgecolor = 'k')
plt.xlim((0.72, 0.80))
plt.xlabel('Score'); plt.ylabel("Count"); plt.title('Random Search Distribution of Scores');

plt.subplot(122)
plt.hist(opt['score'], bins = 20, color = 'blue', edgecolor = 'k')
plt.xlim((0.72, 0.80))
plt.xlabel('Score'); plt.ylabel("Count"); plt.title('Bayes Opt Search Distribution of Scores');
scores.groupby('set')['score'].agg(['mean', 'max', 'min', 'std', 'count'])
plt.rcParams['font.size'] = 16

best_random_score = random.loc[0, 'score']
best_random_iteration = random.loc[0, 'iteration']

best_opt_score = opt.loc[0, 'score']
best_opt_iteration = opt.loc[0, 'iteration']

sns.lmplot('iteration', 'score', hue = 'set', data = scores, size = 8)
plt.scatter(best_random_iteration, best_random_score, marker = '*', s = 400, c = 'blue', edgecolor = 'k')
plt.scatter(best_opt_iteration, best_opt_score, marker = '*', s = 400, c = 'red', edgecolor = 'k')
plt.xlabel('Iteration'); plt.ylabel('ROC AUC'); plt.title("Validation ROC AUC versus Iteration");
random_fit = np.polyfit(random['iteration'], random['score'], 1)
print('Random search slope: {:.8f}'.format(random_fit[0]))
opt_fit = np.polyfit(opt['iteration'], opt['score'], 1)
print('opt search slope: {:.8f}'.format(opt_fit[0]))
opt_fit[0] / random_fit[0]
print('After 10,000 iterations, the random score is: {:.5f}.'.format(
random_fit[0] * 1e5 + random_fit[1]))
print('After 10,000 iterations, the bayesian score is: {:.5f}.'.format(
opt_fit[0] * 1e5 + opt_fit[1]))
import ast
def process(results):
    """Process results into a dataframe with one column per hyperparameter"""
    
    results = results.copy()
    results['hyperparameters'] = results['hyperparameters'].map(ast.literal_eval)
    
    # Sort with best values on top
    results = results.sort_values('score', ascending = False).reset_index(drop = True)
    
     # Create dataframe of hyperparameters
    hyp_df = pd.DataFrame(columns = list(results.loc[0, 'hyperparameters'].keys()))

    # Iterate through each set of hyperparameters that were evaluated
    for i, hyp in enumerate(results['hyperparameters']):
        hyp_df = hyp_df.append(pd.DataFrame(hyp, index = [0]), 
                               ignore_index = True, sort= True)
        
    # Put the iteration and score in the hyperparameter dataframe
    hyp_df['iteration'] = results['iteration']
    hyp_df['score'] = results['score']
    
    return hyp_df
random_hyp = process(random)
opt_hyp = process(opt)

random_hyp.head()
# Hyperparameter grid
param_grid = {
    'is_unbalance': [True, False],
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100))
}
best_random_hyp = random_hyp.loc[0, :]
best_opt_hyp = opt_hyp.loc[0, :]
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18

# Density plots of the learning rate distributions 
sns.kdeplot(param_grid['learning_rate'], label = 'Sampling Distribution', linewidth = 4, color = 'k')
sns.kdeplot(random_hyp['learning_rate'], label = 'Random Search', linewidth = 4, color = 'blue')
sns.kdeplot(opt_hyp['learning_rate'], label = 'Bayesian', linewidth = 4, color = 'green')
plt.vlines([best_random_hyp['learning_rate']],
           ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['blue'])
plt.vlines([best_opt_hyp['learning_rate']],
           ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['green'])
plt.legend()
plt.xlabel('Learning Rate'); plt.ylabel('Density'); plt.title('Learning Rate Distribution');

print('Best value from random search: {:.5f}.'.format(best_random_hyp['learning_rate']))
print('Best value from Bayesian: {:.5f}.'.format(best_opt_hyp['learning_rate']))
def plot_hyp_dist(hyp):
    """Plots distribution of hyp along with best values of hyp as vertical line"""
    plt.figure(figsize = (16, 6))
    plt.rcParams['font.size'] = 18

    # Density plots of the learning rate distributions 
    sns.kdeplot(param_grid[hyp], label = 'Sampling Distribution', linewidth = 4, color = 'k')
    sns.kdeplot(random_hyp[hyp], label = 'Random Search', linewidth = 4, color = 'blue')
    sns.kdeplot(opt_hyp[hyp], label = 'Bayesian', linewidth = 4, color = 'green')
    plt.vlines([best_random_hyp[hyp]],
               ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['blue'])
    plt.vlines([best_opt_hyp[hyp]],
               ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['green'])
    plt.legend()
    plt.xlabel(hyp); plt.ylabel('Density'); plt.title('{} Distribution'.format(hyp));

    print('Best value from random search: {:.5f}.'.format(best_random_hyp[hyp]))
    print('Best value from Bayesian: {:.5f}.'.format(best_opt_hyp[hyp]))
    plt.show()
plot_hyp_dist('min_child_samples')
plot_hyp_dist('num_leaves')
plot_hyp_dist('reg_alpha')
plot_hyp_dist('reg_lambda')
plot_hyp_dist('subsample_for_bin')
plot_hyp_dist('colsample_bytree')
random_hyp.groupby('boosting_type')['score'].agg(['mean', 'max', 'min', 'std', 'count'])
opt_hyp.groupby('boosting_type')['score'].agg(['mean', 'max', 'min', 'std', 'count'])
plt.figure(figsize = (16, 6))

plt.subplot(121)
random_hyp.groupby('boosting_type')['score'].agg(['mean', 'max', 'min', 'std'])['mean'].plot.bar(color = 'b')
plt.ylabel('Score'); plt.title('Random Search Boosting Type Scores', size = 14);

plt.subplot(122)
opt_hyp.groupby('boosting_type')['score'].agg(['mean', 'max', 'min', 'std'])['mean'].plot.bar(color = 'b')
plt.ylabel('Score'); plt.title('Bayesian Boosting Type Scores', size = 14);
plt.figure(figsize = (20, 8))
plt.rcParams['font.size'] = 18

# Density plots of the learning rate distributions 
sns.kdeplot(param_grid['subsample'], label = 'Sampling Distribution', linewidth = 4, color = 'k')
sns.kdeplot(random_hyp[random_hyp['boosting_type'] == 'gbdt']['subsample'], label = 'Random Search', linewidth = 4, color = 'blue')
sns.kdeplot(opt_hyp[opt_hyp['boosting_type'] == 'gbdt']['subsample'], label = 'Bayesian', linewidth = 4, color = 'green')
plt.vlines([best_random_hyp['subsample']],
           ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['blue'])
plt.vlines([best_opt_hyp['subsample']],
           ymin = 0.0, ymax = 50.0, linestyles = '--', linewidth = 4, colors = ['green'])
plt.legend()
plt.xlabel('Subsample'); plt.ylabel('Density'); plt.title('Subsample Distribution');

print('Best value from random search: {:.5f}.'.format(best_random_hyp['subsample']))
print('Best value from Bayesian: {:.5f}.'.format(best_opt_hyp['subsample']))
random_hyp.groupby('is_unbalance')['score'].agg(['mean', 'max', 'min', 'std', 'count'])
opt_hyp.groupby('is_unbalance')['score'].agg(['mean', 'max', 'min', 'std', 'count'])
fig, axs = plt.subplots(1, 4, figsize = (24, 6))
i = 0

# Plot of four hyperparameters
for i, hyper in enumerate(['colsample_bytree', 'learning_rate', 'min_child_samples', 'num_leaves']):
        opt_hyp[hyper] = opt_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot('iteration', hyper, data = opt_hyp, ax = axs[i])
        axs[i].scatter(best_opt_hyp['iteration'], best_opt_hyp[hyper], marker = '*', s = 200, c = 'k')
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = '{} over Search'.format(hyper));

plt.tight_layout()
fig, axs = plt.subplots(1, 3, figsize = (18, 6))
i = 0

# Plot of four hyperparameters
for i, hyper in enumerate(['reg_alpha', 'reg_lambda', 'subsample_for_bin']):
        opt_hyp[hyper] = opt_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot('iteration', hyper, data = opt_hyp, ax = axs[i])
        axs[i].scatter(best_opt_hyp['iteration'], best_opt_hyp[hyper], marker = '*', s = 200, c = 'k')
        axs[i].set(xlabel = 'Iteration', ylabel = '{}'.format(hyper), title = '{} over Search'.format(hyper));

plt.tight_layout()
random_hyp['set'] = 'Random Search'
opt_hyp['set'] = 'Bayesian'

# Append the two dataframes together
hyp = random_hyp.append(opt_hyp, ignore_index = True, sort = True)
hyp.head()
fig, axs = plt.subplots(1, 4, figsize = (24, 6))
i = 0

# Plot of four hyperparameters
for i, hyper in enumerate(['colsample_bytree', 'learning_rate', 'min_child_samples', 'num_leaves']):
        random_hyp[hyper] = random_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data = random_hyp, ax = axs[i], color = 'b', scatter_kws={'alpha':0.6})
        axs[i].scatter(best_random_hyp[hyper], best_random_hyp['score'], marker = '*', s = 200, c = 'b', edgecolor = 'k')
        axs[i].set(xlabel = '{}'.format(hyper), ylabel = 'Score', title = 'Score vs {}'.format(hyper));
        
        opt_hyp[hyper] = opt_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data = opt_hyp, ax = axs[i], color = 'g', scatter_kws={'alpha':0.6})
        axs[i].scatter(best_opt_hyp[hyper], best_opt_hyp['score'], marker = '*', s = 200, c = 'g', edgecolor = 'k')

plt.legend()
plt.tight_layout()


# hyper = 'learning_rate'

# fig, ax = plt.subplots(1, 1, figsize = (6, 6))

# random_hyp[hyper] = random_hyp[hyper].astype(float)
# # Scatterplot
# sns.regplot(hyper, 'score', data = random_hyp, ax = ax, color = 'b', scatter_kws={'alpha':0.6})
# ax.scatter(best_random_hyp[hyper], best_random_hyp['score'], marker = '*', s = 200, c = 'b', edgecolor = 'k')

# opt_hyp[hyper] = opt_hyp[hyper].astype(float)
# # Scatterplot
# sns.regplot(hyper, 'score', data = opt_hyp, ax = ax, color = 'g', scatter_kws={'alpha':0.6})
# ax.scatter(best_opt_hyp[hyper], best_opt_hyp['score'], marker = '*', s = 200, c = 'g', edgecolor = 'k')

# ax.set(xlabel = '{}'.format(hyper), ylabel = 'Score', title = 'Score vs {}'.format(hyper))
# ax.set(xscale = 'log');
fig, axs = plt.subplots(1, 4, figsize = (24, 6))
i = 0

# Plot of four hyperparameters
for i, hyper in enumerate(['reg_alpha', 'reg_lambda', 'subsample_for_bin', 'subsample']):
        random_hyp[hyper] = random_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data = random_hyp, ax = axs[i], color = 'b', scatter_kws={'alpha':0.6})
        axs[i].scatter(best_random_hyp[hyper], best_random_hyp['score'], marker = '*', s = 200, c = 'b', edgecolor = 'k')
        axs[i].set(xlabel = '{}'.format(hyper), ylabel = 'Score', title = 'Score vs {}'.format(hyper));
        
        opt_hyp[hyper] = opt_hyp[hyper].astype(float)
        # Scatterplot
        sns.regplot(hyper, 'score', data = opt_hyp, ax = axs[i], color = 'g', scatter_kws={'alpha':0.6})
        axs[i].scatter(best_opt_hyp[hyper], best_opt_hyp['score'], marker = '*', s = 200, c = 'g', edgecolor = 'k')

plt.legend()
plt.tight_layout()


from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['axes.labelpad'] = 12
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(random_hyp['reg_alpha'], random_hyp['reg_lambda'],
           random_hyp['score'], c = random_hyp['score'], 
           cmap = plt.cm.seismic_r, s = 40)

ax.set_xlabel('Reg Alpha')
ax.set_ylabel('Reg Lambda')
ax.set_zlabel('Score')

plt.title('Score as Function of Reg Lambda and Alpha');
best_random_hyp
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(random_hyp['learning_rate'], random_hyp['n_estimators'],
           random_hyp['score'], c = random_hyp['score'], 
           cmap = plt.cm.seismic_r, s = 40)

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Number of Estimators')
ax.set_zlabel('Score')

plt.title('Score as Function of Learning Rate and Estimators', size = 16);
plt.figure(figsize = (8, 7))
plt.plot(random_hyp['learning_rate'], random_hyp['n_estimators'], 'ro')
plt.xlabel('Learning Rate'); plt.ylabel('N Estimators'); plt.title('Number of Estimators vs Learning Rate');
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_3d(x, y, z, df, cmap = plt.cm.seismic_r):
    """3D scatterplot of data in df"""

    fig = plt.figure(figsize = (10, 10))
    
    ax = fig.add_subplot(111, projection='3d')
    
    # 3d scatterplot
    ax.scatter(df[x], df[y],
               df[z], c = df[z], 
               cmap = cmap, s = 40)

    # Plot labeling
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    plt.title('{} as function of {} and {}'.format(
               z, x, y), size = 18);
    
plot_3d('learning_rate', 'n_estimators', 'score', opt_hyp)
plt.figure(figsize = (8, 7))
plt.plot(opt_hyp['learning_rate'], opt_hyp['n_estimators'], 'ro')
plt.xlabel('Learning Rate'); plt.ylabel('N Estimators'); plt.title('Number of Estimators vs Learning Rate');
plot_3d('reg_alpha', 'reg_lambda', 'score', opt_hyp)
best_opt_hyp
random_hyp['n_estimators'] = random_hyp['n_estimators'].astype(np.int32)
random_hyp.corr()['score']
random_hyp[random_hyp['boosting_type'] == 'gbdt'].corr()['score']['subsample']
opt_hyp['n_estimators'] = opt_hyp['n_estimators'].astype(np.int32)
opt_hyp.corr()['score']
opt_hyp[opt_hyp['boosting_type'] == 'gbdt'].corr()['score']['subsample']
plt.figure(figsize = (12, 12))

# Heatmap of correlations
sns.heatmap(random_hyp.corr().round(2), cmap = plt.cm.gist_heat_r, vmin = -1.0, annot = True, vmax = 1.0)
plt.title('Correlation Heatmap');
plt.figure(figsize = (12, 12))

# Heatmap of correlations
sns.heatmap(opt_hyp.corr().round(2), cmap = plt.cm.gist_heat_r, vmin = -1.0, annot = True, vmax = 1.0)
plt.title('Correlation Heatmap');
# Create training data and labels
hyp = hyp.drop(columns = ['metric', 'set', 'verbose'])
hyp['n_estimators'] = hyp['n_estimators'].astype(np.int32)
hyp['min_child_samples'] = hyp['min_child_samples'].astype(np.int32)
hyp['num_leaves'] = hyp['num_leaves'].astype(np.int32)
hyp['subsample_for_bin'] = hyp['subsample_for_bin'].astype(np.int32)
hyp = pd.get_dummies(hyp)

train_labels = hyp.pop('score')
train = np.array(hyp.copy())
from sklearn.linear_model import LinearRegression

# Create the lasso regression with cv
lr = LinearRegression()

# Train on the data
lr.fit(train, train_labels)
x = list(hyp.columns)
x_values = lr.coef_

coefs = {variable: coef for variable, coef in zip(x, x_values)}
coefs
import lightgbm as lgb
train = pd.read_csv('../input/home-credit-simple-featuers/simple_features_train.csv')
print('Full Training Features Shape: ', train.shape)
test = pd.read_csv('../input/home-credit-simple-featuers/simple_features_test.csv')
print('Full Testing Features Shape: ', test.shape)
train_labels = np.array(train['TARGET'].astype(np.int32)).reshape((-1, ))
train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])

test_ids = list(test['SK_ID_CURR'])
test = test.drop(columns = ['SK_ID_CURR'])
features = list(train.columns)
random_best = ast.literal_eval(random.loc[0, 'hyperparameters'])

rmodel = lgb.LGBMClassifier(**random_best)
rmodel.fit(train, train_labels)
rpreds = rmodel.predict_proba(test)[:, 1]
rsub = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': rpreds})
rsub.to_csv('submission_random_search.csv', index = False)
bayes_best = ast.literal_eval(opt.loc[0, 'hyperparameters'])

bmodel = lgb.LGBMClassifier(**bayes_best)
bmodel.fit(train, train_labels)
bpreds = bmodel.predict_proba(test)[:, 1]
bsub = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': bpreds})
bsub.to_csv('submission_bayesian_optimization.csv', index = False)
random_fi = pd.DataFrame({'feature': features, 'importance': rmodel.feature_importances_})
bayes_fi = pd.DataFrame({'feature': features, 'importance': bmodel.feature_importances_})
def plot_feature_importances(df):
    """
    Plots 15 most important features and returns a sorted feature importance dataframe.
    
    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance

        
    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column
    
    """
    
    plt.rcParams['font.size'] = 18
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()

    
    return df
norm_randomfi = plot_feature_importances(random_fi)
norm_randomfi.head(10)
norm_bayesfi = plot_feature_importances(bayes_fi)
norm_bayesfi.head(10)
random.loc[0, 'hyperparameters']
