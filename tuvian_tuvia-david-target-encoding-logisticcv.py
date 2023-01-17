# !wget https://raw.githubusercontent.com/avidale/ps4ds2019/master/homework/week4/credit_card_clients_split.csv
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats

from sklearn.neighbors import KernelDensity 

%matplotlib inline
data = pd.read_csv('credit_card_clients_split.csv')

print(data.shape) # number of rows and columns

data.head()
data.isnull().mean()
data.describe()
train = data.dropna().copy()

print(train.shape)
train.avg_monthly_turnover.describe()
np.log(1 + train.avg_monthly_turnover).hist(bins=50, density=True)

plt.xlabel('Logarithm of monthly turnover');

plt.ylabel('Density');
positive = train[train.avg_monthly_turnover > 0].copy()
positive['log_turnover'] = np.log10(1 + positive.avg_monthly_turnover)

data['log_turnover'] = np.log10(1 + data.avg_monthly_turnover)

positive['log_turnover'].hist(bins=50)

plt.xlabel('Logarithm of monthly turnover')

plt.ylabel('Density')

print(positive['log_turnover'].describe())

print(positive['log_turnover'].skew())
plt.figure(figsize=(12,3))

plt.subplot(1, 3, 1)

positive.income.hist(bins=50, density=True)

plt.title('Distribution of income')

plt.subplot(1, 3, 2)

np.log10(positive.income).hist(bins=50, density=True)

plt.title('Distribution of log income')

plt.subplot(1, 3, 3)

positive.age.hist(bins=61, density=True)

plt.title('Distribution of age')

plt.tight_layout()
positive['log_income'] = np.log10(1 + positive.income)

data['log_income'] = np.log10(1 + data.income)
positive[['log_turnover', 'log_income', 'age']].corr()
plt.scatter(positive.log_income, positive.log_turnover, s=0.1);
mean_vector = positive[['log_income',  'log_turnover']].mean()

print(mean_vector)

cov_matrix = positive[['log_income', 'log_turnover']].cov()

print(cov_matrix)
X, Y = np.meshgrid(

    np.linspace(*positive.log_income.quantile([0,1]).values, 1000),

    np.linspace(*positive.log_turnover.quantile([0,1]).values, 1000)

)

density_norm = scipy.stats.multivariate_normal(

    mean=mean_vector, cov=cov_matrix).pdf(

    np.vstack([X.ravel(), Y.ravel()]).T

).reshape(X.shape)

plt.scatter(positive.log_income, positive.log_turnover, s=0.01, c='green');

CS = plt.contour(X, Y, density_norm, levels=10)
from mpl_toolkits.mplot3d import Axes3D 

from matplotlib import cm



# X, Y = np.meshgrid(

#     np.linspace(*positive.log_income.quantile([0,1]).values, 100),

#     np.linspace(*positive.log_turnover.quantile([0,1]).values, 100))



# density = dens_c.cdf(Y.flatten(), X.flatten()).reshape((100,100))



fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X=X, Y=Y, Z=density_norm, cmap=cm.coolwarm)

ax.set_xlabel('log income')

ax.set_ylabel('log turnover')

plt.title('joint normal PDF of log income and log turnover')

ax.view_init(50, -85) # you can change these angles to rotate the plot
def get_conditional_distribution_parameters(mean_vec, cov_mat):

    """ Calculate parameters of conditional distribution of the second variable conditional on the first one """

    std1 = np.sqrt(cov_mat.iloc[0,0])

    std2 = np.sqrt(cov_mat.iloc[1,1])

    cov = cov_mat.iloc[0,1]

    corr = cov / (std1*std2)

    var2 = cov_mat.iloc[1,1]

    

    slp = corr * (std2/std1)

    intcpt = mean_vec[1] - slp*mean_vec[0]

    cond_var = var2 * (1 - np.square(corr))

    print(intcpt, slp, cond_var)

    return intcpt, slp, cond_var



# testing with an independent dataset

intercept, slope, cond_variance = get_conditional_distribution_parameters(pd.Series([7, 11]), pd.DataFrame([[1,0], [0,4]]))

assert intercept == 11

assert slope == 0

assert cond_variance == 4



# testing with an independent dataset

intercept, slope, cond_variance = get_conditional_distribution_parameters(pd.Series([4, 11]), pd.DataFrame([[1,1], [1,4]]))

assert intercept == 7

assert slope == 1

assert cond_variance == 3



mean_vector = positive[['log_income',  'log_turnover']].mean()

print(mean_vector)

cov_matrix = positive[['log_income', 'log_turnover']].cov()

print(cov_matrix)
intercept, slope, cond_variance = get_conditional_distribution_parameters(mean_vector, cov_matrix)
from sklearn.model_selection import train_test_split
train.columns
target_encoding = positive['education'].value_counts()/train['education'].value_counts()
target_encoding.loc[train['education']].values
for cat_variable in ['education', 'sales_channel_id', 'wrk_rgn_code', 'age']:

    target_encoding = positive[cat_variable].value_counts()/train[cat_variable].value_counts()

    train[f'{cat_variable}_target_encoding'] = target_encoding.loc[train[cat_variable]].values
letter_dict = {'A':7, 'US':1, 'S':2, 'SS':3, 'UH':4, 'H':5, 'HH':6, }

class_list = [ 'log_income', 'age', 'education_target_encoding', 'sales_channel_id_target_encoding', 'wrk_rgn_code_target_encoding', 'age_target_encoding']
train['log_income'] = np.log10(1 + train.income)
train['positive_turnover'] = (train.avg_monthly_turnover > 0).astype(int)
positive['education'] = positive['education'].apply(lambda x: letter_dict[x])

train['education'] = train['education'].apply(lambda x: letter_dict[x])

data['education'] = data['education'].apply(lambda x: letter_dict[x])
train['age:log_income']  = train['age']*train['log_income']
x_train, y_train = train[class_list],  train['positive_turnover']
from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=10, max_iter=250,)
for column in class_list:

    x_train[column] = x_train[column].fillna(x_train[column].mean())
x_train.head(1)
clf.fit(x_train[class_list], y_train)
pred = clf.predict_proba(x_train)
plt.hist(pred[:,1], bins=50)

plt.show()
# data['education'] = data['education'].apply(lambda x: letter_dict[x])
data['age:log_income']  = data['age']*data['log_income']
for cat_variable in ['education', 'sales_channel_id', 'wrk_rgn_code', 'age']:

    target_encoding = positive[cat_variable].value_counts()/train[cat_variable].value_counts()

    data[f'{cat_variable}_target_encoding'] = target_encoding.loc[data[cat_variable]].values
data.head(1)
for column in class_list:

    data[column] = data[column].fillna(data[column].mean())
data['probability_of_positive_turnover'] =  clf.predict_proba(data[class_list])[:,1]
data['mean_log_turnover_if_positive'] = (intercept + np.log10(1 + data.income) * slope)
data['mean_log_turnover'] = data['probability_of_positive_turnover'] * data['mean_log_turnover_if_positive'] 
np.mean((data['mean_log_turnover'] - np.log10(1+data.avg_monthly_turnover))**2)
np.mean((np.mean(np.log10(1+data.avg_monthly_turnover)) - np.log10(1+data.avg_monthly_turnover))**2)
data['predicted_turnover'] = 10 ** data['mean_log_turnover'] 
submission = data.loc[data.avg_monthly_turnover.isnull(), ['id', 'predicted_turnover']].copy()
submission.head()
submission.columns = ['id', 'avg_monthly_turnover']

submission.to_csv('first_model.csv', index=None)
submission.head()