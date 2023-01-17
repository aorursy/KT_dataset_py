import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import statistics

import math

import random



from numpy.linalg import inv

from scipy.linalg import cholesky

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

from sklearn.impute import SimpleImputer
train = pd.read_csv("../input/train.csv")

train.head()
train.shape
total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], 

                         axis = 1, 

                         keys = ['Total', 'Percent'])

missing_data.head(20)
COLUMNS_drop = (missing_data[missing_data['Total'] > 1]).index

train = train.drop(labels = COLUMNS_drop, axis = 1)

train.shape
INDEX_drop = train.loc[train['Electrical'].isnull()].index

train = train.drop(labels = INDEX_drop, axis = 0)

train.shape
COLUMNS = train.columns

COLUMNS_res = ""

COLUMNS_num = train._get_numeric_data().columns

COLUMNS_cat = list(set(COLUMNS) - set(COLUMNS_num))

print('Number of numeric columns: {}'.format(len(COLUMNS_num)))

print('Number of categorical columns: {}'.format(len(COLUMNS_cat)))
train = train[COLUMNS_num]

train.shape
corrmat = train.corr()

corrmat = corrmat.rename_axis(None).rename_axis(None, axis = 1)

corrmat = corrmat.stack().reset_index()

corrmat.columns = ['var_1', 'var_2', 'correlation']

corrmat = corrmat[corrmat['correlation'] != 1]

corrmat.sort_values(by = 'correlation', ascending = False).head(15)
COLUMNS_drop = ['GarageArea', 'TotRmsAbvGrd', 'TotalBsmtSF', '2ndFlrSF']

train = train.drop(labels = COLUMNS_drop, axis = 1)

train.shape
X = train.iloc[:, 1:(train.shape[1]-1)].values

y = train.iloc[:, train.shape[1]-1].values
def lpy_X(y, X, g = len(y), nu0 = 1):

    

    n = X.shape[0]

    p = X.shape[1]

    s20 = sum(sm.OLS(y, X).fit().resid**2) / sm.OLS(y, X).fit().df_resid

    

    if p == 0:

        Hg = 0

        s20 = statistics.mean(y**2)

    

    elif p > 0:

        X_T = np.transpose(X)

        Hg = (g/(g+1)) * np.dot(np.dot(X, inv(np.dot(X_T, X))), X_T)

        

    y_T = np.transpose(y)

    identity_mat = np.diag(np.repeat(1, n))

    i_H = identity_mat - Hg

    SSRg = np.dot(np.dot(y_T, i_H), y)

    

    return (-0.5

            * (n * np.log(np.pi) + p * np.log(1+g) + (nu0+n)*np.log(nu0*s20+SSRg) - nu0*np.log(nu0*s20))

            + math.lgamma((nu0+n)/2)

            + math.lgamma(nu0/2))
z = np.repeat(1, X.shape[1])

lpy_c = lpy_X(y = y, X = X[:, z == 1])

S = 100

Z = np.zeros([S, X.shape[1]], dtype = int)
for s in np.arange(0, S, 1):

        

    for j in pd.Series(np.arange(0,29,1)).sample(29, replace = False).values:

        

        zp = z.copy()

        zp[j] = 1 - zp[j]

        lpy_p = lpy_X(y, X[:, zp == 1])

        r = (lpy_p - lpy_c)*(-1)**(zp[j] == 0)

        z[j] = np.random.binomial(n=1, p=(1/(1+np.exp(-r))), size=1)

        

        if z[j] == zp[j]:

            lpy_c = lpy_p

    

    Z[s,:] = z  

    

    # Display sampling process by printing a single dot for each step

    print('.', end = '')

    

    if s == (S-1):

        print('Done!')
ps = pd.Series([tuple(i) for i in Z])

counts = ps.value_counts(normalize = True)

counts[0:5]
COLUMNS = np.array(train.columns)

COLUMNS = np.delete(COLUMNS, [0, 30])

COLUMNS = COLUMNS[np.array(counts.index[0]) == 1]

COLUMNS
train = pd.concat([train[['Id', 'SalePrice']], train[COLUMNS]], axis = 1)

train.head()
neighbor = pd.read_csv('../input/train.csv')[['Id', 'Neighborhood']]

train = pd.merge(left = train, right = neighbor, left_on = 'Id', right_on = 'Id', how = 'left')

print(train.shape)
_ = sns.catplot(x = 'Neighborhood', y = 'SalePrice', kind = 'box', data = train)

_ = plt.xticks(rotation = 90)

_ = plt.title('Boxplot of house prices by locations')

plt.show()
X_COLUMNS = COLUMNS



# preprocessing scale function shows warning if data type is int64

temp_df = train[X_COLUMNS].astype(np.float64)



Neighborhood_list = np.unique(train['Neighborhood'])

m = len(Neighborhood_list)

X_list = []



for i in np.arange(0,m,1):

    temp = temp_df.loc[train['Neighborhood'] == Neighborhood_list[i]]

    # centering, not scaling

    temp = preprocessing.scale(temp.values,

                               with_mean = True,

                               with_std = False)

    # adding intercept. do this after centering, otherwise intercept will be 0

    temp = sm.add_constant(temp)

    X_list.append(temp)

    

X = X_list
temp_df = train['SalePrice']

y_list = []



for i in np.arange(0, len(Neighborhood_list),1):

    y_list.append(temp_df.loc[train['Neighborhood'] == Neighborhood_list[i]])

    

y = y_list
def rmvnorm(n, mu, Sigma):

    E = np.random.normal(0, 1, n*len(mu))

    return np.dot(E, cholesky(Sigma, lower = False)) + mu
def rwish(nu0, S0):

    sS0 = cholesky(S0, lower = False)

    Z = np.dot(np.random.normal(0, 1, rwish_mean*rwish_var.shape[0]).reshape(rwish_mean, rwish_var.shape[0]), sS0)

    S = np.dot(np.transpose(Z), Z)

    return S
m = len(Neighborhood_list)

p = X[0].shape[1]

BETA_LS = np.zeros([m, p])

S2_LS = np.zeros(m)



for i in np.arange(0,m,1):

    

    # fit OLS to each group

    results = sm.OLS(y[i], X[i]).fit()

    

    # calculate parameters

    beta = results.params

    

    # calculate sample variance

    RSS = sum(results.resid ** 2)

    df = results.df_resid

    sample_variance = RSS/df

    

    # store outputs

    BETA_LS[i] = beta

    S2_LS[i] = sample_variance  
p = X[0].shape[1]

theta = pd.DataFrame(BETA_LS).apply(statistics.mean, axis = 0)

mu0 = pd.DataFrame(BETA_LS).apply(statistics.mean, axis = 0)

nu0 = 1

s2 = statistics.mean(S2_LS[np.isfinite(S2_LS)])

s20 = statistics.mean(S2_LS[np.isfinite(S2_LS)])

eta0 = p + 2

Sigma = np.cov(BETA_LS, rowvar = False)

S0 = np.cov(BETA_LS, rowvar = False)

L0 = np.cov(BETA_LS, rowvar = False)

BETA = BETA_LS

iL0 = inv(L0)

iSigma = inv(Sigma)



N = np.zeros(m)

for i in np.arange(0,m,1):

    N[i] = len(X[i])
# MCMC setting and storing

random.seed(0)

S = 5000

S2_b = np.zeros(S)

THETA_b = np.zeros([S, p])

Sigma_ps = np.zeros([p, p])

BETA_ps = BETA * 0

SIGMA_PS = np.zeros([S, p * p])

BETA_pp = np.zeros([S, p])
for s in np.arange(0,S,1):



    # update beta_j

    for j in np.arange(0,m,1):

        Vj = inv(iSigma + np.dot(np.transpose(X[j]), X[j])/s2)

        Ej = np.dot(Vj, np.dot(iSigma, theta) + np.dot(np.transpose(X[j]), y[j])/s2)

        BETA[j] = rmvnorm(1, Ej, Vj)



    # update theta

    Lm = inv(iL0 + m * iSigma)

    mum = np.dot(Lm, np.dot(iL0, mu0) + np.dot(iSigma, pd.DataFrame(BETA).apply(sum, axis = 0)))

    theta = rmvnorm(1, mum, Lm)



    # update Sigma

    rwish_mean = eta0 + m

    rwish_var = inv(S0 + np.dot(np.transpose(BETA - theta), BETA - theta))

    iSigma = rwish(rwish_mean, rwish_var)



    # update s2

    RSS = 0

    for j in np.arange(0,m,1):

        RSS = RSS + sum((y[j] - np.dot(X[j], BETA[j])) ** 2)

    s2 = 1/np.random.gamma(shape = (nu0 + sum(N))/2, scale = 1/((nu0*s20+RSS)/2), size = 1)



    # store results

    S2_b[s] = s2

    THETA_b[s] = theta

    Sigma_ps = Sigma_ps + inv(iSigma)

    BETA_ps = BETA_ps + BETA

    SIGMA_PS[s] = np.matrix.flatten(inv(iSigma))

    BETA_pp[s] = rmvnorm(1, theta, inv(iSigma))

    

    # Display sampling process by printing a single dot

    if (s % 100 == 0):

        print('.', end = '')

        

    if s == (S-1):

        print('Done!')
neighborhood_list_df = pd.DataFrame(Neighborhood_list,

                                    columns = ['Neighborhood'])

coef = pd.DataFrame(BETA_ps/S)

coef = pd.concat([neighborhood_list_df, coef], 

                 axis = 1)



# Rename columns

COLUMNS_coef = COLUMNS.copy()

COLUMNS_coef = COLUMNS_coef + '_coef'

COLUMNS_coef = np.append(['Neighborhood', 'Intercept_coef'], COLUMNS_coef)

coef.columns = COLUMNS_coef



coef
temp_y = np.matrix(y[0]).transpose()

temp_x = np.matrix(X[0])

temp_yx = pd.DataFrame(np.concatenate((temp_y, temp_x), axis = 1))

temp_yx['Neighborhood'] = Neighborhood_list[0]



m = len(Neighborhood_list)



for i in range(1, m):

    temp_y = np.matrix(y[i]).transpose()

    temp_x = np.matrix(X[i])

    temp = pd.DataFrame(np.concatenate((temp_y, temp_x), axis = 1))

    temp['Neighborhood'] = Neighborhood_list[i]

    temp_yx = pd.concat([temp_yx, temp], axis = 0)

    

# Rename columns

COLUMNS_pred = COLUMNS.copy()

COLUMNS_pred = np.append(['SalePrice', 'Intercept'], COLUMNS_pred)

COLUMNS_pred = np.append(COLUMNS_pred, 'Neighborhood')

temp_yx.columns = COLUMNS_pred

    

train_pred_df = temp_yx.merge(coef, on = 'Neighborhood', how = 'left')
train_pred_df.head()
# the 2nd argument of np.delete specifies the element position in array that you wan to drop.

COLUMNS_pred = np.delete(COLUMNS_pred, 0)

COLUMNS_pred = np.delete(COLUMNS_pred, len(COLUMNS_pred)-1)

train_pred_df['SalePrice_pred'] = 0



for i in range(len(COLUMNS_pred)):

    coefs = COLUMNS_pred[i] + '_coef'

    train_pred_df['SalePrice_pred'] = train_pred_df['SalePrice_pred'] + train_pred_df[coefs] * train_pred_df[COLUMNS_pred[i]]
np.round(train_pred_df[['SalePrice', 'SalePrice_pred']].head(), decimals = 0)
MSPE_train_BDA = statistics.mean((train_pred_df['SalePrice'] - train_pred_df['SalePrice_pred'])**2)

print('MSPE in training data by Bayesian model selection and hierarchical regression: {:,}'.format(round(MSPE_train_BDA)))
test = pd.read_csv("../input/test.csv")

test.shape
COLUMNS_pred = np.delete(COLUMNS_pred, 0)

COLUMNS_pred = np.append(['Id', 'Neighborhood'], COLUMNS_pred)

X_test = test[COLUMNS_pred]

print(X_test.shape)

print(X_test.isnull().sum())
# imputing missing values

COLUMNS_missing = COLUMNS_pred[X_test.isnull().sum() > 0]

test_temp = test[COLUMNS_missing]

imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')

test_temp = imp_mean.fit_transform(test_temp)

test_temp = pd.DataFrame(test_temp, columns = COLUMNS_missing)

print(test_temp.shape)

print(test_temp.isnull().sum())
COLUMNS_nonmiss = COLUMNS_pred[X_test.isnull().sum() == 0]

X_test = X_test[COLUMNS_nonmiss]

X_test = pd.concat([X_test, test_temp], axis = 1)
train = pd.read_csv("../input/train.csv")



for col in COLUMNS:

    

    mean = train.groupby('Neighborhood')[col].mean()

    mean_df = pd.DataFrame(data = {'Neighborhood':mean.index, 'mean':mean.values})

    X_test = pd.merge(left = X_test, right = mean_df, on = 'Neighborhood', how = 'left')

    X_test[col] = X_test[col] - X_test['mean'] # update original values with scaled values

    X_test = X_test.drop(columns = ['mean'])

    

X_test = pd.merge(left = X_test, right = coef, on = 'Neighborhood', how = 'left')
COLUMNS_pred = np.append('Intercept', COLUMNS)

X_test['SalePrice'] = 0

X_test['Intercept'] = 1



for i in range(len(COLUMNS_pred)):

    coefs = COLUMNS_pred[i] + '_coef'

    X_test['SalePrice'] = X_test['SalePrice'] + X_test[coefs] * X_test[COLUMNS_pred[i]]
np.round(X_test['SalePrice'].describe(), decimals = 0)
np.round(train['SalePrice'].describe(), decimals = 0)