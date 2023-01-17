import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# for the Q-Q plots
import scipy.stats as stats

# the dataset for the demo
from sklearn.datasets import load_boston

# for linear regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

# to split and standarize the dataset
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,RobustScaler

# to evaluate the regression model
from sklearn.metrics import mean_squared_error

from feature_engine import outlier_removers as outr

from feature_engine import categorical_encoders as ce

from sklearn.neighbors import LocalOutlierFactor

distance = 1.5
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/Hitters.csv')
data
data.info()
data.describe([0.05,0.10,0.25,0.50,0.75,0.80,0.90,0.95,0.99]).T
data.ndim
data.shape
# find the variables with missing observations

vars_with_na = [var for var in data.columns if data[var].isnull().mean() > 0]
vars_with_na
# let's find out whether they are numerical or categorical
data[vars_with_na].dtypes
# let's have a look at the values of the variables with
# missing data

data[vars_with_na].head(10)
# let's find out the percentage of observations missing per variable

# calculate the percentage of missing (as we did in section 3)
# using the isnull() and mean() methods from pandas
data_na = data[vars_with_na].isnull().mean()

# transform the array into a dataframe
data_na = pd.DataFrame(data_na.reset_index())

# add column names to the dataframe
data_na.columns = ['variable', 'na_percentage']

# order the dataframe according to percentage of na per variable
data_na.sort_values(by='na_percentage', ascending=False, inplace=True)

# show
data_na
#veri setini inceledigimizde , sadece salary bagimli degiskeninde na degerler oldugnu gözlemliyoruz.
#ve bu na degiskenlerinin sistematik degil tamamen rastgle bir sekilde nan atandigini görüyoruz.
# bu nedenle prediction öncesi mean medyan degerleri atamak yerine siliyoruz 

data.dropna(subset = ["Salary"], inplace=True)
data.head()
#birden fazla independent degiskenin birbirine cok fazla korele olmasi 


# we calculate the correlations using pandas corr
# and we round the values to 2 decimals
correlation_matrix = data.corr().round(2) 

# plot the correlation matrix usng seaborn
# annot = True to print the correlation values
# inside the squares

figure = plt.figure(figsize=(12, 12))
sns.heatmap(data=correlation_matrix, annot=True)
correlation_matrix = data.corr().round(2)
threshold=0.75
filtre=np.abs(correlation_matrix['Salary'])>0.50
corr_features=correlation_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot=True,fmt=".2f")
plt.title('Correlation btw features')
plt.show()

# correlation between RAD (index of accessibility to radial highways)
# and TAX (full-value property-tax rate per $10,000)

sns.lmplot(x="CRuns", y="CHits", data=data, order=1)
sns.lmplot(x="CWalks", y="Hits", data=data, order=1)
#sns.pairplot(data,diag_kind='kde',markers='+')
#plt.show()
plt.figure()
sns.countplot(data['League'])
plt.figure()
sns.countplot(data['NewLeague'])
plt.figure()
sns.countplot(data['Division'])
# function to create histogram, Q-Q plot and
# boxplot. We learned this in section 3 of the course


def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.distplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    plt.show()
# let's find outliers in Salary

diagnostic_plots(data, 'Salary')
# let's find outliers in Errors

diagnostic_plots(data, 'Errors')
# let's find outliers in Assists

diagnostic_plots(data, 'Assists')
# let's find outliers in PutOuts

diagnostic_plots(data, 'PutOuts')
# let's find outliers in CWalks

diagnostic_plots(data, 'CWalks')
# let's find outliers in CRBI

diagnostic_plots(data, 'CRBI')
# let's find outliers in CHmRun
diagnostic_plots(data, 'CHmRun')
# let's find outliers in Years
diagnostic_plots(data, 'Years')
# let's find outliers in HmRun
diagnostic_plots(data, 'HmRun')
# let's find outliers in CAtBat
diagnostic_plots(data, 'CAtBat')
# let's find outliers in CHits
diagnostic_plots(data, 'CHits')
# let's find outliers in CRuns
diagnostic_plots(data, 'CRuns')
def find_skewed_boundaries(df, variable):

    # Let's calculate the boundaries outside which sit the outliers
    # for skewed distributions

    # distance passed as an argument, gives us the option to
    # estimate 1.5 times or 3 times the IQR to calculate
    # the boundaries.

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary
RM_upper_limit, RM_lower_limit = find_skewed_boundaries(data, 'Salary')
RM_upper_limit, RM_lower_limit

outliers_salary = np.where(data['Salary'] > RM_upper_limit, True, np.where(data['Salary'] < RM_lower_limit, True, False))
# let's trimm the dataset

data = data.loc[~(outliers_salary)]
data[data["Salary"] > 2000]
diagnostic_plots(data, 'Salary')
#diagnostic_plots(data1, 'Salary')
data = pd.get_dummies(data, columns = ['League', 'Division', 'NewLeague'], drop_first = True)
data.head()
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(data)[0:10]
X_scores = clf.negative_outlier_factor_

X_scores
threshold=np.sort(X_scores)[10]
threshold
data=data.loc[X_scores > threshold]
data.shape
# log transform the variables
data['CRuns'] = np.log(data['CRuns'])
data['CHits'] = np.log(data['CHits'])
data['CAtBat'] = np.log(data['CAtBat'])
data['Years'] = np.log(data['Years'])
data['CRBI'] = np.log(data['CRBI'])
data['CWalks'] = np.log(data['CWalks'])

# Separate into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
            data.drop(['Salary'], axis=1),
            data['Salary'], test_size=0.2, random_state=42)

# set up the capper
capper = outr.OutlierTrimmer(
    distribution='skewed', tail='right', fold=3 , variables=[ 
                                                              'CRuns',
                                                              'CHits',
                                                              'CAtBat',
                                                              'HmRun',
                                                              'Years',
                                                              'CHmRun',
                                                              'CRBI',
                                                              'CWalks',
                                                              'PutOuts',
                                                              'Assists',
                                                              'Errors'])
# fit the capper
capper.fit(X_train)

# transform the data
train_t= capper.transform(X_train)
test_t= capper.transform(X_test)
train_t[['CRuns', 'CHits']].max()
data[['CRuns', 'CHits']].max()
# let's find outliers in CHits
diagnostic_plots(data, 'CRuns')
# let's find outliers in CHits
diagnostic_plots(train_t, 'CRuns')
# let's find outliers in CHits
diagnostic_plots(train_t, 'CHits')
# let's find outliers in CHits
diagnostic_plots(test_t, 'CRuns')
train_t= X_train
test_t= X_test
train_t
# let's scale the features
scaler = RobustScaler()
scaler.fit(train_t)
# model build using the natural distributions

# call the model
linreg = LinearRegression()

# fit the model
linreg.fit(scaler.transform(train_t), y_train)

# make predictions and calculate the mean squared
# error over the train set
print('Train set')
pred = linreg.predict(scaler.transform(train_t))
print('Linear Regression mse: {}'.format(mean_squared_error(y_train, pred)))

# make predictions and calculate the mean squared
# error over the test set
print('Test set')
pred = linreg.predict(scaler.transform(test_t))
print('Linear Regression mse: {}'.format(mean_squared_error(y_test, pred)))
print(np.sqrt(mean_squared_error(y_test, pred)))
error = y_test - pred
sns.distplot(error, bins=30)
np.sqrt(mean_squared_error(y_test, pred))
#train_t= encoder.transform(X_train)
#test_t= encoder.transform(X_test)


ridge=Ridge(random_state=42,max_iter=30000)
alphas=np.logspace(-4,-0.5,30)
tuned_parameters=[ {"alpha":alphas} ]
n_folds=5

clf=GridSearchCV(ridge,tuned_parameters,cv=n_folds,scoring="neg_mean_squared_error",refit=True)
clf.fit(train_t,y_train)
scores=clf.cv_results_["mean_test_score"]
scores_std= clf.cv_results_["std_test_score"]
clf.best_estimator_.coef_

ridge= clf.best_estimator_
ridge
y_predicted_dummy=clf.predict(test_t)
mse=mean_squared_error(y_test, y_predicted_dummy)

mse

np.sqrt(mse)
