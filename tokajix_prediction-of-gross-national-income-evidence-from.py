# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True, rc={'figure.figsize':(200,150)})

import seaborn as seabornInstance

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# read the csv

df1 = pd.read_csv("/kaggle/input/human-development/human_development.csv")

df2 = pd.read_csv("/kaggle/input/human-development/gender_inequality.csv")

df3 = pd.read_csv("/kaggle/input/human-development/inequality_adjusted.csv")



# select the variable we interested in

df1 = df1[['Country','Life Expectancy at Birth','Mean Years of Education','Gross National Income (GNI) per Capita']]

df2 = df2[['Country','Maternal Mortality Ratio','Percent Representation in Parliament','Labour Force Participation Rate (Male)','Labour Force Participation Rate (Female)']]

df3 = df3[['Country','Income Inequality (Gini Coefficient)']]



# merge all the dataset

merged_inner = pd.merge(left=df1,right=df2, left_on='Country', right_on='Country')

df = pd.merge(left=merged_inner,right=df3, left_on='Country', right_on='Country')



# data cleaning

# string.replace(",", "")

df['Gross National Income (GNI) per Capita'] = df['Gross National Income (GNI) per Capita'].str.replace(',', '')



# to numeric vaule

for col in df.columns[1:]:

    df[col] = df[col].apply(pd.to_numeric, errors='coerce')

    

# caculate Gender Difference in Labour Force Participation Rate

df['Gender Difference in Labour Force Participation Rate'] = df['Labour Force Participation Rate (Male)']- df['Labour Force Participation Rate (Female)']

df = df.drop(['Labour Force Participation Rate (Male)', 'Labour Force Participation Rate (Female)'], axis=1)



# do linear transformation on Maternal Mortality Ratio

df['Maternal Mortality Ratio transformed']  = df['Maternal Mortality Ratio'].transform(lambda x: np.log(x))
# drop the region (e.g.: Asia/Europe) data

df.drop([189,190,191,192,193,194])



#change the col order

cols = df.columns.tolist()

cols = cols[0:3] + cols[5:] + cols[4:5] +cols[3:4]

df = df[cols]



# delete the row that contains the null value

df = df.fillna(method='ffill')



#descriptive statistics summary

df.describe()
# show median

for i in cols[1:]:

    print('the median of ',i ,'is ',df[i].median())
# Density & Hist Plot of each R.V.

sns.distplot(df[cols[1]], hist=True, kde=True, 

             bins=int(35), color = 'b', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
sns.distplot(df[cols[2]], hist=True, kde=True, 

             bins=int(35), color = 'b', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
sns.distplot(df[cols[3]], hist=True, kde=True, 

             bins=int(35), color = 'b', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
sns.distplot(df[cols[4]], hist=True, kde=True, 

             bins=int(35), color = 'b', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
sns.distplot(df[cols[5]], hist=True, kde=True, 

             bins=int(35), color = 'b', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
sns.distplot(df[cols[6]], hist=True, kde=True, 

             bins=int(35), color = 'b', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
sns.distplot(df[cols[7]], hist=True, kde=True, 

             bins=int(35), color = 'b', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
sns.distplot(df[cols[8]], hist=True, kde=True, 

             bins=int(35), color = 'b', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 2})
# check the skewness and kurtosis of each R.V.

print("Gross National Income (GNI) per Capita->Skewness: %f" % df['Gross National Income (GNI) per Capita'].skew())

print("Gross National Income (GNI) per Capita->Kurtosis: %f" % df['Gross National Income (GNI) per Capita'].kurt())

print('')

print("Life Expectancy at Birth->Skewness: %f" % df['Life Expectancy at Birth'].skew())

print("Life Expectancy at Birth->Kurtosis: %f" % df['Life Expectancy at Birth'].kurt())

print('')

print("Mean Years of Education->Skewness: %f" % df['Mean Years of Education'].skew())

print("Mean Years of Education->Kurtosis: %f" % df['Mean Years of Education'].kurt())

print('')

print("Maternal Mortality Ratio->Skewness: %f" % df['Maternal Mortality Ratio'].skew())

print("Maternal Mortality Ratio->Kurtosis: %f" % df['Maternal Mortality Ratio'].kurt())

print('')

print("Maternal Mortality Ratio transformed->Skewness: %f" % df['Maternal Mortality Ratio transformed'].skew())

print("Maternal Mortality Ratio transformed->Kurtosis: %f" % df['Maternal Mortality Ratio transformed'].kurt())

print('')

print("Percent Representation in Parliament->Skewness: %f" % df['Percent Representation in Parliament'].skew())

print("Percent Representation in Parliament->Kurtosis: %f" % df['Percent Representation in Parliament'].kurt())

print('')

print("Income Inequality (Gini Coefficient)->Skewness: %f" % df['Income Inequality (Gini Coefficient)'].skew())

print("Income Inequality (Gini Coefficient)->Kurtosis: %f" % df['Income Inequality (Gini Coefficient)'].kurt())

print('')

print("Gender Difference in Labour Force Participation Rate->Skewness: %f" % df['Gender Difference in Labour Force Participation Rate'].skew())

print("Gender Difference in Labour Force Participation Rate->Kurtosis: %f" % df['Gender Difference in Labour Force Participation Rate'].kurt())

# Correlation Matrix

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, cmap="RdBu", square=True, annot=True, center=0);
# multivariable regression model



import statsmodels.api as sm



# only take the promising variables:life expe:['Life Expectancy at Birth', 'Mean Years of Education', 'Maternal Mortality Ratio transformed']

multivariable_cols = cols[1:3]+cols[6:7]



X = df[multivariable_cols]

y = df['Gross National Income (GNI) per Capita']

X = sm.add_constant(X)



predictions = sm.OLS(y, X).fit().predict(X) # make the predictions by the model



# Print out the statistics

model.summary()
# multivariable RLM

# only take the promising variables:life expe:['Life Expectancy at Birth', 'Mean Years of Education', 'Maternal Mortality Ratio transformed']



X = df[multivariable_cols]

y = df['Gross National Income (GNI) per Capita']

X = sm.add_constant(X)



model = sm.RLM(y, X).fit()



# Print out the statistics

model.summary()
# RLM for signal variable

import statsmodels.api as sm



for i in range(1,8):

    X = df[cols[i]] ## X usually means our input variables (or independent variables)

    y = df['Gross National Income (GNI) per Capita'] ## Y usually means our output/dependent variable

    X = sm.add_constant(X)



    # Fit model and print summary

    rlm_model = sm.RLM(y, X)

    rlm_results = rlm_model.fit()

    print(rlm_results.summary())

# linear regression with each single variable

import statsmodels.api as sm # import statsmodels



for i in range(1,8):

    X = df[cols[i]] ## X usually means our input variables (or independent variables)

    y = df['Gross National Income (GNI) per Capita'] ## Y usually means our output/dependent variable

    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model



    # Note the difference in argument order

    model = sm.OLS(y, X).fit() ## sm.OLS(output, input)

    predictions = model.predict(X)



    # Print out the statistics

    print('the depend variable is: ', cols[i])

    print(model.summary())

    print('')

    print('')
sns.set(style="whitegrid")



# Plot the residuals after fitting a linear model

sns.residplot(df[cols[1]], df['Gross National Income (GNI) per Capita'], lowess=True, color="g")
sns.residplot(df[cols[2]], df['Gross National Income (GNI) per Capita'], lowess=True, color="g")
sns.residplot(df[cols[3]], df['Gross National Income (GNI) per Capita'], lowess=True, color="g")
sns.residplot(df[cols[4]], df['Gross National Income (GNI) per Capita'], lowess=True, color="g")
sns.residplot(df[cols[5]], df['Gross National Income (GNI) per Capita'], lowess=True, color="g")
sns.residplot(df[cols[6]], df['Gross National Income (GNI) per Capita'], lowess=True, color="g")
sns.residplot(df[cols[7]], df['Gross National Income (GNI) per Capita'], lowess=True, color="g")
# visualization the regression of rach R.V. 

sns.regplot(x=cols[1],y='Gross National Income (GNI) per Capita', data=df)
sns.regplot(x=cols[2],y='Gross National Income (GNI) per Capita', data=df)
sns.regplot(x=cols[3],y='Gross National Income (GNI) per Capita', data=df)
sns.regplot(x=cols[4],y='Gross National Income (GNI) per Capita', data=df)
sns.regplot(x=cols[5],y='Gross National Income (GNI) per Capita', data=df)
sns.regplot(x=cols[6],y='Gross National Income (GNI) per Capita', data=df)
sns.regplot(x=cols[7],y='Gross National Income (GNI) per Capita', data=df)
# Bonus:ML/regression model 



X = df[multivariable_cols].values

y = df['Gross National Income (GNI) per Capita'].values



# spilt the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# train the simple model

regressor = LinearRegression()  

regressor.fit(X_train, y_train)



# use model to do the predication

y_pred = regressor.predict(X_test)

predictdf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})



# model performance

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))