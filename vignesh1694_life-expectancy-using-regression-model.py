import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.impute import SimpleImputer

import seaborn as sns

%matplotlib inline
df = pd.read_excel(r'../input/who-life-expectancy/Life_Expectancy_Data.xlsx')

df
df.columns = df.columns.str.replace(' ', '_')

df.columns = df.columns.str.rstrip('_')

df
df.drop(df[df["Year"] == 2015].index, inplace= True)
x = df['Life_expectancy']

plt.hist(x)

plt.show
df_mean = df['Life_expectancy'].mean()

df_mean
df['Life_Exp_log'] = np.log(df['Life_expectancy'])

y = df['Life_Exp_log']



plt.hist(y)

plt.show()
y.mean()
# plot data

fig, ax = plt.subplots(figsize=(12,5))

# use unstack()

df.groupby(['Year','Status']).mean()['Life_expectancy'].unstack().plot(ax=ax)
df_diff = df.groupby(['Year', 'Status']).mean()['Life_expectancy']

df_diff = pd.DataFrame(df_diff)

df_diff = df_diff.reset_index()

df_diff = df_diff.pivot_table('Life_expectancy', ['Year'], 'Status')

df_diff
df.plot(y = 'Adult_Mortality' , x = 'Life_expectancy', style = 'o')

plt.title('Adult_Mortality Vs Life_expectancy')

plt.xlabel('Life_expectancy')

plt.ylabel('Adult_Mortality')

plt.show()
df_temp = df.drop(['Country', 'Status', 'Year'], axis = 1)



fig, axes = plt.subplots(nrows = 5, ncols = 4, figsize = (60, 100))

for ax, column in zip(axes.flatten(), df_temp.columns):

    sns.distplot(df_temp[column].dropna(), ax = ax, color = 'darkred')

    ax.set_title(column, fontsize = 43)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 35)

    ax.tick_params(axis = 'both', which = 'minor', labelsize = 35)

    ax.set_xlabel('')

fig.tight_layout(rect = [0, 0.03, 1, 0.85])
null_columns = df.columns[df.isnull().any()]

df[null_columns].isnull().sum()
df_categorical = df.filter(['Country', 'Status'], axis =1)

df = df.drop(['Country', 'Status'], axis =1)

df
df_categorical = df_categorical.reset_index()

df_categorical = df_categorical.filter(['Country', 'Status'])
imputer = SimpleImputer(missing_values = np.nan, strategy ='mean')

imputer = imputer.fit(df)

df_impute = imputer.transform(df)

df_impute
df_impute = pd.DataFrame(df_impute)

df_impute = df_impute.rename(columns = {0 : "Year", 1: 'Life_expectancy', 2: 'Adult_Mortality', 3: 'infant_deaths', 4: 'Alcohol',

                                       5: 'percentage_expenditure', 6: 'Hepatitis_B', 7: 'Measles', 8: 'BMI', 9: 'under-five_deaths',

                                       10: 'Polio', 11: 'Total_expenditure', 12:'Diphtheria', 13: 'HIV/AIDS', 14: 'GDP', 15: 'Population',

                                       16: ' thinness_1-19_years', 17: 'thinness_5-9_years', 18: 'Income_composition_of_resources',

                                       19: 'Schooling', 20 : 'Life_exp_log'})
df = pd.merge(df_categorical, df_impute, left_index = True, right_index = True)

df['Year'] = df['Year'].astype(int)

df
X = df[['Adult_Mortality', 'infant_deaths', 'Alcohol', 'percentage_expenditure', 'Hepatitis_B', 'Measles', 'BMI', 'Polio','Diphtheria', 'Total_expenditure', 'GDP', 'Population', 'Income_composition_of_resources', 'Schooling']].values

Y = df['Life_exp_log'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()

regressor.fit(X_train, Y_train)
coeff_df = pd.DataFrame(regressor.coef_, columns=['Coefficient'])  

coeff_df
Y_pred = regressor.predict(X_test)

df_result = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

df_result
fig, ax = plt.subplots()

ax.scatter(Y_test, Y_pred)

ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=3)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
print(regressor.score(X_test, Y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))