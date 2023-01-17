# Load library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import statsmodels.api as sm
# Load the dataset
df = pd.read_csv("../input/final_dataset_V2copy.csv")
df.head()
df_sub = df[['HomeTeam','AwayTeam','Day_Eve','Hol_Type','Day_Type','Capacity','Average_Travelling_Fans','Cheapest_Season_T',
             'Home_League_Position','Away_League_Position','Form_Home','Form_Away','Distance','Temperature','Lowest_Home_Ticket_Price',
             'Lowest_Away_Ticket_Price','Highest_Home_Ticket_Price','match_month','Attendance']]
df_sub.iloc[30,:]
# convert the teams columns to numbers using label decoding
df_sub['HTeam']=df_sub['HomeTeam'].astype('category')
df_sub["HTeam"] = df_sub["HTeam"].cat.codes
df_sub['ATeam']=df_sub['AwayTeam'].astype('category')
df_sub["ATeam"] = df_sub["ATeam"].cat.codes
df_sub["DayEve"] = df_sub["Day_Eve"].astype('category')
df_sub["DayEve"] = df_sub["DayEve"].cat.codes
df_sub.head()
# show the encoded values/mappings
htcodes = df_sub[['HomeTeam','HTeam']]
htcodes = htcodes.drop_duplicates(['HomeTeam','HTeam'])
htcodes
# show the encoded values/mappings
atcodes = df_sub[['AwayTeam','ATeam']]
atcodes = atcodes.drop_duplicates(['AwayTeam','ATeam'])
atcodes
# show the encoded values/mappings
dayeve_codes = df_sub[['Day_Eve','DayEve']]
dayeve_codes = dayeve_codes.drop_duplicates(['Day_Eve','DayEve'])
dayeve_codes
# drop the original home and away team columns
df_sub = df_sub[['HTeam','ATeam','DayEve','Hol_Type','Day_Type','Capacity','Average_Travelling_Fans','Cheapest_Season_T',
             'Home_League_Position','Away_League_Position','Form_Home','Form_Away','Distance','Temperature','Lowest_Home_Ticket_Price',
             'Lowest_Away_Ticket_Price','Highest_Home_Ticket_Price','match_month','Attendance']]
df_sub.head()
X = np.array(df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position','Away_League_Position',
       'Form_Home','Form_Away','Distance','Temperature','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']])
y = df_sub.Attendance
correlationMatrix = df_sub.corr().abs()

plt.subplots(figsize=(18, 8))
sns.heatmap(correlationMatrix,annot=True)

# Mask unimportant features
sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar=False)
plt.show()
# Scale the features: X_scaled
X_scaled = scale(X)
X_scaled
# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()
# Fit the regressor to the training data
reg_all.fit(X_train,y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# calculate MAE using scikit-learn
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, y_pred))
# calculate MSE using scikit-learn
print(metrics.mean_squared_error(y_test, y_pred))
# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
import matplotlib.pyplot as plt

# regression coefficients
print('Coefficients: \n', reg_all.coef_)
 
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg_all.score(X_test, y_test)))
 
# plot for residual error
 
## setting plot style
plt.style.use('fivethirtyeight')
 
## plotting residual errors in training data
plt.scatter(reg_all.predict(X_train), reg_all.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(reg_all.predict(X_test), reg_all.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
 
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Residual errors")
 
## function to show plot
plt.show()
residuals = y_test - y_pred
residuals.describe()
import scipy.stats as stats
plt.figure(figsize=(9,9))
stats.probplot(residuals, dist="norm", plot=plt)
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position','Away_League_Position',
       'Form_Home','Form_Away','Distance','Temperature','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
y = df_sub.Attendance
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2, fit_intercept=True).fit()
regressor_OLS.summary()
# remove Temperature as greater than 0.05
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position','Away_League_Position',
       'Form_Home','Form_Away','Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove away_league_position as greater than 0.05
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position',
       'Form_Home','Form_Away','Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove lowest_home_Ticket_price as greater than 0.05
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position',
       'Form_Home','Form_Away','Distance','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove form_away as greater than 0.05
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position',
       'Form_Home','Distance','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove flowest_away_Ticket_price as greater than 0.05
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position',
       'Form_Home','Distance',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove ATeam as greater than 0.05
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position',
       'Form_Home','Distance',
       'Highest_Home_Ticket_Price','match_month','HTeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove distance as greater than 0.05
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position',
       'Form_Home','Highest_Home_Ticket_Price','match_month','HTeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove match_month as greater than 0.05
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position',
       'Form_Home','Highest_Home_Ticket_Price','HTeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# summarize our model
regOLS_model_summary = regressor_OLS.summary()
regOLS_model_summary
fig = plt.figure(figsize=(20,24))
fig = sm.graphics.plot_partregress_grid(regressor_OLS, fig=fig)
# seaborn residual plot
sns.residplot(regressor_OLS.fittedvalues, df_sub['Attendance'], lowess=True, line_kws={'color':'r', 'lw':1})
plt.title('Residual plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals');
fig, ax = plt.subplots(figsize=(24,16))
fig = sm.graphics.influence_plot(regressor_OLS, ax=ax, criterion="cooks")
# Q-Q plot for normality
figqq= sm.qqplot(regressor_OLS.resid, line='r')
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position','Away_League_Position',
       'Form_Home','Form_Away','Distance','Temperature','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
y = np.log(y)
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove temperature variable
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position','Away_League_Position',
       'Form_Home','Form_Away','Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove form_away variable
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position','Away_League_Position',
       'Form_Home','Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','HTeam','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove HTeam variable
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Cheapest_Season_T','Home_League_Position','Away_League_Position',
       'Form_Home','Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove Cheapest_Season_T variable
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Home_League_Position','Away_League_Position',
       'Form_Home','Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove Away_League_Position variable
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Home_League_Position',
       'Form_Home','Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove Form_Home variable
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Home_League_Position',
       'Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','match_month','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove match_month variable
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Home_League_Position',
       'Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price','ATeam']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
# remove ATeam variable
x = df_sub[['Day_Type','Hol_Type','DayEve','Capacity','Average_Travelling_Fans','Home_League_Position',
       'Distance','Lowest_Home_Ticket_Price','Lowest_Away_Ticket_Price',
       'Highest_Home_Ticket_Price']]
x2 = sm.add_constant(x)
regressor_OLS = sm.OLS(endog = y, exog = x2).fit()
regressor_OLS.summary()
fig = plt.figure(figsize=(20,24))
fig = sm.graphics.plot_partregress_grid(regressor_OLS, fig=fig)
# seaborn residual plot
sns.residplot(regressor_OLS.fittedvalues, y, lowess=True, line_kws={'color':'r', 'lw':1})
plt.title('Residual plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals');
fig, ax = plt.subplots(figsize=(24,16))
fig = sm.graphics.influence_plot(regressor_OLS, ax=ax, criterion="cooks")
# Q-Q plot for normality
figqq= sm.qqplot(regressor_OLS.resid, line='r')