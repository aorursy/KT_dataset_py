import pandas as pd



# read in our data 

bmi_data = pd.read_csv("../input/eating-health-module-dataset//ehresp_2014.csv")

nyc_census = pd.read_csv("../input/new-york-city-census-data/nyc_census_tracts.csv")
# remove rows where the reported BMI is less than 0 (impossible)

bmi_data = bmi_data[bmi_data['erbmi']>0]
X = bmi_data[['euexfreq', 'euwgt', 'euhgt', 'ertpreat']]

y = bmi_data['erbmi']

# fit a glm model

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X,y)
import matplotlib.pyplot as plt

y_pred = reg.predict(X)

residual = y - y_pred

plt.scatter(y_pred,residual)
# examine our model

print('coefficient = ' + str(reg.coef_))

print('intercept = ' + str(reg.intercept_))
# added-variable plots for our model
nyc_census.head()
nyc_census.columns
selected_nyc_census = nyc_census[['Unemployment', 'Hispanic', 'White', 'Black', 'Native']]

selected_nyc_census = selected_nyc_census.dropna()

X = selected_nyc_census[['Hispanic', 'White', 'Black', 'Native']]

y = selected_nyc_census['Unemployment']
# fit a glm model

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X,y)
y_pred = reg.predict(X)

residual = y - y_pred

plt.scatter(y_pred,residual)
# examine our model

print('coefficient = ' + str(reg.coef_))

print('intercept = ' + str(reg.intercept_))