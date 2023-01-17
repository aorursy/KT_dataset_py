# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading full dataframe of Covid data
full_dataframe = pd.read_csv("/kaggle/input/overall-covid19/owid-covid-data (1).csv", index_col="date", parse_dates=True)
full_dataframe.head(10)
#Filtering out only Austria and Sweden and looking on first rows
c=['Austria', 'Sweden']
df_austria_sweden = full_dataframe[full_dataframe.location.isin(c)]
df_austria_sweden.head(10)
# Last rows to see also Sweden data
df_austria_sweden.tail(10)
# Import library for plotting
import matplotlib.pyplot as plt
#New cases boxplot for both locations
df_austria_sweden.boxplot(column="new_cases", by="location", figsize=(20,10))

#New deaths boxplot for both locations
df_austria_sweden.boxplot(column="new_deaths", by="location", figsize=(15,10))

#Sweden experienced more deaths per day as Austria; the interquartile ranges are bigger in the Sweden dataset
#Austria overall kept low number of deaths per day
#Renaming for shorter name
df = df_austria_sweden
#To see the trend over time for total cases by country
df.groupby('location')['total_cases'].plot(legend='True')
#To see the trend over time for total deaths by country
df.groupby('location')['total_deaths'].plot(legend='True')
#Filtering out data where there actually starts COVID cases
df = df.reset_index()
df = df[df.date>="2020-02-27"]
df.head(10)
#Check some info about data columns
df.info()
#Basic statistics about the dataset
df.describe()
#To see separately data about Austria
df_austria = df[df.location=="Austria"]
df_austria.head(10)
df_austria.info()
df_austria.describe()
#To see separately data about Sweden
df_sweden = df[df.location=="Sweden"]
df_sweden.head(10)
df_sweden.info()
df_sweden.describe()

#Decided to leave missing values as will not use data about tests for prediction anyway
#Histogram of new_deaths by country
hist1 = df.hist(column="new_deaths", by="location", bins=20, figsize=(15,7)) 
#Creating histogram for Austrian data of new deaths
df_austria.hist(column="new_deaths", bins=30, figsize=(15,7)) 
plt.xlabel('New Deaths by Day')
plt.ylabel('Day counts')
plt.title('Austria')
#Creating histogram for Swedish data of new deaths
df_sweden.hist(column="new_deaths", bins=30, figsize=(15,7)) 
plt.xlabel('New Deaths by Day')
plt.ylabel('Day counts')
plt.title('Sweden')
# import graph objects as "go"
import plotly.graph_objs as go
#Comparing total cases and total deaths per million
# create trace1 
df_summed1 = df.groupby(['location'])['new_cases_per_million'].sum().reset_index()
trace1 = go.Bar(
                x = df_summed1.location,
                y = df_summed1.new_cases_per_million,
                name = "Total Cases per Million",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(255, 174, 255, 0.5)',width=1.5)),
                text = df.location)
# create trace2
df_summed2 = df.groupby(['location'])['new_deaths_per_million'].sum().reset_index()
trace2 = go.Bar(
                x = df_summed2.location,
                y = df_summed2.new_deaths_per_million,
                name = "Total Deaths per Million",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(255, 255, 128, 0.5)',width=1.5)),
                text = df.location)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
fig.show()
#Reading mobility data
data_mobility = pd.read_csv("/kaggle/input/mobility-new/Global_Mobility_Report_0605.csv", index_col="date", parse_dates=True)
data_mobility.head(10)
#Filtering Austria and Swedend data (without regions)
c=['Austria', 'Sweden']
df_mb = data_mobility[data_mobility.country_region.isin(c) & (data_mobility.sub_region_1.isnull())]
df_mb.head(10)
#Only Austria
mb_austria =  data_mobility[(data_mobility.country_region =="Austria") & (data_mobility.sub_region_1.isnull())]
mb_austria.info()
mb_austria.describe()
#Only Sweden
mb_sweden =  data_mobility[(data_mobility.country_region =="Sweden") & (data_mobility.sub_region_1.isnull())]

mb_sweden.info()
mb_sweden.describe()
#Import library
import seaborn as sns
#Mobility data graphs - Austria
# Set the width and height of the figure
plt.figure(figsize=(20,10))

# Add title
plt.title("Daily mobility change Austria in %")

# Line chart showing changes of 'Retail and Recreation'
sns.lineplot(data=mb_austria['retail_and_recreation_percent_change_from_baseline'], label="Retail and Recreation")

# Line chart showing changes of 'Groceries and Pharmacy'
sns.lineplot(data=mb_austria['grocery_and_pharmacy_percent_change_from_baseline'], label="Groceries and Pharmacy")

# Line chart showing changes of 'Workplaces'
sns.lineplot(data=mb_austria['workplaces_percent_change_from_baseline'], label="Workplaces")

# Line chart showing changes of 'Transit Stations'
sns.lineplot(data=mb_austria['transit_stations_percent_change_from_baseline'], label="Transit Stations")

# Line chart showing changes of 'Parks'
sns.lineplot(data=mb_austria['parks_percent_change_from_baseline'], label="Parks")


# Add label for horizontal axis
plt.xlabel("date")
#Mobility data graphs - Sweden
# Set the width and height of the figure
plt.figure(figsize=(20,10))

# Add title
plt.title("Daily mobility change Sweden in %")

# Line chart showing changes of 'Retail and Recreation'
sns.lineplot(data=mb_sweden['retail_and_recreation_percent_change_from_baseline'], label="Retail and Recreation")

# Line chart showing changes of of 'Groceries and Pharmacy'
sns.lineplot(data=mb_sweden['grocery_and_pharmacy_percent_change_from_baseline'], label="Groceries and Pharmacy")

# Line chart showing changes of 'Workplaces'
sns.lineplot(data=mb_sweden['workplaces_percent_change_from_baseline'], label="Workplaces")

# Line chart showing changes of 'Transit Stations'
sns.lineplot(data=mb_sweden['transit_stations_percent_change_from_baseline'], label="Transit Stations")

# Line chart showing changes of 'Parks'
sns.lineplot(data=mb_sweden['parks_percent_change_from_baseline'], label="Parks")


# Add label for horizontal axis
plt.xlabel("date")
# reseting index column for Mobility Data - Austria
mb_austria = mb_austria.reset_index()
#looking for min. date and max. date in both files (Austria)
print('Covid dataset for Austria starts on', min(df_austria.date))
print('Covid dataset for Austria ends on',max(df_austria.date))
print('Mobility dataset for Austria starts on',min(mb_austria.date))
print('Mobility dataset for Austria ends on',max(mb_austria.date))
# reseting index column for Mobility Data - Sweden
mb_sweden = mb_sweden.reset_index()
#looking for min. date and max. date in both files (Sweden)
print('Covid dataset for Sweden starts on', min(df_sweden.date))
print('Covid dataset for Sweden ends on',max(df_sweden.date))
print('Mobility dataset for Sweden starts on',min(mb_sweden.date))
print('Mobility dataset for Sweden ends on',max(mb_sweden.date))
#Filtering out the matching data period
mb_austria = mb_austria[(mb_austria.date>="2020-02-27") & (mb_austria.date<='2020-05-27')]
mb_sweden = mb_sweden[(mb_sweden.date>="2020-02-27") & (mb_sweden.date<='2020-05-27')]
#Renaming column for joining data
mb_austria.rename(columns={'country_region':'location'}, inplace=True)
mb_austria.head(10)
#Renaming column for joining data
mb_sweden.rename(columns={'country_region':'location'}, inplace=True)
mb_sweden.head(10)
#Merging both data sets together
a=['date', 'location']
df_all_austria = pd.merge(df_austria, mb_austria, on=a)
df_all_austria.head(10)
#Merging both data sets together
a=['date', 'location']
df_all_sweden = pd.merge(df_sweden, mb_sweden, on=a)
df_all_sweden.head(10)
#############################Prediction part starts##################################################3
#Some libraries maybe needed for trying out different prediction methods
import warnings
import scipy
from datetime import timedelta

# Forceasting with decompasable model
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# For marchine Learning Approach
from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
#Importing libraries for prediction analysis and DecisionTreeClassifier and RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#Calculating some new columns for predictions
import datetime
df_all_austria1= df_all_austria

#Days from first case
df_all_austria1['days_from_start'] = (df_all_austria1['date']-min(df_all_austria1.date))/ np.timedelta64(1, 'D')

#Days from max case count


#Rolling averages for new cases
df_all_austria1['avg3days_cases']= df_all_austria1.iloc[:,4].rolling(window=3).mean()
df_all_austria1['avg3days_cases']=df_all_austria1['avg3days_cases'].replace(np.nan,df_all_austria1['new_cases'])
df_all_austria1['avg7days_cases']= df_all_austria1.iloc[:,4].rolling(window=7).mean()
df_all_austria1['avg7days_cases']= df_all_austria1['avg7days_cases'].replace(np.nan,df_all_austria1['new_cases'])
df_all_austria1['avg10days_cases']= df_all_austria1.iloc[:,4].rolling(window=10).mean()
df_all_austria1['avg10days_cases']=df_all_austria1['avg10days_cases'].replace(np.nan,df_all_austria1['new_cases'])

#Rolling averages for new deaths
df_all_austria1['avg3days_deaths']= df_all_austria1.iloc[:,6].rolling(window=3).mean()
df_all_austria1['avg3days_deaths']=df_all_austria1['avg3days_deaths'].replace(np.nan,df_all_austria1['new_deaths'])
df_all_austria1['avg7days_deaths']= df_all_austria1.iloc[:,6].rolling(window=7).mean()
df_all_austria1['avg7days_deaths']= df_all_austria1['avg7days_deaths'].replace(np.nan,df_all_austria1['new_deaths'])
df_all_austria1['avg10days_deaths']= df_all_austria1.iloc[:,6].rolling(window=10).mean()
df_all_austria1['avg10days_deaths']=df_all_austria1['avg10days_deaths'].replace(np.nan,df_all_austria1['new_deaths'])

####################Trying out decision trees###########################################
from sklearn.model_selection import train_test_split # Import train_test_split function


#split dataset in features and target variable
feature_cols = ["new_cases", "avg3days_cases", "avg7days_cases", "avg10days_cases", 
                "retail_and_recreation_percent_change_from_baseline", "grocery_and_pharmacy_percent_change_from_baseline",
                "parks_percent_change_from_baseline", "transit_stations_percent_change_from_baseline","workplaces_percent_change_from_baseline",
                "residential_percent_change_from_baseline", "days_from_start", "avg3days_deaths", "avg7days_deaths", "avg10days_deaths"]

X = df_all_austria1[feature_cols] # Features
y = df_all_austria1.new_deaths # Target variable

#Split data into traing and testing data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Cross validation
cv_scores = cross_val_score(clf, X, y, cv=3)

#Print accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Cross validation score:", np.mean(cv_scores))
feature_importance = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(feature_importance)
#####################################################################################################################
#Trial with copy/paste example from lecture


def classification_model(model, data, predictors, outcome, kfoldnumber):
    ## fit data
    model.fit(data[predictors], data[outcome])
    ## predict train-data
    predictvalues = model.predict(data[predictors])
    ##accuracy
    accuracy = metrics.accuracy_score(predictvalues, data[outcome])
    print("Accuracy: %s" % "{0:.03%}".format(accuracy))
   
    ## k-fold cross-validation
    # kfold = KFold(n_splits=5)
    kfold = KFold(n_splits=kfoldnumber)
    error = []
   
    for train, test in kfold.split(data):
        # print("------ run ------")
        # print("traindata")
        # print(train)
        # print("testdata")
        # print(test)
        ##
        ## filter training data
        train_data = df_austria[predictors].iloc[train,:]
        train_target = df_austria[outcome].iloc[train]
        ##
        # print("Traindata")
        # print(train_data)
        # print("TrainTarget")
        # print(train_target)
        ##
        ## fit data
        model.fit(train_data, train_target)
        ##
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    ##
    print("Cross Validation Score: %s" % "{0:.3%}".format(np.mean(error)))
    ##
    model.fit(data[predictors], data[outcome])
model = DecisionTreeClassifier()
outcome_var = "new_deaths"
predictor_var = ["new_cases", "new_cases_per_million", "total_cases", "total_cases_per_million", "total_deaths"]
model = LogisticRegression(solver="lbfgs")
classification_model(model, df_austria, predictor_var, outcome_var, 10)
model = DecisionTreeClassifier()
outcome_var = "new_deaths"
predictor_var = ["new_cases", "new_cases_per_million", "total_cases", "total_cases_per_million", "total_deaths"]
##
classification_model(model, df_sweden, predictor_var, outcome_var, 10)
model = RandomForestClassifier(n_estimators=100)
outcome_var = "new_deaths"
predictor_var = ["new_cases", "new_cases_per_million"]
##
classification_model(model, df_austria, predictor_var, outcome_var, 10)
#evaluating the variables
#because of very high Accuracy and very low Cross Validation Score
feature_importance = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print(feature_importance)
#total_deaths has highest feature importance from all variables --> so we used this primarly

#note: other variables in the dataset are fixed values, such as hospital_beds_per_100k, aged_65_older, density, etc. --> feature importance of 0! 
model = DecisionTreeClassifier()
outcome_var = "new_deaths"
predictor_var = ["total_deaths"]
##
classification_model(model, df_austria, predictor_var, outcome_var, 10)
model = RandomForestClassifier(n_estimators=100)
outcome_var = "new_deaths"
predictor_var = ["total_deaths"]
##
classification_model(model, df_sweden, predictor_var, outcome_var, 10)
#result: Why not to use random forests and decision trees for time series data:
#Random forests, like most ML methods, have no awareness of time. 
#On the contrary, they take observations to be independent and identically distributed. 
#This assumption is obviously violated in time series data which is characterized by serial dependence. 
#Moreover, random forests or decision tree-based methods are unable to predict a trend, i.e., they do not extrapolate."
#importing graph libraries for Tree and Forest
import graphviz
from sklearn.tree import export_graphviz
model = DecisionTreeClassifier()
outcome_var = "new_deaths"
predictor_var = ["total_deaths"]
##
classification_model(model, df_austria, predictor_var, outcome_var, 5)
##
dot_data = export_graphviz(model, out_file=None, feature_names=predictor_var, filled=True, rounded=True, special_characters=True)
graph=graphviz.Source(dot_data)
graph

######## New Prediction Analysis #### Own researched 
#### NEW CASES PREDICTION ###
#### AUSTRIA VS SWEDEN
#importing Facebook Prophet library
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
#some libraries needed for Facebook Prophet
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import datetime
from datetime import date, timedelta
from sklearn.cluster import KMeans
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import plotly.offline as py
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
#Prediction of new cases for Austria PART 1
pred_dth2 = df_austria.loc[:,["date","new_cases"]]
#cap with 0 at the bottom 
#to have no minus values in the prediction
pr_data_d = pred_dth2.tail(60)
pr_data_d.columns = ['ds','y']
pr_data_d['floor'] = 0
pr_data_d['cap'] = 850
m=Prophet(growth='logistic')
m.fit(pr_data_d)
future=m.make_future_dataframe(periods=30)
future['floor'] = 0
future['cap']= 850
forecast=m.predict(future)
cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['date','new_cases']
cnfrm.head(10)
# Visualation of the prediction of new cases in Austria for the next 30 days, taking only 60 days of past data
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='date',ylabel='new_cases')
#predicition of new cases in Sweden PART 1
pred_dth3 = df_sweden.loc[:,["date","new_cases"]]
#cap bottom to 0 in order to elimate minus values
pr_data_d = pred_dth3.tail(60)
pr_data_d.columns = ['ds','y']
pr_data_d['cap'] = 1200
pr_data_d['floor'] = 0
m=Prophet(growth='logistic')
m.fit(pr_data_d)
future=m.make_future_dataframe(periods=30)
future['cap']= 1200
future['floor'] = 0
forecast=m.predict(future)
forecast
cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['date','new_cases']
cnfrm.head(20)
# Visualation of the prediction of new cases in Sweden for the next 60 days, taking only 60 days of past data
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='date',ylabel='new_cases')
#prediction of new cases in Austria PART 2
pred_dth99 = df_austria.loc[:,["date","new_cases"]]
# cap again at 0 at bottom
# to elimate minus values
# this time taking all days for prediction (91 days why --> back to 27.02.2020 (first day of cases))
pr_data_d = pred_dth99.tail(91)
pr_data_d.columns = ['ds','y']
pr_data_d['cap'] = 1200
pr_data_d['floor'] = 0
m=Prophet(growth='logistic')
m.fit(pr_data_d)
future=m.make_future_dataframe(periods=60)
future['cap']= 1200
future['floor'] = 0
forecast=m.predict(future)
forecast
cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['date','new_cases']
cnfrm.head(20)
# Visualation of the prediction of new cases in Austria for the next 60 days, taking all days of positive value data
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='date',ylabel='new_cases')
#prediction of new cases in Sweden PART 2
pred_dth77 = df_sweden.loc[:,["date","new_cases"]]
# cap again at 0 at bottom
# to elimate minus values
# this time taking all days for prediction (91 days why --> back to 27.02.2020 (first day of cases))
pr_data_d = pred_dth77.tail(91)
pr_data_d.columns = ['ds','y']
pr_data_d['cap'] = 1000
pr_data_d['floor'] = 0
m=Prophet(growth='logistic')
m.fit(pr_data_d)
future=m.make_future_dataframe(periods=60)
future['cap']= 1000
future['floor'] = 0
forecast=m.predict(future)
forecast
cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['date','new_cases']
cnfrm.head(20)
# Visualation of the prediction of new cases in Sweden for the next 60 days
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='date',ylabel='new_cases')
### outcome prediction of new cases Austria vs Sweden 
# Austria: will decrease - might took early enough actions to prevent increase in new cases
# Sweden will increase again - maybe due to the mild approach of no real lockdown regulation?


# biased outcomes? whether using all data (91 days vs 60 days) --> changed outcomes
#using only 60 days evaluates the recent cases, 91 days evaluates also the massive peaks from March/early April 
##### TRANSIT CHANGE PREDICTION #### 
# prediction of Transit Station Mobility in Sweden 
pred_dth88 = df_all_sweden.loc[:,["date","transit_stations_percent_change_from_baseline"]]
# taking only 60 days of past data -- predecting 90 days in the future
# note: we have also analysed the prediction taking again all 91 days --> graph shown in presentation
pr_data_d = pred_dth88.tail(60)
pr_data_d.columns = ['ds','y']
m=Prophet()
m.fit(pr_data_d)
future=m.make_future_dataframe(periods=90)
forecast=m.predict(future)
forecast
cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['date','transit_stations_percent_change_from_baseline']
cnfrm.head(20)
# Visualation of the prediction of the change of transit station mobility in Sweden for the next 90 days
# using data from past 60 days 
fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='date',ylabel='transit_stations_percent_change_from_baseline')


#note: we visualed also the prediction from the data from past 91 days --> shown in presentation)
# prediction of change in transit station mobility in Austria
pred_dth55 = df_all_austria.loc[:,["date","transit_stations_percent_change_from_baseline"]]
# taking only 60 days of past data -- predicting 90 days in the future
# note: we have also analysed the prediction taking again all 91 days --> graph shown in presentation
pr_data_d = pred_dth55.tail(60)
pr_data_d.columns = ['ds','y']
m=Prophet()
m.fit(pr_data_d)
future=m.make_future_dataframe(periods=90)
forecast=m.predict(future)
forecast
cnfrm = forecast.loc[:,['ds','trend']]
cnfrm = cnfrm[cnfrm['trend']>0]
cnfrm.columns = ['date','transit_stations_percent_change_from_baseline']
cnfrm.head(20)
# Visualation of the prediction of the change of transit station mobility in Austria for the next 90 days
#using data from past 60 days 

fig = plot_plotly(m, forecast)
py.iplot(fig) 

fig = m.plot(forecast,xlabel='date',ylabel='transit_stations_percent_change_from_baseline')
# mobility in transit station will again increase, due to recent loosening of regulations 

#note: we visualed also the prediction from the data from past 91 days --> shown in presentation)