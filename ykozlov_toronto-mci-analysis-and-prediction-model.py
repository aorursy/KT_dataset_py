#!pip install pingouin
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

import folium

import calendar

from datetime import datetime

import pingouin as pg
#Option to turn off warnings 

"""

import sys

import warnings



if not sys.warnoptions:

    warnings.simplefilter("ignore")

"""
df = pd.read_csv('../input/mci.csv')

df.head()
print('Shape:',df.shape)

print('-------------INFO---------------')

print(df.info())

print('-------------COLUMNS---------------')

print(df.columns)

print('-------------INDEX---------------')

print(df.index)
print('Duplicates:',df.duplicated().any())
print('Missing Data')

df.isnull().sum().sort_values(ascending=False)
#convert to proper format

from datetime import datetime

df['Occurrence_Date'] = df['Occurrence_Date'].astype('datetime64[ns]') 

df['Occurrence_year'] = df['Occurrence_year'].astype('Int32') 

df['Occurrence_Month'] = df['Occurrence_Month'].astype(str) 

df.info()
df['Occurrence_year'] = df.apply(

    lambda row: row['Occurrence_Date'].year if np.isnan(row['Occurrence_year']) else row['Occurrence_year'],

    axis=1

)


# <4 cux nan = 3

df['Occurrence_Month'] = df.apply(

    lambda row: calendar.month_name[row['Occurrence_Date'].month]  if len(row['Occurrence_Month'])<4 else row['Occurrence_Month'],

    axis=1

)

df.head()
# df has its index, so drop csv index

df = df.drop(columns=['Index_'])

df.head()
print('Missing Data')

df.isnull().sum().sort_values(ascending=False)
#include years from 2014 2018 

df = df[df.Occurrence_year >=2014]

df = df[df.Occurrence_year <=2018]
df_ct = pd.crosstab(df.Type, df.Occurrence_year, margins=True)

df_ct
df_ct_plot = pd.crosstab(df.Type, df.Occurrence_year)

df_ct_plot.plot(kind='bar', stacked=False, rot = 0, figsize=(10, 10));

plt.ylabel('Total')

plt.title('MCI Type by Year')
# remove row all from min/max analysis and create mci per year 

mciPerYear = df_ct.loc['All']

df_ct = df_ct.iloc[:-1,:]

mciPerYear = mciPerYear[:-1]

mciPerYear
print('Leading Crime')

print('-----------------')

for i in range(0,df_ct.shape[1]-1): 

    row = df_ct.iloc[:,i][df_ct.iloc[:,i] == df_ct.iloc[:,i].max()]

    print(row)

    print('-----------------')
print('Least occuring Crime')

print('-----------------')

for i in range(0,df_ct.shape[1]-1): 

    row = df_ct.iloc[:,i][df_ct.iloc[:,i] == df_ct.iloc[:,i].min()]

    print(row)

    print('-----------------')
description = df_ct[0:-1].describe()

description

df_ct[0:-1].kurtosis()
for i in range(0, df_ct.shape[0]):

    print (df_ct.iloc[i, 0:-1])

    sns.distplot(df_ct.iloc[i, 0:-1])

    #df_ct.iloc[i, 0:-1].hist(bins=5)

    plt.show()
#Transpoded crosstab & find correlation

df_ct_t =  pd.crosstab(df.Type, df.Occurrence_year, margins=False).reindex()

df_ct_t = df_ct_t.T

type_corr = df_ct_t.corr()

type_corr
#Correlation heatmap

plt.figure(figsize=(10,7))

ax = sns.heatmap(type_corr, annot=True) #notation: "annot" not "annote"

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)

 
df_annual = pd.concat([pd.Series(mciPerYear.index, name='Year'), 

                       pd.Series(mciPerYear.values, name='Type')], axis=1).reset_index()

df_annual = df_annual.drop(columns=['index'])

df_annual
plt.bar('Year', height='Type', width=0.8, align='center', data=df_annual)

plt.ylabel('Total')

plt.title('Annual MCI')
fig, ax = plt.subplots(figsize=(6,6))

sns.lineplot(x='Year', y='Type', data=df_annual, color='b')

ax.set_title('Annual Total MCI')

plt.show()
df_hood_ct = pd.crosstab(df.Neighbourhood, df.Occurrence_year, margins=True)

df_hood_ct.sort_values("All", axis = 0, ascending = False,

                 inplace = True) 

df_hood_ct.head(6)
top5dangerous = df_hood_ct.iloc[1:6,:5].T

fig, ax = plt.subplots(1,1,figsize=(10,7))

ax.plot(top5dangerous)

ax.legend(top5dangerous.columns.to_list(), loc="best", bbox_to_anchor = (1,1))
df_hood_ct.tail(5)
top5safest = df_hood_ct.iloc[-5:,:5].T

fig, ax = plt.subplots(1,1,figsize=(10,7))

ax.plot(top5safest)

ax.legend(top5safest.columns.to_list(), loc="best", bbox_to_anchor = (1,1))
df_hood_ct_var = df_hood_ct;

df_hood_ct_var['min'] = df_hood_ct_var.iloc[:,:-1].min(axis=1)

df_hood_ct_var['max'] = df_hood_ct_var.iloc[:,:-2].max(axis=1)

df_hood_ct_var['Delta'] = 100*(df_hood_ct_var['max'] - df_hood_ct_var['min'])/df_hood_ct_var['All']

df_hood_ct_var.sort_values("Delta", axis = 0, ascending = False,

                 inplace = True) 

df_hood_ct_var.head(5)
top5volatile = df_hood_ct_var.iloc[0:5,:5].T

fig, ax = plt.subplots(1,1,figsize=(10,7))

ax.plot(top5volatile)

ax.legend(top5volatile.columns.to_list(), loc="best", bbox_to_anchor = (1,1))
#lets take one step down the list

top5volatile = df_hood_ct_var.iloc[1:6,:5].T

fig, ax = plt.subplots(1,1,figsize=(10,7))

ax.plot(top5volatile)

ax.legend(top5volatile.columns.to_list(), loc="best", bbox_to_anchor = (1,1))
#lets take one more step down the list
top5volatile = df_hood_ct_var.iloc[2:7,:5].T

fig, ax = plt.subplots(1,1,figsize=(10,7))

ax.plot(top5volatile)

ax.legend(top5volatile.columns.to_list(), loc="best", bbox_to_anchor = (1,1))
import geopandas as gpd

sns.set(style="darkgrid")
#group data by Neighbourhood and count 

df_gp = df.groupby(['Hood_ID','Neighbourhood']).count()[['Type']]

df_gp= df_gp.reset_index()

df_gp
location = '../input/Neighbourhoods/Neighbourhoods.shp'

hoods = gpd.read_file(location)

hoods.sample(5)
merged = hoods.set_index('FIELD_7').join(df_gp.set_index('Neighbourhood'))

merged = merged.reset_index()

merged[['FIELD_7', 'FIELD_11', 'FIELD_12', 'geometry', 'Type']].sample(5)

merged.head(5)
fig, ax = plt.subplots(1, figsize=(40, 20))

ax.axis('off')

ax.set_title('MCI by Neighbourhood, Toronto', fontdict={'fontsize': '40', 'fontweight' : '3'})

color = 'Reds'

vmin, vmax = 0, 231

sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)

cbar.ax.tick_params(labelsize=20)

merged.plot('Type', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(40,20))
df_hood_income = pd.read_csv('../input/hood_income.csv')

#convert to lower case for for mergin

df_hood_income['Neighborhood']=df_hood_income['Neighborhood'].str.lower()

df_hood_income.head()
#clrear hood numbers, because other table doesnt have numbers

merged['neighbourhood'] = merged['FIELD_8'].str.replace(' \(.+\)', '').str.lower()
merged.head()
#add income to table

merged_income = merged.set_index('neighbourhood').join(df_hood_income.set_index('Neighborhood'))

merged_income = merged_income.reset_index()

merged_income.head()
#plot a map

fig, ax = plt.subplots(1, figsize=(40, 20))

ax.axis('off')

ax.set_title('Income by Neighbourhood, Toronto', fontdict={'fontsize': '40', 'fontweight' : '3'})

color = 'Blues'

vmin, vmax = 0, 231

sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)

cbar.ax.tick_params(labelsize=20)

merged_income.plot('HoodAveIncome', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(40,20))
print('Correlation HoodIncome and MCI')

merged_income['Type'].corr(merged_income['HoodAveIncome']) 
pg.corr(x=merged_income['Type'], y=merged_income['HoodAveIncome'])
sns.lmplot(data=merged_income, x='HoodAveIncome', y='Type')
merged_income['Ratio'] = 100 * merged_income['Type'] / merged_income['HoodAveIncome']

merged_income.head()
merged_income.sort_values("Ratio", axis = 0, ascending = False,

                 inplace = True)

merged_income[['FIELD_7','Type','HoodAveIncome','Ratio']].head(5)
merged_income[['FIELD_7','Type','HoodAveIncome','Ratio']].tail(11)
#plot a map

fig, ax = plt.subplots(1, figsize=(40, 20))

ax.axis('off')

ax.set_title('Crime to Income ratio by Neighbourhood, Toronto', fontdict={'fontsize': '40', 'fontweight' : '3'})

color = 'Oranges'

vmin, vmax = 0, 231

sm = plt.cm.ScalarMappable(cmap=color, norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

cbar = fig.colorbar(sm)

cbar.ax.tick_params(labelsize=20)

merged_income.plot('Ratio', cmap=color, linewidth=0.8, ax=ax, edgecolor='0.8', figsize=(40,20))
#prepare population data

population = pd.read_csv('../input/pop_estimate.csv')

population
plt.bar('Year', height='Population', width=0.8, align='center', data=population)
population.info()
#remove years that are not used

#population = population[13:]

#population = population[:-2]

population = population[(population['Year'] >=2014) & (population['Year'] <=2018)]

population = population.reset_index(drop=True)

population
#add population to annual homicide dataset

df_annual_percap = df_annual

df_annual_percap = df_annual_percap.join(population.Population)
df_annual_percap
#number of homicide per 1,000 and  in percent

df_annual_percap['PerCap%'] = round(100*df_annual_percap.Type/df_annual_percap.Population,4)

#df_annual_percap['PerCap1K%'] = round(100*df_annual_percap.Type/df_annual_percap.Population,2)

df_annual_percap['Pop100'] = round(df_annual_percap.Population/100,2)

df_annual_percap
#prepare unemployment rate

unemployment = pd.read_csv('../input/unemployment_ontario.csv')

unemployment
unemployment.info()
plt.bar('Year', height='UnempRate', width=0.8, align='center', data=unemployment)
#remove unused years 

unemployment = unemployment[(unemployment['Year'] >=2014) & (unemployment['Year']<=2018)]

unemployment = unemployment.reset_index(drop=True)

unemployment
df_annual_percap = df_annual_percap.join(unemployment.UnempRate)

df_annual_percap
#Load and prepare average income data

aveIncome = pd.read_csv('../input/average_income.csv')

aveIncome
plt.bar('Year', height='AverageIncome', width=0.8, align='center', data=aveIncome)
#remove unused years 

aveIncome = aveIncome[(aveIncome['Year'] >=2014) & (aveIncome['Year']<=2018)]

aveIncome = aveIncome.reset_index(drop=True)

aveIncome
df_annual_percap = df_annual_percap.join(aveIncome.AverageIncome)

df_annual_percap
print('Correlation MCI per Capita and Population')

df_annual_percap['Population'].corr(df_annual_percap['Type']) 
pg.corr(x=df_annual_percap['Population'], y=df_annual_percap['Type'])
sns.lmplot(data=df_annual_percap, x='Population', y='Type')
print('Correlation MCI per Capita and Unemployment')

df_annual_percap['UnempRate'].corr(df_annual_percap['PerCap%']) 

pg.corr(x=df_annual_percap['UnempRate'], y=df_annual_percap['PerCap%'])
print('Correlation MCI and Unemployment')

df_annual_percap['UnempRate'].corr(df_annual_percap['Type']) 
pg.corr(x=df_annual_percap['UnempRate'], y=df_annual_percap['Type'])
sns.lmplot(data=df_annual_percap, x='UnempRate', y='Type')
fig, ax = plt.subplots(figsize=(6,6))

sns.lineplot(x='Year', y='PerCap%', data=df_annual_percap, color='b')

ax.set_title('MCI per capita')

plt.show()
fig, ax = plt.subplots(figsize=(6,6))

sns.lineplot(x='Year', y='UnempRate', data=df_annual_percap, color='r')

sns.lineplot(x='Year', y='PerCap%', data=df_annual_percap, color='b')

ax.legend(['Unemployment', 'MCI per Capita'], facecolor='w')

ax.set_title('Unemployment rate vs MCI per capita')

plt.show()
print('Correlation MCI per Capita and Average Income')

df_annual_percap['AverageIncome'].corr(df_annual_percap['PerCap%']) 

pg.corr(x=df_annual_percap['AverageIncome'], y=df_annual_percap['PerCap%'])
print('Correlation MCI and Average Income')

df_annual_percap['AverageIncome'].corr(df_annual_percap['Type']) 
pg.corr(x=df_annual_percap['AverageIncome'], y=df_annual_percap['Type'])
sns.lmplot(data=df_annual_percap, x='AverageIncome', y='Type')
fig, ax = plt.subplots(figsize=(6,6))

sns.lineplot(x='Year', y='AverageIncome', data=df_annual_percap, color='r')

sns.lineplot(x='Year', y='Type', data=df_annual_percap, color='b')

ax.legend(['Average Income', 'MCI'], facecolor='w')

ax.set_title('Average Incomerate vs MCI')

plt.show()
#Linear regression function

def LinearPredict(x,y,years):

    #reshape data

    x = x.reshape((-1, 1))

    y = y.reshape((-1, 1))

    #build model and train

    model = LinearRegression()

    model.fit(x, y)

    #evaluate error

    r_sq = model.score(x, y)

    #make predictions

    y_pred = model.predict(years)

    

    print('intercept:', model.intercept_)

    print('slope:', model.coef_)

    print('coefficient of determination:', r_sq)

    print('Prediction of homicides in', years, 'will be', np.round(y_pred,0))    

    

    return model.coef_, model.intercept_, y_pred

#Years to predict

yearsToPredict = np.array([[2019],[2020]])
#get MCI data

x = df_annual['Year'].values

y = df_annual['Type'].values

m,b, pred = LinearPredict(x,y,yearsToPredict)

#print("data", m,b,pred)
#Plot Results

plt.scatter(df_annual_percap['Year'], df_annual_percap['Type'], marker='o')

y =  m[0]*df_annual_percap['Year']+b

x = df_annual_percap['Year']



plt.plot(x, y, '-r')

plt.plot(yearsToPredict,pred,'-b')    
# Reshape Data

x = (df_annual['Year'].values).reshape(-1, 1)

y = df_annual['Type'].values



#build and train model

poly_reg = PolynomialFeatures(degree=2)

x_poly = poly_reg.fit_transform(x)

pol_reg = LinearRegression()

pol_reg.fit(x_poly, y)



#make prediction

pred = pol_reg.predict(poly_reg.fit_transform(yearsToPredict))



# Visualizing the Polymonial Regression results



plt.scatter(x, y, color='red')

plt.plot(x, pol_reg.predict(poly_reg.fit_transform(x)), color='blue')  

plt.plot(yearsToPredict,pred,'-r')

plt.show()



print('Polynomical Predict for ', yearsToPredict , np.round(pred,0))

#build model

poly_reg = PolynomialFeatures(degree=3)

x_poly = poly_reg.fit_transform(x)

pol_reg = LinearRegression()

#train

pol_reg.fit(x_poly, y)

#predict 

pred = pol_reg.predict(poly_reg.fit_transform(yearsToPredict))

# Visualizing the Polymonial Regression results

plt.scatter(x, y, color='red')

plt.plot(x, pol_reg.predict(poly_reg.fit_transform(x)), color='blue')  

plt.plot(yearsToPredict,pred,'-r')

plt.show()



print('Polynomical Predict for ', yearsToPredict , pred)
from sklearn import svm
x = (df_annual['Year'].values).reshape((-1, 1))

y = (df_annual['Type'].values)

yearsToPredict
clf = svm.SVR(gamma='auto')

clf.fit(x,y)





resSVR = clf.predict(yearsToPredict)



resSvc_df =pd.DataFrame({'Predict': resSVR[:]})

resSvc_df
#print dataset again

df_ct = df_ct.iloc[:, :-1]

df_ct
def LinearRegGroup(x,y,name):

    yearsToPredict = np.array([[2019],[2020]])

    

    x=x.reshape((-1,1))

    y=y.reshape((-1,1))

    

    model = LinearRegression()

    model.fit(x, y)

    #evaluate error

    r_sq = model.score(x, y)

    #make predictions

    y_pred = model.predict(yearsToPredict)

    

    print('*********************')

    print(name)

    print('*********************')

    

    print('intercept:', model.intercept_)

    print('slope:', model.coef_)

    print('coefficient of determination:', r_sq)

    print('Prediction of',name,'in', yearsToPredict, 'will be', np.round(y_pred,0))    

    

    m = model.coef_

    b =  model.intercept_

             

    plt.scatter(x, y, marker='o')

    y1 =  m[0]*x+b

    x1 = x



    plt.plot(x1, y1, '-r')

    plt.plot(yearsToPredict,y_pred,'-b') 

    plt.show()
#cycle through each row and make prediction

for i in range(len(df_ct)):

    name = df_ct.iloc[i:i+1].index[0]

    y = df_ct.iloc[i:i+1].values

    x =df_ct.columns.values

    LinearRegGroup(x,y,name)
def PolynomialRegressionGroup(x,y,name, d):

    yearsToPredict = np.array([[2019],[2020]])

        

    x=x.reshape((-1,1))

    y=y.reshape((-1,1))

    

    poly_reg = PolynomialFeatures(degree=d)

    x_poly = poly_reg.fit_transform(x)

    pol_reg = LinearRegression()

    pol_reg.fit(x_poly, y)



    pred = pol_reg.predict(poly_reg.fit_transform(yearsToPredict))

        

    print('*********************')

    print(name)

    print('*********************')

        

    # Visualizing the Polymonial Regression results



    plt.scatter(x, y, color='red')

    plt.plot(x, pol_reg.predict(poly_reg.fit_transform(x)), color='blue')  

    plt.plot(yearsToPredict,pred,'-r')

    plt.show()



    print('Polynomical Predict for ', yearsToPredict , np.round(pred,0))
for i in range(len(df_ct)):

    name = df_ct.iloc[i:i+1].index[0]

    y = df_ct.iloc[i:i+1].values

    x =df_ct.columns.values

    PolynomialRegressionGroup(x,y,name,2)
for i in range(len(df_ct)):

    name = df_ct.iloc[i:i+1].index[0]

    y = df_ct.iloc[i:i+1].values

    x =df_ct.columns.values

    PolynomialRegressionGroup(x,y,name,3)
#Set Occurance date as index

df_ts = df.set_index('Occurrence_Date')

df_ts
df_ts['Occurrence_year']

df_ts[['Occurrence_year']]

#Resample monthly and weekly

#We need to drop NA because during resampling N/A might be added 

df_ts_monthly = df_ts.Occurrence_year.resample('M').count()

df__ts_monthly = df_ts_monthly.dropna()

df_ts_weekly = df_ts.Occurrence_year.resample('W').count()

df_ts_weekly = df_ts_weekly.dropna()
#Autocorrelation formatted plot

def AutocorrelationFormatedPlot(stock, symbol):

    TimePlot = pd.plotting.autocorrelation_plot(stock, color='r')

    title = symbol + " Autocorrelation Graph"

    TimePlot.set_title(title)

    TimePlot.set_ylabel("Correlation")

    TimePlot.set_xlabel("Lag Month")

    None
AutocorrelationFormatedPlot(df_ts_monthly,"MCI Monthly")
AutocorrelationFormatedPlot(df_ts_weekly,"MCI Monthly")
plt.figure(figsize=(7,5))

plt.ylabel('Total')

plt.title('MCI Annual Trend')

plt.plot(df_ts_monthly)
#during re-sampling last few days are shifted to 2019, therefore, we remove last point from the plot [:-1]

plt.figure(figsize=(12,5))

plt.ylabel('Total')

plt.title('MCI Weekly Trend')

#plt.xticks(np.arange(12), calendar.month_name[1:13], rotation=20)

plt.plot(df_ts_weekly[:-1])
monthlyTotal = df_ts_monthly.loc['2014'].values

for val in range (2015,2019):

    monthlyTotal = monthlyTotal + df_ts_monthly.loc[str(val)].values
plt.figure(figsize=(7,5))

plt.ylabel('Total')

plt.title('MCI Total Per Month')

plt.xticks(np.arange(12), calendar.month_name[1:13], rotation=20)

plt.plot(monthlyTotal)
print('Monthly Median')

df_ts_monthly.median()
df_ts_monthly_ct = pd.crosstab(df_ts.Occurrence_Month, df_ts.Occurrence_year)

df_ts_monthly_ct['Median'] = df_ts_monthly_ct.median(axis=1)

df_ts_monthly_ct.sort_values('Median',axis = 0, ascending = False,

                 inplace = True)

df_ts_monthly_ct
df_ts_monthly_ct['Median'].plot(kind='bar', stacked=False, rot = 0, figsize=(10, 7));
df_ts_monthly_ct['Median'].plot(stacked=False, rot = 0, figsize=(10, 7));
x = df[['Lat', 'Long']].values

print(type(x))

x
y = df[['Type']].values.flatten()

print(type(y))

y
#Split data into training and testing datasets

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
#Evaluation function

def Evaluate(predicted, y_test):

    true_pred = np.sum(predicted == y_test)

    total_pred = predicted.shape[0]

    print('True predictions', true_pred, 'out of', total_pred)

    print('Percent of correct predictions', round(100*true_pred/total_pred,2), '%')

    
from sklearn.neighbors import KNeighborsClassifier
#Define Model

knc = KNeighborsClassifier(n_neighbors =15) # 	n_neighbors : int, optional (default = 5)

#Train

knc = knc.fit(x_train, y_train)

#Predict

resKN = knc.predict(x_test)

#Evaluate

Evaluate(resKN,y_test)

#convert resKN to dataframe 

resKN_df =pd.DataFrame({'knc': resKN[:]})



resKN_df.head(10)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=20)

dtc = dtc.fit(x_train,y_train)

resDtc = dtc.predict(x_test)

Evaluate(resDtc,y_test)

dtc_df =pd.DataFrame({'dtc': resDtc[:]})

dtc_df.head()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)

lr = lr.fit(x_train, y_train)

resLr = lr.predict(x_test)

Evaluate(resLr,y_test)



lr_df =pd.DataFrame({'lr': resLr[:]})

lr_df.head()
df_predict = resKN_df.join(dtc_df)

df_predict = df_predict.join(lr_df)

#df_predict.head()
df_predict_votes = df_predict.copy()
df_predict_votes.head()
df_predict_votes['vote'] = 'none'

df_predict_votes.head()
for items, vals in df_predict_votes.iterrows():    

    if(vals['dtc'] == vals['lr']):

        vals['vote'] = vals['dtc']

    else:

        vals['vote'] = vals['knc']

    

    #print(items, vals['knc'],vals['dtc'],vals['lr'], vals['vote'])
y_voted =df_predict_votes[['vote']].values.flatten()

print(type(y_voted))

y_voted
Evaluate(y_voted,y_test)