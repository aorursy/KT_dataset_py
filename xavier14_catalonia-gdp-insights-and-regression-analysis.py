import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns 

% matplotlib inline
df=pd.read_csv('../input/GDP_CAT.csv')
# Let's start of years from 2000



df = df.iloc[::-1]  # We can easily do this step with the iloc funtion
df = df.set_index('Year') # We establish  the YEARS as our index 
df # We can examine our df
# Let's check the GDP output evolution in the series



df.GDP.plot(figsize=(20,7), kind='area', legend=False, use_index=True, grid=True, color='aqua')



SIZE = 22

plt.rc('xtick', labelsize=SIZE)                                       # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE)                                       # fontsize of the tick Y labels 



plt.xlabel('YEARS', size=22)                                          # x title label 

plt.ylabel('GDP in Millions of €', size=22)                           # y title label 

plt.title('Total GDP of Catalonia (2000-2016)',size=30)               # plot title label                              

plt.legend(loc='upper left', prop={'size': 20})                       # legend location and size
# With pandas we can easily generate a series of the **percentual change of the accumulated GDP**. Let's visualize it :



# We use the .pct_change method to create a series in our DataFrame of the GDP percentual change on year basis



df['pct_change']=(df.GDP.pct_change()*100).plot(figsize=(20,7), kind='line', legend=False, use_index=True, 

                grid=True, color='aqua')



SIZE = 18

plt.rc('xtick', labelsize=SIZE)                            # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE)                            # fontsize of the tick Y labels 

plt.axhline(y=0)                                           # we create a line for the 0% (Y=0)



plt.xlabel('YEARS', size=22)                               # x title label 

plt.ylabel('Yearly GDP growth %', size=22)                 # y title label 

plt.title('Yearly GDP growth %', size=30)                  # plot title label                              
df.plot(y= 'GDP', x ='Domestic demand',kind='hexbin',gridsize=45, 

        sharex=False,colormap='cubehelix',figsize=(15,5))



SIZE1 = 14

plt.rc('xtick', labelsize=SIZE1)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE1)    # fontsize of the tick Y labels 



plt.xlabel('Domestic demand', size=16)

plt.ylabel('GDP', size=16)

plt.title('Hexbin of Domestic Demand vs GDP', size=20)
df.plot(y= 'GDP', x ='Exports goods and services',kind='hexbin',gridsize=45, 

        sharex=False,colormap='cubehelix',figsize=(15,5))



plt.rc('xtick', labelsize=SIZE1)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE1)    # fontsize of the tick Y labels



plt.xlabel('Exports goods and services', size=16)

plt.ylabel('GDP', size=16)

plt.title('Hexbin of Exports goods and services VS GDP', size=20)
df['Const.'].plot(figsize=(20,7), kind='area', legend=False, use_index=True, grid=True, color='darkcyan')



SIZE = 22

plt.rc('xtick', labelsize=SIZE)                         # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE)                         # fontsize of the tick Y labels 



plt.xlabel('YEARS', size=26)                            # x title label 

plt.ylabel('Millions of €', size=22)                    # y title label 

plt.title('Construction (2000-2016)',

          size=30)                                      # plot title label                              
# we create and plot a new series called Cons_per_GDP



Cons_per_GDP=df['Const.']/df.GDP*100



Cons_per_GDP.plot(figsize=(20,7), kind='bar', legend=False, use_index=True, grid=True, color='darkcyan')



SIZE = 18

plt.rc('xtick', labelsize=SIZE)                                        # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE)                                        # fontsize of the tick Y labels 



plt.xlabel('YEARS', size=20)                                           # x title label 

plt.ylabel('Ratio with the GDP %', size=20)                            # y title label 

plt.title('Construction as a ratio of the overall GDP (2000-2016)',

          size=26)                                                     # plot title label                         
# We create a new series called Exports_per_GDP

Exports_per_GDP=df['Total exports goods and services']/df.GDP*100

Exports_per_GDP.plot(figsize=(20,8), kind='bar', use_index=True, grid=True, color='b')

SIZE = 20

plt.rc('xtick', labelsize=SIZE)                     # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE)                     # fontsize of the tick Y labels 

plt.xticks(rotation=0)



plt.xlabel('YEARS', size=22)                        # x title label 

plt.ylabel('Ratio with the GDP %', size=22)         # y title label 

plt.title('Total exports of goods and services as a ratio of the overall GDP (2000-2016)',

          size=28)                                  # plot title label                              
# We add both columns to our DataFrame



df['Cons_per_GDP']=Cons_per_GDP

df['Exports_per_GDP']=Exports_per_GDP



df.plot(y= 'Cons_per_GDP', x ='Exports_per_GDP',kind='hexbin',gridsize=45, 

        sharex=False,colormap='cubehelix',figsize=(15,5))



plt.rc('xtick', labelsize=SIZE1)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE1)    # fontsize of the tick Y labels



plt.xlabel('Ratio of total exports with the GDP %', size=16)

plt.ylabel('Ratio of construction with the GDP %', size=16)

plt.title('Hexbin of the GDP ratio of exports vs GDP ratio of construction %', size=16)
# We create the series of the GDP ratio of Domestic Demand without construction 



Domestic_Demand_per_GDP_wc=(df['Domestic demand']-df['Const.'])/df.GDP*100

df['Domestic_Demand_per_GDP_wc']=Domestic_Demand_per_GDP_wc



df.plot(y='Domestic_Demand_per_GDP_wc', x ='Exports_per_GDP',kind='hexbin',gridsize=45, 

        sharex=False,colormap='cubehelix',figsize=(15,5))



plt.rc('xtick', labelsize=SIZE1)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE1)    # fontsize of the tick Y labels



plt.xlabel('Ratio of total exports with the GDP %', size=16)

plt.ylabel('Ratio of Domestic Demand without cons. with the GDP %', size=16)

plt.title('Hexbin of the GDP ratio of exports VS GDP ratio of domestic demand without cons. %', size=16)
# We set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 10))



plt.title('Pearson Correlation')



# We draw the heatmap using seaborn

sns.heatmap(df.corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu",linecolor='black', annot=True)
# We create our new series Trade openness adding the total exports and the total imports 



df['trad_op']= (df['Total exports goods and services']+df['Total imports goods and services'])
# We add both series to our DataFrame:



df['trad_op']= (df['trad_op']/df.GDP*100)         # We calculate the GDP ratio

df['pct_change']=(df.GDP.pct_change()*100)        
# Let's plot both trends in the same graph using the secondary_y method



ax=df['trad_op'].plot(figsize=(20,7), kind='line', legend=False, grid=False, use_index=True,

                      color='aqua')

ax1=df['pct_change'].plot(secondary_y=True, figsize=(20,7), kind='line', legend=False, 

                          use_index=True,grid=False, color='r')



plt.axhline(y=0)

ax.legend(loc='lower left', prop={'size': 28})                  # legend location and size

ax1.legend (loc='lower right', prop={'size': 28})               # legend location and size



SIZE = 22

plt.rc('xtick', labelsize=SIZE)                                     # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE)                                     # fontsize of the tick Y labels 

                

ax.set_xlabel('YEAR',size=26)                                       # We arrange our axis labels

ax1.set_xlabel('YEAR',size=26) 

ax.set_ylabel('% Ratio of Trade Openness with the GDP', size=26)

ax1.set_ylabel('GDP growth in %', size=32)



plt.title('Trade openness and GDP growth', size=30)                 # plot title label  

plt.show() 
# Let's change the names:

ax=df['Cons_per_GDP'].plot(figsize=(20,7), kind='line', legend=False, use_index=True, color='aqua')

ax1=df['pct_change'].plot(secondary_y=True, figsize=(20,7), kind='line', legend=False,  use_index=True, grid=False, color='red')



plt.axhline(y=0)

ax.legend(loc='lower center', prop={'size': 28})                

ax1.legend (loc='lower left', prop={'size': 28})                 

                               

ax.set_xlabel('YEAR',size=26)                                              

ax1.set_xlabel('YEAR',size=26) 

ax.set_ylabel('Construction Ratio of the GDP in %', size=26)

ax1.set_ylabel('GDP growth in %', size=32)



plt.title('Construction weight in the GDP and GDP growth', size=30)        

plt.show() 
from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as sm

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
X=['Consumer expenditure household','Consumer public adm','Equip. Goods others','Const.',

   'Total exports goods and services','Total imports goods and services']
# We create our matrix of regressors (independent variables)

X=df[X]



# We create our dependant variable

y=df.GDP
# We create a linear regression object

lm = LinearRegression()
# We fit our model

lm.fit(X,y)
# From the stats models we built our linear model.

model=lm.fit(X,y)



result = sm.ols(formula="y ~ X", data=df).fit()

print(result.summary())
p=lm.predict(X)
plt.figure(num=3, figsize=(20, 10), dpi=90, facecolor='w', edgecolor='aqua')



sns.regplot(y, p, data=df, marker='*', scatter_kws={"s": 350})



SIZE2=20  

plt.rc('xtick', labelsize=SIZE2)    # fontsize of the tick X labels 

plt.rc('ytick', labelsize=SIZE2)    # fontsize of the tick Y labels





plt.title('Predicted GDP vs Actual GDP', size=30)

plt.xlabel('Actual value', size=26)

plt.ylabel('Predicted value', size=26)

plt.show()
Errors=(y-p)



print(Errors)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=4)



print ('Fit a model X_train, and calculate MSE with y_train:', np.mean((y_train - lm.predict(X_train)) ** 2))

print ('Fit a model X_train, and calculate MSE with x_test, Y_test:', np.mean((y_test - lm.predict(X_test)) ** 2))
y_train - lm.predict(X_train) 
y_test - lm.predict(X_test)