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
#import modules



#import libraries for data handling

import os

import pandas as pd # for dataframes

import numpy as np



#import for visualization

import seaborn as sns # for plotting graphs

import matplotlib

import matplotlib.pyplot as plt # for plotting graphs

%matplotlib inline



#import for Linear regression

from sklearn.linear_model import LinearRegression





#importing plotly and cufflinks in offline mode

import cufflinks as cf

import plotly.offline

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



#import warnings

import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)
basetable1 = pd.read_excel('/kaggle/input/bollywood/bollywood.xlsx')

basetable1.head(5)
basetable1.info()
# Make a copy for analysis and chages without disturbing the base data

df1 = basetable1.copy()
# Movies Count Trend

df1[['Year', 'Title']].groupby(['Year']).count().iplot(kind='bar',xTitle='Year', yTitle='Movie Count')
# Movie Revenue & Budget trend

df1[['Year', 'BO Rev', 'Total Budget']].groupby(['Year']).sum().iplot(kind='bar',xTitle='Year', yTitle='Movie Revenue/Budget', title = 'Movie Revenue & Budget trend')
# Revenue and budget trend

df1[['Year', 'BO Rev', 'Total Budget']].groupby(['Year']).sum()
# Production wise Revenue and Budget

df1[['Big Production', 'BO Rev', 'Total Budget']].groupby(['Big Production']).sum()
df1[['Big Production', 'BO Rev', 'Total Budget']].groupby(['Big Production']).sum().iplot(kind='bar',xTitle='Big Production', yTitle='Movie Revenue', title = 'Production wise Revenue and Budget')
# Movie count trend

df1[['Big Production', 'BO Rev']].groupby(['Big Production']).count().iplot(kind='bar',xTitle='Big Production', yTitle='Movie Count', title = 'Movie count by Production')
df1[['Big Production', 'BO Rev']].groupby(['Big Production']).count()
# Insider/Outsider: Revenue and Budget

df1[['Insider', 'BO Rev', 'Total Budget']].groupby(['Insider']).sum().iplot(kind='bar',xTitle='Insider', yTitle='Movie Revenue/Budget', title = 'Insider/Outsider: Revenue and Budget')
df1[['Insider', 'BO Rev']].groupby(['Insider']).sum()
# Insider/Outsider: Movie Count

df1[['Insider', 'BO Rev']].groupby(['Insider']).count().iplot(kind='bar',xTitle='Insider', yTitle='Movie Count',title = 'Insider/Outsider: Movie Count' )
df_canvas = df1.copy()

df_canvas.head()

df_canvas.info()
df_canvas["Insider"] = df_canvas["Insider"].astype(str)

df_canvas["Big Production"] = df_canvas["Big Production"].astype(str)

df_canvas.info()

#Converted Categorical variables to str for discret categorization.
import plotly.express as px



#Big Production - Budget vs Profit

fig = px.scatter(df_canvas, x="Total Budget", y="profit", color="Big Production",

    size="BO Rev", size_max=45, title = 'Big Production - Budget vs Profit')



fig.update_layout(legend=dict(

    orientation="h",

    yanchor="bottom",

    y=1.02,

    xanchor="right",

    x=1

))



fig.show()
import plotly.express as px





fig = px.scatter(df_canvas, x="Total Budget", y="profit", color="Insider",

    size="BO Rev", size_max=45, title = 'Insider - Budget vs Profit')



fig.update_layout(legend=dict(

    orientation="h",

    yanchor="bottom",

    y=1.02,

    xanchor="right",

    x=1

))



fig.show()
fig = px.scatter(df_canvas, x="Rating", y="Total Budget", color="Big Production",

    size="BO Rev", size_max=45, title = 'Big Production - Rating vs Budget')



fig.update_layout(legend=dict(

    orientation="h",

    yanchor="bottom",

    y=1.02,

    xanchor="right",

    x=1

))



fig.show()
import plotly.express as px

fig = px.scatter(df_canvas, x="Rating", y="Total Budget", color="Insider",

    size="BO Rev", size_max=45, title = 'Insider - Rating vs Budget')



fig.update_layout(legend=dict(

    orientation="h",

    yanchor="bottom",

    y=1.02,

    xanchor="right",

    x=1

))



fig.show()
df_corr = df1[['Big Production','Insider','Total Budget','BO Rev','Rating','Reviews']]

df_corr.head()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df_corr.corr(), annot=True, linewidths=1, fmt='.2f',ax=ax)
# Distribution

plt.figure(figsize=(10,5))

sns.distplot(df1['BO Rev'], bins=24, color='g').set_title('Box Office Revenue All')
df2 = df1[df1["BO Rev"] > 0]

df2.head()
ax = sns.distplot(df2['Rating'], vertical=True, bins = 10)
plt.figure(figsize=(10,5))

sns.distplot(df2['BO Rev'], bins=24, color='g').set_title('Box Office Revenue >0')
# Movies only Rev higher than 0 

df2 = df1[df1["BO Rev"] > 0]



# Fetch Ratings for Big/Small Production movies

BP_REV = df2[(df2["Big Production"] == 1)]

SP_REV = df2[(df2["Big Production"] == 0)]



A = BP_REV["BO Rev"]

B = SP_REV["BO Rev"]



label1 = "BP Revenue"

label2 = "SP Revenue"

title = 'Revenue Big/Small Production'

#-------



# Plot Distribution 

plt.figure(figsize=(10,5))

sns.distplot(A,hist=False, bins=10, color='g', label = label1).set_title(title)

sns.distplot(B,hist=False, bins=10, color='b', label = label2).set_title(title)







# Perform Z-test

from scipy import stats

from statsmodels.stats import weightstats as stests



ztest ,pval2 = stests.ztest(A, x2=B, value=0,alternative='two-sided')



print("p-Value is " + str("%.10f" % pval2))



if pval2<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")



    

# Mean Median for Distributoin

# initialise data of lists. 

data = {'Mean':[A.mean(), B.mean()], 'Median': [A.median(), B.median()]} 



# Creates pandas DataFrame. 

df_mean_median = pd.DataFrame(data, index =['Big Production', 'Small Production']) 





# print the data 

df_mean_median
# Movies only Budget higher than 0 

df4 = df1[df1["Total Budget"] > 0]



# Fetch Profit for Big/Small Production movies

BP_PROF = df4[(df4["Big Production"] == 1)]

SP_PROF = df4[(df4["Big Production"] == 0)]



A = BP_PROF["profit"]

B = SP_PROF["profit"]



label1 = "BP Profit"

label2 = "SP Profit"

title = 'Profit Big/Small Production'

#-------



# Plot Distribution 

plt.figure(figsize=(10,5))

sns.distplot(A,hist=False, bins=10, color='g', label = label1).set_title(title)

sns.distplot(B,hist=False, bins=10, color='b', label = label2).set_title(title)







# Perform Z-test

from scipy import stats

from statsmodels.stats import weightstats as stests



ztest ,pval2 = stests.ztest(A, x2=B, value=0,alternative='two-sided')



print("p-Value is " + str("%.5f" % pval2))



if pval2<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")



    

# Mean Median for Distributoin

# initialise data of lists. 

data = {'Mean':[A.mean(), B.mean()], 'Median': [A.median(), B.median()]} 



# Creates pandas DataFrame. 

df_mean_median = pd.DataFrame(data, index =['Big Production', 'Small Production']) 





# print the data 

df_mean_median
ax = sns.distplot(df1["Total Budget"], vertical=True)
# Movies only Rev higher than 0 

df3 = df1[df1["Rating"] > 0]



# Fetch Ratings for Big/Small Production movies

BP_RAT = df3[(df3["Big Production"] == 1)]

SP_RAT = df3[(df3["Big Production"] == 0)]



A = BP_RAT["Rating"]

B = SP_RAT["Rating"]



label1 = "BP Rating"

label2 = "SP Rating"

title = 'Rating Big/Small Production'

#-------



#A = pd.DataFrame(BP_RAT["Rating"])



# Plot Distribution 

plt.figure(figsize=(10,5))

sns.distplot(A,hist=False, bins=10, color='g', label = label1).set_title(title)

sns.distplot(B,hist=False, bins=10, color='b', label = label2).set_title(title)





# Perform Z-test

from scipy import stats

from statsmodels.stats import weightstats as stests



ztest ,pval2 = stests.ztest(A, x2=B, value=0,alternative='two-sided')



print("p-Value is " + str("%.5f" % pval2))



if pval2<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")



    

# Mean Median for Distributoin

# initialise data of lists. 

data = {'Mean':[A.mean(), B.mean()], 'Median': [A.median(), B.median()]} 



# Creates pandas DataFrame. 

df_mean_median = pd.DataFrame(data, index =['Big Production', 'Small Production']) 





# print the data 

df_mean_median
ax = sns.distplot(df3['Rating'], vertical=True, bins = 10)
# Movies only Rev higher than 0 

df2 = df1[df1["BO Rev"] > 0]



# Fetch Ratings for Insider/Outsider movies

IN_REV = df2[(df2["Insider"] == 1)]

OS_REV = df2[(df2["Insider"] == 0)]



A = IN_REV["BO Rev"]

B = OS_REV["BO Rev"]



label1 = "IN Revenue"

label2 = "OS Revenue"

title = 'Revenue Insider/Outsider'

#-------



# Plot Distribution 

plt.figure(figsize=(10,5))

sns.distplot(A,hist=False, bins=10, color='g', label = label1).set_title(title)

sns.distplot(B,hist=False, bins=10, color='b', label = label2).set_title(title)







# Perform Z-test

from scipy import stats

from statsmodels.stats import weightstats as stests



ztest ,pval2 = stests.ztest(A, x2=B, value=0,alternative='two-sided')



print("p-Value is " + str("%.5f" % pval2))





if pval2<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")



    

# Mean Median for Distributoin

# initialise data of lists. 

data = {'Mean':[A.mean(), B.mean()], 'Median': [A.median(), B.median()]} 



# Creates pandas DataFrame. 

df_mean_median = pd.DataFrame(data, index =['Insider', 'Outsider']) 





# print the data 

df_mean_median
# Movies only Budget higher than 0 

df4 = df1[df1["Total Budget"] > 0]



# Fetch Profit for Big/Small Production movies

IN_PROF = df4[(df4["Insider"] == 1)]

OS_PROF = df4[(df4["Insider"] == 0)]



A = IN_PROF["profit"]

B = OS_PROF["profit"]



label1 = "IN Profit"

label2 = "OS Profit"

title = 'Profit Insider/Outsider'

#-------



# Plot Distribution 

plt.figure(figsize=(10,5))

sns.distplot(A,hist=False, bins=10, color='g', label = label1).set_title(title)

sns.distplot(B,hist=False, bins=10, color='b', label = label2).set_title(title)







# Perform Z-test

from scipy import stats

from statsmodels.stats import weightstats as stests



ztest ,pval2 = stests.ztest(A, x2=B, value=0,alternative='two-sided')



print("p-Value is " + str("%.5f" % pval2))



if pval2<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")



    

# Mean Median for Distributoin

# initialise data of lists. 

data = {'Mean':[A.mean(), B.mean()], 'Median': [A.median(), B.median()]} 



# Creates pandas DataFrame. 

df_mean_median = pd.DataFrame(data, index =['Insider', 'Outsider']) 





# print the data 

df_mean_median
# Movies only Rev higher than 0 

df3 = df1[df1["Rating"] > 0]



# Fetch Ratings for Big/Small Production movies

IN_RAT = df3[(df3["Insider"] == 1)]

OS_RAT = df3[(df3["Insider"] == 0)]



A = IN_RAT["Rating"]

B = OS_RAT["Rating"]



label1 = "IN Rating"

label2 = "OS Rating"

title = 'Rating Insider/Outsider'

#-------



# Plot Distribution 

plt.figure(figsize=(10,5))

sns.distplot(A,hist=False, bins=10, color='g', label = label1).set_title(title)

sns.distplot(B,hist=False, bins=10, color='b', label = label2).set_title(title)







# Perform Z-test

from scipy import stats

from statsmodels.stats import weightstats as stests



ztest ,pval2 = stests.ztest(A, x2=B, value=0,alternative='two-sided')



print("p-Value is " + str("%.5f" % pval2))



if pval2<0.05:

    print("reject null hypothesis")

else:

    print("accept null hypothesis")



    

# Mean Median for Distributoin

# initialise data of lists. 

data = {'Mean':[A.mean(), B.mean()], 'Median': [A.median(), B.median()]} 



# Creates pandas DataFrame. 

df_mean_median = pd.DataFrame(data, index =['Insider', 'Outsider']) 





# print the data 

df_mean_median