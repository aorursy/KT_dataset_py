# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#pip install --upgrade seaborn
df2 = pd.read_csv('../input/world-population-19602018/pop_worldometer_data.csv')
df2.describe()
df2.isnull().sum()
# Taking care of missing data
df2=df2.fillna(df2.mean())
top10 = df2[df2.index<=5]
others = df2[df2.index>5]
others['Country (or dependency)'] = 'Others'
pie_df = top10.append(others)
pie_df = pie_df.groupby('Country (or dependency)').sum().sort_values('Population (2020)',ascending=False).reset_index()
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(aspect="equal"))
ingredients = pie_df['Country (or dependency)']
data= pie_df['Population (2020)']

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%".format(pct, absolute)

wedges, texts, autotexts = ax.pie(data, labels = pie_df['Country (or dependency)'], autopct=lambda pct: func(pct, data),
                                  textprops=dict(color="w"),shadow=True)
ax.legend(wedges, ingredients,
          title="Countries",
          loc="best",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=15, weight="bold")
ax.set_title("World Population ", size=25)
grow_fast=df2.sort_values('Yearly Change %',ascending=False).head(10)
grow_most=df2.sort_values('Net Change',ascending=False).head(10)
density_most=df2.sort_values('Density (P/Km²)',ascending=False).head(10)
fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True,figsize=(12,10))
ax1 = plt.subplot(221)
ax1.margins(0)           # Default margin is 0.05, value 0 means fit
ax1.bar(density_most['Country (or dependency)'], density_most['Density (P/Km²)'],color='r')
plt.ylabel("(P/Km²)")
ax1.set_title('Densely Most')
plt.xticks(rotation=45)

ax2 = plt.subplot(222)
ax2.margins(0)           # Default margin is 0.05, value 0 means fit
ax2.bar(grow_fast['Country (or dependency)'], grow_fast['Yearly Change %'],color='g')
ax2.set_title('Grow Fastest')
plt.ylabel("(%)")
plt.xticks(rotation=45)

ax3 = plt.subplot(212)
ax3.margins(0)           # Default margin is 0.05, value 0 means fit
ax3.bar(grow_most['Country (or dependency)'], grow_most['Net Change'])
plt.ylabel("(10 Million)")
ax3.set_title('Grow Most')
plt.xticks(rotation=45)
fig.tight_layout(pad=1.0)
sns.despine(left=True, bottom=True, right=True)
corr = df2.corr()
plt.figure(figsize=(15,12))
sns.heatmap(corr, annot=True, linewidths=0.5)
#seems land area and Net change are most relevant variables
# Importing the dataset
dataset = df2
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:,1].values
X=X[:,:-1]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))
X_dist = pd.DataFrame(X_train)
X_dist = X_dist.rename(columns={1:'Yearly Change',2:'Net Change',3:'Density',4:'Land Area',5:'Migrants',6:'Fert. Rate',
                   7:'Med. Age',8:'Urban Pop'})
X_dist = X_dist.iloc[:,1:]
a = X_dist.stack().reset_index().drop(columns =['level_0']).rename(columns={'level_1':'variables',0:'values'})
a

sns.displot(
    a, x="values", col="variables",col_wrap=3,
    binwidth=1, height=3, facet_kws=dict(margin_titles=True),
)

import scipy
for x in range(0,8):  
    print(scipy.stats.normaltest(X_train[x]))
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X_train = np.append(arr = np.ones((188, 1)).astype(int), values = X_train, axis = 1)
X_opt = X_train [:, [0, 1, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train [:, [1, 2, 3, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X_train [:, [1, 2, 4, 5, 6, 7]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

#Here is the optimal with Adj. R-squared (uncentered): 0.830
X_opt = X_train [:, [1, 2, 4, 5, 7]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
df = pd.read_csv('../input/world-population-19602018/population_total_long.csv')
df.describe()
df.info()
#Check if there are nulls
df.isnull().sum()
df.head()
#Get world pop data
wd_pop =df.groupby('Year').sum().reset_index()
wd_pop = wd_pop.rename({'Count':'World'},axis=1)
#Then, China's
cn_pop = df[df['Country Name']=='China'].reset_index().iloc[:,2:]
cn_pop = cn_pop.rename({'Count':'China'},axis=1)
#Now transform the dataframe into what I need
country_name = df['Country Name'].unique()
country_count = len(df['Country Name'].unique())

for x in range(0,country_count):
    x_pop = df[df['Country Name']== country_name[x]].reset_index().iloc[:,2:]
    x_pop = x_pop.rename({'Count':country_name[x]},axis=1)
    wd_pop[country_name[x]]=x_pop[country_name[x]]
#Now I have the data I wanted and then rename it 
pop = wd_pop
# Plot the pop for China and world
plt.figure(figsize=(12,6))
sns.lineplot(x="Year", y="China",label="China",
             data = pop)
sns.lineplot(x="Year", y="World",label="World",
             data = pop)
sns.lineplot(x="Year", y="India",label="India",
             data = pop)
plt.xticks(rotation=-45)
plt.title("Population Increment 1960-2017")
plt.xlabel("")
plt.ylabel("Population")

pop2=pop.tail(30)
pop2.Year=pop2.Year.astype(str)
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

# Plot the world population
sns.set_color_codes("pastel")
sns.barplot(y="World", x="Year", data=pop2, label="World", color="b")

# Plot the population of China
sns.set_color_codes("muted")
sns.barplot(y="China",x="Year", data = pop2, label = "China", color="c")

# Plot the population of China
sns.set_color_codes("muted")
sns.barplot(y="India",x="Year",  data = pop2, label="India", color="b")

ax.set_yscale('log')
ax.set_yticks(np.arange(10**9, 1.2*10**10, 10**10))

plt.title("Population Increment 1988-2017")
plt.xlabel("")
plt.ylabel("")
# Add a legend and informative axis label
ax.legend(ncol=1, loc="upper right", frameon=True)
sns.despine(left=False, bottom=False)
#Preparation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

#use make_pipeline to combine linear regression with polynomial regiression
def PolynomialRegression(degree=2,**kwargs):
    poly_reg = PolynomialFeatures(degree)
    lin_reg = LinearRegression(**kwargs)
    return make_pipeline(poly_reg,lin_reg)

sns.set(rc={'figure.figsize':(10,8)},font='SimHei',font_scale=1.5)
# Polynomial Regression for China's population
# Importing the dataset
dataset=pop[['Year','China']]
y = dataset.iloc[:, 1].values.reshape(-1,1)
X = dataset.iloc[:, 0].values.reshape(-1,1)

X_test = np.linspace(1960,2040,71)[:,None]

plt.scatter(X.ravel(),y,color='black')
plt.axis()
for degree in [1,5,10]:
    regressor = PolynomialRegression(degree).fit(X, y)
    y_test = regressor.predict(X_test)
    plt.plot(X_test.ravel(),y_test,label='degree={0}'.format(degree))
plt.title("China's Population Predict on Different Fit Degree")
plt.legend(loc='best')
#Finding optimal model
from sklearn.model_selection import validation_curve

degree= np.arange(0,15)
train_score, val_score= validation_curve(PolynomialRegression(),X,y,
        'polynomialfeatures__degree',degree, cv=7)

plt.plot(degree, np.median(train_score,1), 
         color='blue',label='training score')
plt.plot(degree, np.median(val_score,1),
         color='red',label='validation score')
plt.legend(loc='best')
plt.ylim(0,1)
plt.xlabel('degree')
plt.ylabel('score')
#Finding optimal polynomial degree
validation_score=np.median(val_score,1).tolist()
max_index=validation_score.index(max(validation_score))
degree[max_index]
plt.scatter(X.ravel(),y,c='r')
lim= plt.axis()
regressor=PolynomialRegression(14).fit(X, y)
y_test1 = regressor.predict(X_test)
plt.plot(X_test,y_test1)
plt.axis(lim)