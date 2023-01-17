from __future__ import print_function  # Compatability with Python 3

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
from pandas import DataFrame
from pandas import TimeGrouper

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline 

# Seaborn for easier visualization
import seaborn as sns

# datetime
import datetime
# Load data from CSV
df = pd.read_csv("../input/train.csv")
df2 = pd.read_csv("../input/store.csv")
# Dataframe dimensions
print(df.shape)
print(df2.shape)
# Column datatypes
print(df.dtypes,'\n')
print(df2.dtypes)
# Display first 5 rows of df
df.head()
# Display last 5 rows of data
df.tail()
df2.head()
df2.tail()
# Drop duplicates
df = df.drop_duplicates()
df2 = df2.drop_duplicates()
print(df.shape)
print(df2.shape)
# drop closed observation
df = df[df.Open != 0]
len(df[df.Customers == 0])
df[df.Customers == 0].sort_values(by=['Store'])
len(df[df.Sales == 0])
# after checking the data, decide to drop sales == 0 observations
df = df[df.Sales != 0]
print(df.shape)
df['AvgPurchasing'] = df.Sales / df.Customers
# Plot histogram grid
df.hist(xrot=-45,figsize=(10,10))
# Clear the text "residue"
plt.show()
# Plot histogram grid
df2.hist(xrot=-45,figsize=(10,10))
# Clear the text "residue"
plt.show()
# Summarize numerical features
df.describe()
# Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
df2.describe()
# Box plot of 'Sales'
plt.figure(figsize=(4,3))
sns.boxplot(y='Sales', data=df)
plt.figure(figsize=(4,3))
sns.boxplot(y='Customers', data=df)
plt.figure(figsize=(4,3))
sns.boxplot(y='AvgPurchasing', data=df)
df[df.Sales < 1000][['Store','Sales']].describe()
df.groupby('Store')['Sales'].mean().sort_values()
df[df.Store == 652]['Sales'].describe()
df[df.Store == 652]['Sales'].sort_values()
df=df.reset_index()
def find_low_high(feature):
    # find store specific Q1 - 3*IQ and Q3 + 3*IQ
    IQ = df.groupby('Store')[feature].quantile(0.75)-df.groupby('Store')[feature].quantile(0.25)
    Q1 = df.groupby('Store')[feature].quantile(0.25)
    Q3 = df.groupby('Store')[feature].quantile(0.75)
    low = Q1 - 3*IQ
    high = Q3 + 3*IQ
    low = low.to_frame()
    low = low.reset_index()
    low = low.rename(columns={feature: "low"})
    high = high.to_frame()
    high = high.reset_index()
    high = high.rename(columns={feature: "high"})
    return {'low':low, 'high':high}
def find_outlier_index(feature):
    main_data = df[['Store',feature]]
    low = find_low_high(feature)["low"]
    high = find_low_high(feature)["high"]
    
    new_low = pd.merge(main_data, low, on='Store', how='left')
    new_low['outlier_low'] = (new_low[feature] < new_low['low'])
    index_low = new_low[new_low['outlier_low'] == True].index
    index_low = list(index_low)
    
    new_high = pd.merge(main_data, high, on='Store', how='left')
    new_high['outlier_high'] = new_high[feature] > new_high['high']
    index_high = new_high[new_high['outlier_high'] == True].index
    index_high = list(index_high)
    
    index_low.extend(index_high)
    index = list(set(index_low))
    return index
len(find_outlier_index("Sales"))
# decide only to delete the 1113 observations above to delete the sales outlier
df.drop(find_outlier_index("Sales"), inplace=True, axis=0)
df.shape
from scipy.stats import boxcox
df['Sales'], lam1 = boxcox(df.Sales)
df['Customers'], lam2 = boxcox(df.Customers)
df['AvgPurchasing'], lam3 = boxcox(df.AvgPurchasing)
print(lam1)
df.Sales.hist(figsize=(4,2))
plt.show()

print(lam2)
df.Customers.hist(figsize=(4,2))
plt.show()

print(lam3)
df.AvgPurchasing.hist(figsize=(4,2))
plt.show()
print(df.select_dtypes(exclude=['object']).isnull().sum(),'\n')
print(df2.select_dtypes(exclude=['object']).isnull().sum())
# for competion data, check the 3 missing CompetitionDistance
df2[df2['CompetitionDistance'].isnull()]
# fill and flag the missing numeric data
df2.CompetitionOpenSinceMonth.fillna(0, inplace=True)
df2.CompetitionOpenSinceYear.fillna(0, inplace=True)
df2.CompetitionDistance.fillna(0, inplace=True)
# flag: indicator variable for missing numeric data
df2['CompetitionOpenSinceMonth_missing'] = df2.CompetitionOpenSinceMonth.isnull().astype(int)
df2['CompetitionOpenSinceYear_missing'] = df2.CompetitionOpenSinceYear.isnull().astype(int)
df2['CompetitionDistance_missing'] = df2.CompetitionDistance.isnull().astype(int)
# check是否当且仅当promo2为0时，Promo2SinceWeek，Promo2SinceYear，Promo2Interval为Nan？
df2[df2['Promo2']==0][['Promo2SinceWeek','Promo2SinceYear','PromoInterval']].isnull().sum()
# just fill the nan with 0 because it is actually not missing data 
df2.Promo2SinceWeek.fillna(0, inplace=True)
df2.Promo2SinceYear.fillna(0, inplace=True)
df2.PromoInterval.fillna(0, inplace=True)
df2.isnull().sum()
# Plot bar plot for each categorical feature
plt.figure(figsize=(4,4))
sns.countplot(y='SchoolHoliday', data=df)
plt.show()
plt.figure(figsize=(4,4))
sns.countplot(y='StateHoliday', data=df)
plt.show()
for feature in df2.dtypes[df2.dtypes=='object'].index:
    plt.figure(figsize=(4,4))
    sns.countplot(y=feature, data=df2)
    plt.show()
# Display unique values of 'basement'
df.StateHoliday.unique()
df.StateHoliday.replace(0, '0',inplace=True)
# Display number of missing values by feature (categorical)
print(df.select_dtypes(include=['object']).isnull().sum(), '\n')
print(df2.select_dtypes(include=['object']).isnull().sum())
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='StateHoliday', data=df)
plt.figure(figsize=(4,4))
sns.boxplot(y='Customers', x='StateHoliday', data=df)
plt.figure(figsize=(4,4))
sns.boxplot(y='AvgPurchasing', x='StateHoliday', data=df)
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='SchoolHoliday', data=df)
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='Store', data=df)
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='DayOfWeek', data=df)
plt.figure(figsize=(4,4))
sns.boxplot(y='Sales', x='Promo', data=df)
plt.figure(figsize=(4,4))
sns.boxplot(y='Customers', x='Promo', data=df)
plt.figure(figsize=(4,4))
sns.boxplot(y='AvgPurchasing', x='Promo', data=df)
df.index = df['Store']
df2.index = df2['Store']
df = df.drop(['Store'], axis=1)
df_combined = df.join(df2)
df_combined = df_combined.reset_index(drop=True)
df_combined.head()
# note that the order from the most to the least number in each type: a,d,c,b
sns.boxplot(y='Sales', x='StoreType', data=df_combined)
sns.boxplot(y='Sales', x='Assortment', data=df_combined)
sns.factorplot(data=df_combined, x="StoreType", y="Sales", col="Assortment")
# only 9 stores has assortment == 'b'
df_combined[df_combined.Assortment == 'b'].Store.unique()
# only 17 stores has StoreType == 'b'
df_combined[df_combined.StoreType == 'b'].Store.unique()
g = sns.FacetGrid(df_combined, col="StoreType")
g.map(sns.distplot, "Sales")
g = sns.FacetGrid(df_combined, col="StoreType")
g.map(plt.scatter, "Customers", "Sales")
sns.lmplot(x='Customers', y='Sales', data=df_combined, hue='StoreType',fit_reg=False)
sns.lmplot(x='AvgPurchasing', y='Sales', data=df_combined, hue='StoreType',fit_reg=False)
sns.lmplot(x='Customers', y='Sales', data=df_combined, hue='Assortment',fit_reg=False)
sns.lmplot(x='AvgPurchasing', y='Sales', data=df_combined, hue='Assortment',fit_reg=False)
sns.lmplot(x='Customers', y='Sales', data=df_combined, hue='Promo',fit_reg=False)
# Calculate correlations between numeric features
correlations = df_combined.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Make the figsize 10 x 8
plt.figure(figsize=(9,8))
# Plot heatmap of annotated correlations
sns.heatmap(correlations*100, annot=True, fmt='.0f',mask = mask, cbar=False)
def get_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Quarter'] = df['Date'].dt.quarter
    df['Week'] = df['Date'].dt.week
    
    return df
df_combined = get_date_features(df_combined)
def get_series(Store_i):
    new_df = df_combined[df_combined.Store == Store_i][['Date','Sales']]
    new_df.index = new_df.Date
    new_df.drop('Date', axis = 1, inplace = True)
    new_series = new_df.T.squeeze()
    return new_series
for i in df_combined.StoreType.unique():
    print(i, df_combined[df_combined.StoreType == i].Store[:1])
new_series_2 = get_series(2)
new_series_85 = get_series(85)
new_series_1 = get_series(1)
new_series_13 = get_series(13)
plt.figure(figsize=(16,2))
new_series_2.plot(style = 'k--')
plt.show()
plt.figure(figsize=(16,2))
new_series_85.plot(style = 'k--')
plt.show()
plt.figure(figsize=(16,2))
new_series_1.plot(style = 'k--')
plt.show()
plt.figure(figsize=(16,2))
new_series_13.plot(style = 'k--')
plt.show()
new_series_2.index = pd.to_datetime(new_series_2.index)
groups = new_series_2.groupby([TimeGrouper("A")])
plt.figure(figsize=(20,3))
a=311
print("Store2 Daily Sales Plot")
for name, group in groups:
    plt.subplot(a) 
    group.plot()
    a+=1
    plt.title(name.year)
groups = new_series_2['2013'].groupby([TimeGrouper("A"),TimeGrouper("Q")])
plt.figure(figsize=(20,3))
a=411
print("Store2 Daily Sales Plot")
for name, group in groups:
    plt.subplot(a) 
    group.plot()
    a+=1
    plt.title(name)
    
groups = new_series_2['2014'].groupby([TimeGrouper("A"),TimeGrouper("Q")])
plt.figure(figsize=(20,3))
a=411
for name, group in groups:
    plt.subplot(a) 
    group.plot()
    a+=1
    plt.title(name)
    
groups = new_series_2['2015'].groupby([TimeGrouper("A"),TimeGrouper("Q")])
plt.figure(figsize=(20,3))
a=411
for name, group in groups:
    plt.subplot(a) 
    group.plot()
    a+=1
    plt.title(name)
groups = new_series_2['2013'].groupby([TimeGrouper("A"),TimeGrouper("M")])
plt.figure(figsize=(15,6))
a=611
print("Store2 Daily Sales Plot")
i = 1
for name, group in groups:
    if i>6:
        break
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

plt.figure(figsize=(15,6))
i = 1
a=611
for name, group in groups:
    if i<=6:
        i+=1
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)
df_combined.Date = pd.to_datetime(df_combined.Date)
daily_sales_sum = df_combined.groupby(['Date'])['Sales'].sum()
daily_sales_mean = df_combined.groupby(['Date'])['Sales'].mean()
daily_sales_median = df_combined.groupby(['Date'])['Sales'].median()
daily_sales_max = df_combined.groupby(['Date'])['Sales'].max()
daily_sales_min = df_combined.groupby(['Date'])['Sales'].min()
print("All stores total monthly sales - by Year")
plt.figure(figsize=(16,2))
daily_sales_sum['2013'].groupby([TimeGrouper("A"),TimeGrouper("M")]).sum().plot()
plt.show()

plt.figure(figsize=(16,2))
daily_sales_sum["2014"].groupby([TimeGrouper("A"),TimeGrouper("M")]).sum().plot()
plt.show()

plt.figure(figsize=(16,2))
daily_sales_sum["2015"].groupby([TimeGrouper("A"),TimeGrouper("M")]).sum().plot()
plt.show()
groups = daily_sales_sum["2013"].groupby([TimeGrouper("A"),TimeGrouper("M")])

plt.figure(figsize=(15,6))
a=611
print("2013 All Store Daily Total Sales Plot - by Month")
i = 1
for name, group in groups:
    if i>6:
        break
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

plt.figure(figsize=(15,6))
i = 1
a=611
for name, group in groups:
    if i<=6:
        i+=1
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)
groups = daily_sales_mean["2014"].groupby([TimeGrouper("A"),TimeGrouper("M")])

plt.figure(figsize=(15,6))
a=611
print("2014 All Store Daily Total Sales Plot - by Month")
i = 1
for name, group in groups:
    if i>6:
        break
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

plt.figure(figsize=(15,6))
i = 1
a=611
for name, group in groups:
    if i<=6:
        i+=1
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)
groups = daily_sales_mean["2015"].groupby([TimeGrouper("A"),TimeGrouper("M")])

plt.figure(figsize=(15,6))
a=611
print("2015 All Store Daily Total Sales Plot - by Month")
i = 1
for name, group in groups:
    if i>6:
        break
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)

plt.figure(figsize=(15,6))
i = 1
a=611
for name, group in groups:
    if i<=6:
        i+=1
    else:
        plt.subplot(a) 
        group.plot()
        a+=1
        i+=1
        plt.title(name)
groups = df_combined.groupby(['Year','Month'])['Sales'].mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = df_combined.groupby(['Year','Month'])['Sales'].median()
line2=groups.plot(label = 'median')

groups = df_combined.groupby(['Year','Month'])['Sales'].max()
line3=groups.plot(label = 'max')

groups = df_combined.groupby(['Year','Month'])['Sales'].min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])
plt.title("overall sales: monthly statistics",color='blue')
plt.show()
groups = df_combined.groupby(['Year'])['Sales'].mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = df_combined.groupby(['Year'])['Sales'].median()
line2=groups.plot(label = 'median')

groups = df_combined.groupby(['Year'])['Sales'].max()
line3=groups.plot(label = 'max')

groups = df_combined.groupby(['Year'])['Sales'].min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])
plt.title("overall sales: yearly statistics",color='blue')
plt.show()
groups = new_series_2.groupby(TimeGrouper("W")).mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = new_series_2.groupby(TimeGrouper("W")).median()
line2=groups.plot(label = 'median')

groups = new_series_2.groupby(TimeGrouper("W")).max()
line3=groups.plot(label = 'max')

groups = new_series_2.groupby(TimeGrouper("W")).min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])
plt.show()
groups = new_series_2.groupby(TimeGrouper("M")).mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = new_series_2.groupby(TimeGrouper("M")).median()
line2=groups.plot(label = 'median')

groups = new_series_2.groupby(TimeGrouper("M")).max()
line3=groups.plot(label = 'max')

groups = new_series_2.groupby(TimeGrouper("M")).min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])
plt.show()
groups = new_series_2.groupby(TimeGrouper("Q")).mean()
plt.figure(figsize=(10,3))
a = plt.subplot(1,1,1)
#plt.subplot(131) 
#plt.title('Monthly mean plot',color='blue') 
line1=groups.plot(label = 'mean')

groups = new_series_2.groupby(TimeGrouper("Q")).median()
line2=groups.plot(label = 'median')

groups = new_series_2.groupby(TimeGrouper("Q")).max()
line3=groups.plot(label = 'max')

groups = new_series_2.groupby(TimeGrouper("Q")).min()
line4=groups.plot(label = 'min')

handles, labels = a.get_legend_handles_labels()
a.legend(handles[::-1], labels[::-1])
plt.show()
def lag_n_plot(series, n):
    series_lag_n = series.shift(n)
    df_from_series = pd.DataFrame(series)
    df_from_series = df_from_series.rename(columns={'Sales':'Sales_t'})
    df_from_series_lag_n = pd.DataFrame(series_lag_n)
    df_from_series_lag_n = df_from_series_lag_n.rename(columns={'Sales':'Sales_t-n'})
    new_df = pd.concat([df_from_series, df_from_series_lag_n], axis=1)
    plt.title('Lag %d plot' %(n))
    #plt.figure(figsize=(3,3))
    plt.scatter(y = "Sales_t", x = "Sales_t-n", data=new_df, alpha = 0.5)
print('lag plot of All Store daily sales sum')
plt.figure(figsize=(16,2))
plt.subplot(151) 
lag_n_plot(daily_sales_sum, 1)

plt.subplot(152) 
lag_n_plot(daily_sales_sum, 7)

plt.subplot(153) 
lag_n_plot(daily_sales_sum, 14)

plt.subplot(154) 
lag_n_plot(daily_sales_sum, 28)

plt.subplot(155) 
lag_n_plot(daily_sales_sum, 90)
print('lag plot of All Store daily sales mean')
plt.figure(figsize=(16,2))
plt.subplot(151) 
lag_n_plot(daily_sales_mean, 1)

plt.subplot(152) 
lag_n_plot(daily_sales_mean, 7)

plt.subplot(153) 
lag_n_plot(daily_sales_mean, 14)

plt.subplot(154) 
lag_n_plot(daily_sales_mean, 28)

plt.subplot(155) 
lag_n_plot(daily_sales_mean, 90)
print('lag plot of Store2 daily sales')
plt.figure(figsize=(16,2))
plt.subplot(151) 
lag_n_plot(new_series_2, 1)

plt.subplot(152) 
lag_n_plot(new_series_2, 7)

plt.subplot(153) 
lag_n_plot(new_series_2, 14)

plt.subplot(154) 
lag_n_plot(new_series_2, 28)

plt.subplot(155) 
lag_n_plot(new_series_2, 90)
from pandas.plotting import autocorrelation_plot
print('autocorrelation plot of Store2 daily sales')
plt.figure(figsize=(20,4))
plt.xticks([x for x in range(900) if x % 28 == 0]) 
autocorrelation_plot(new_series_2)
plt.show()
print('autocorrelation plot of All Store daily sales mean')
plt.figure(figsize=(20,4))
plt.xticks([x for x in range(900) if x % 28 == 0])  
autocorrelation_plot(daily_sales_mean)
plt.show()
print('autocorrelation plot of All Store daily sales sum')
plt.figure(figsize=(20,4))
plt.xticks([x for x in range(900) if x % 28 == 0]) 
autocorrelation_plot(daily_sales_sum)
plt.show()
df_combined.to_csv('df_combined_cleaned.csv', index=None)