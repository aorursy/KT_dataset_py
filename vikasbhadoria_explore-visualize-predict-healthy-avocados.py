import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/avocado-prices/avocado.csv')
df.head()
df = df.drop('Unnamed: 0',axis=1)
df.info()
df.describe()
df['Date'] = pd.to_datetime(df['Date'],errors='coerce')
sns.heatmap(df.isnull(),cbar=False,cmap='Blues',yticklabels=False)
df.year.value_counts()
round_columns = df[['Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags']]
for i in round_columns.columns:
    df[i] = df[i].apply(np.round)
df.head()
df.hist(bins=30,figsize=(12,10),color='skyblue',ec="black")
plt.figure(figsize=(10,5))
plt.title("Price Distribution")
ax = sns.distplot(df["AveragePrice"], color = 'b')
sns.barplot(x=df['type'],y=df['Total Volume'].value_counts())
df.year.value_counts().sort_index().plot(kind='barh',figsize=(6,4),color='skyblue',ec='black')
regionsToRemove = ['California', 'GreatLakes', 'Midsouth', 'NewYork', 'Northeast', 'SouthCarolina', 'Plains', 'SouthCentral', 'Southeast', 'TotalUS', 'West']
df = df[~df.region.isin(regionsToRemove)]
len(df.region.unique())
plt.figure(figsize=(15,12))
sns.set(style="white", context="talk")
plt.title("Avg.Price of Avocado by City")
sns.barplot(x="AveragePrice",y="region",data= df,palette="rocket")

# As seen avg price of avocado is the most in San Francisco & hartford springfield
plt.figure(figsize=(15,12))
sns.set(style="white", context="talk")
plt.title("Total volume of Avocado sold by City")
sns.barplot(x="Total Volume",y="region",data= df,palette="deep")
plt.figure(figsize=(8,4))
sns.set(style="white", context="talk")
plt.title("Avg.price of Avocado as per type")
sns.boxplot(x="AveragePrice",y="type",data= df,palette="vlag")
# Making a new column 'Month'
df['Month'] = pd.DatetimeIndex(df['Date']).month
df.head(1)
axis = df.groupby('Month')[['AveragePrice']].mean().plot(figsize=(10,5),marker='o',color='r')
plt.figure()
axis = df.groupby('Month')[['Total Volume']].mean().plot(figsize=(10,5),marker='o',color='g')
# Making a new column 'Day'.
df['Day'] = pd.DatetimeIndex(df['Date']).day
axis = df.groupby('Day')[['AveragePrice']].mean().plot(figsize=(14,5),marker='o',color='r')
plt.figure()
axis = df.groupby('Day')[['Total Volume']].mean().plot(figsize=(14,5),marker='o',color='g')
plt.figure(figsize=(18,18))
sns.set(style="white", context="talk")
plt.title("Avg.Price of Avocado by City")
sns.boxplot(x="AveragePrice",y="region",data= df,palette="deep")
fig,ax = plt.subplots(figsize=(15,6))
df.groupby(['Date','type']).mean()['AveragePrice'].unstack().plot(ax=ax)
plt.title('Avg Price of avocado as per type on avocado over time')
fig,ax = plt.subplots(figsize=(15,6))
df.groupby(['Date','type']).mean()['Total Bags'].unstack().plot(ax=ax)
plt.title('Total volume of avocado sold as per type on avocado over time')
avacado_type = df['type']=='organic'
plt.figure(figsize=(18,18))
sns.set(style="white", context="talk")
plt.title("Average price of organic Avocado as per City")
sns.boxplot(x="AveragePrice",y="region",data= df[avacado_type],palette="deep")
avacado_type = df['type']=='conventional'
plt.figure(figsize=(18,18))
sns.set(style="white", context="talk")
plt.title("Average Price of conventional Avocado as per City")
sns.boxplot(x="AveragePrice",y="region",data= df[avacado_type],palette="deep")
df_corr = df[['AveragePrice','Total Volume','Total Bags','Month']]
correlations = df_corr.corr()
plt.figure(figsize=(8,5))
sns.heatmap(correlations,annot=True,cmap="YlGnBu",linewidths=.5)
# Month has a very good correlation with AvgPrice.
df_to_plot = df.drop(['Date','AveragePrice', 'Total Volume', 'Total Bags','type','region','Month','Day'], axis = 1).groupby('year').agg('sum')
df_to_plot.head()
index = ['4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags']
series = pd.DataFrame({'2015': df_to_plot.loc[[2015],:].values.tolist()[0],
                      '2016': df_to_plot.loc[[2016],:].values.tolist()[0],
                      '2017': df_to_plot.loc[[2017],:].values.tolist()[0],
                      '2018': df_to_plot.loc[[2018],:].values.tolist()[0]}, index=index)
series.plot.pie(y='2015',figsize=(9, 9), autopct='%1.1f%%', colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2015 Volume Distribution').set_ylabel('')
series.plot.pie(y='2016',figsize=(9, 9), autopct='%1.1f%%', colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2016 Volume Distribution').set_ylabel('')
series.plot.pie(y='2017',figsize=(9, 9), autopct='%1.1f%%', colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2017 Volume Distribution').set_ylabel('')
series.plot.pie(y='2018',figsize=(9, 9), autopct='%1.1f%%', colors=['silver', 'pink', 'orange', 'palegreen', 'aqua', 'blue'], fontsize=18, legend=False, title='2018 Volume Distribution').set_ylabel('')
from fbprophet import Prophet
df_pr = df.copy()
df_pr = df_pr[['Date', 'AveragePrice']].rename(columns = {'Date': 'ds', 'AveragePrice':'y'})
train_data_pr = df_pr.iloc[:len(df)-30]
test_data_pr = df_pr.iloc[len(df)-30:]
m = Prophet()
m.fit(train_data_pr)
future = m.make_future_dataframe(periods=30,freq='MS')
prophet_pred = m.predict(future)
prophet_pred.tail()
prophet_pred = pd.DataFrame({"Date" : prophet_pred[-30:]['ds'], "Pred" : prophet_pred[-30:]["yhat"]})
prophet_pred = prophet_pred.set_index("Date")
prophet_pred.index.freq = "MS"
test_data_pr["Prophet_Predictions"] = prophet_pred['Pred'].values
test_data_pr = test_data_pr.set_index("ds")
test_data_pr.head(10)
test_data_pr.tail(10)
plt.figure(figsize=(16,5))
ax = sns.lineplot(x= test_data_pr.index, y=test_data_pr["y"])
sns.lineplot(x=test_data_pr.index, y = test_data_pr["Prophet_Predictions"]);
