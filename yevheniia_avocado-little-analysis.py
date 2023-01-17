import pandas as pd
import numpy as np
data = pd.read_csv('../input/avocado.csv',index_col=0)
data.head()
data.info()
data['type'].unique()
import matplotlib.pyplot as plt
import seaborn as sns
region_mean = data['AveragePrice'].groupby(data['region']).mean()
region_mean.head()
region = data['region'].unique()
stat_reg = { region[i]: region_mean[i] for i in range(len(region))}
import operator
sorted_d = sorted(stat_reg.items(), key=operator.itemgetter(1))
X = [ sorted_d[i][0] for i in range(len(sorted_d))]
Y = [ sorted_d[i][1] for i in range(len(sorted_d))]
plt.figure(figsize=(20,10))
ax=sns.barplot(x=X,y=Y,palette="Reds")
plt.xticks(rotation=86)
plt.xlabel('Region')
plt.tick_params(axis = 'both', labelsize = 14)
plt.ylabel('Average Price')
plt.title('Average Price of Avocado According to Region')
region_mean_volume = data['Total Volume'].groupby(data['region']).mean()
stat_reg_vol = { region[i]: region_mean_volume[i] for i in range(len(region))}
del(stat_reg_vol['TotalUS'])
sorted_d = sorted(stat_reg_vol.items(), key=operator.itemgetter(1))
X = [ sorted_d[i][0] for i in range(len(sorted_d))]
Y = [ sorted_d[i][1] for i in range(len(sorted_d))]
plt.figure(figsize=(20,10))
ax=sns.barplot(x=X,y=Y,palette='Reds')

plt.xticks(rotation=86)
plt.xlabel('Region')
plt.tick_params(axis = 'both', labelsize = 14)
plt.ylabel('Average volume')
plt.title('Average volume of Avocado According to Region')
new_region = [region[i] for i in range(len(region)) if region[i]!= 'TotalUS']


del(region_mean['TotalUS'])
del(region_mean_volume['TotalUS'])

mean_value_price = [ region_mean[i] for i in range(len(region_mean))]
mean_value_volume = [ region_mean_volume[i] for i in range(len(region_mean_volume))]
fig, ax1 = plt.subplots(figsize=(20,10))
ax1.set_ylabel('Average price', color = 'coral',fontsize=20)
ax1.set_xlabel('Region', color = 'coral',fontsize=20)
plt.xticks(rotation=86)
ax1.tick_params(axis = 'both', labelsize = 15)


ax1.plot(new_region, mean_value_price,  marker='o', markerfacecolor='#5e0000', markersize=12, color='#fd742d', linewidth=5)
ax1.grid(True)



ax2 = ax1.twinx()
ax2.set_ylabel('Average volume', color='#738830', fontsize=20)
ax2.plot( new_region, mean_value_volume,  marker='o', markerfacecolor='#5e0000', markersize=12, color='#738830', linewidth=5)

fig.tight_layout() 
plt.show()
type_mean = data['AveragePrice'].groupby(data['type']).mean()
df_type_mean = pd.DataFrame({'type':  data['type'].unique(),'AveragePrice':[type_mean[0],type_mean[1]]})
df_type_mean
f = plt.figure(figsize=(15,8))
sns.barplot(x=df_type_mean['type'],y = df_type_mean['AveragePrice'] , palette='rocket' )
plt.show()
type_mean = data['Total Volume'].groupby(data['type']).mean()
df_type_mean = pd.DataFrame({'type':  data['type'].unique(),'AverageVolume':[type_mean[0],type_mean[1]]})
f = plt.figure(figsize=(15,8))
sns.barplot(x=df_type_mean['type'],y = df_type_mean['AverageVolume'] , palette='rocket' )
plt.show()
f = plt.figure(figsize=(15,8))
sns.distplot(data['AveragePrice'], bins = 25,color='#fd742d' )
f = plt.figure(figsize=(15,8))
sns.regplot(x=data["Total Bags"], y=data["Small Bags"],color='#5e0000',line_kws={"color":"r","alpha":0.7,"lw":2})
f = plt.figure(figsize=(15,8))
f = sns.boxplot(x="region",y = "AveragePrice",order = region[0:4],data= data )
f = sns.stripplot(x="region",y = "AveragePrice",data= data,order = region[0:4], color="orange", jitter=0.2, size=3)
plt.title("Boxplot with jitter", loc="left")
data.isnull().sum()
f = plt.figure(figsize=(15,8))
sns.heatmap(data.isnull(),yticklabels=False,cmap= 'rocket')
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

model = LinearRegression()

not_in = ['Date','region','type']
col= [list(data.head(0))[i] for i in range(12) if list(data.head(0))[i] not in not_in]

df_x = data[col]
df_y = data['AveragePrice']
del(df_x['AveragePrice'])

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

model.fit(x_train, y_train)

prediction = model.predict(x_test)
rmsle(prediction,y_test) # accuracy with rmsle
from sklearn.neighbors import KNeighborsClassifier

not_in = ['Date','region','type']
col= [list(data.head(0))[i] for i in range(12) if list(data.head(0))[i] not in not_in]
_type = pd.get_dummies(data['type'])

df_x = data[col]
df_y = _type['conventional']

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

model_knn = KNeighborsClassifier(n_neighbors=3)

model_knn.fit(x_train,y_train)

pred = model_knn.predict(x_test)

rmsle(pred,y_test)  # accuracy with rmsle
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from sklearn.linear_model import LogisticRegression

df_x = data[col]
df_y = _type['conventional']

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

prediction = logreg.predict(x_test)
rmsle(prediction,y_test)