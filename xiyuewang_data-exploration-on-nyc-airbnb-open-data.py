import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge



%matplotlib inline

sns.set_style("darkgrid")
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

df.sample(5)
df.info()
df.dtypes
# Fix datatype

df['last_review'] = pd.to_datetime(df['last_review'])
# dealing with missing data

df.isnull().sum()
df['name'].fillna('None', inplace=True)

df['host_name'].fillna('None', inplace=True)

df['last_review'].fillna('None', inplace=True)
review_avg = df['reviews_per_month'].mean()

df['reviews_per_month'].fillna(review_avg, inplace = True)
df.isnull().any()
plt.figure(figsize=(8, 6))

sns.distplot(df['price'], bins=200, kde=True)

plt.xscale('log') # Log transform the price 

plt.xticks([100, 200, 500, 1000, 10000], ['100', '200', '500', '1k', '10k'])

plt.ylabel('Percentage', fontsize=12)

plt.xlabel('Price (dollar)', fontsize=12)

plt.title('Listed Price Distribution', fontsize=14);
plt.figure(figsize=(8, 6))

sns.distplot(df['reviews_per_month'], kde=True)

plt.ylabel('Percentage', fontsize=12)

plt.xlabel('Reviews per Month', fontsize=12)

plt.title('Reviews per Month Distribution', fontsize=14);
neighbourhood_counts = df['neighbourhood_group'].value_counts()

plt.figure(figsize=(8, 8))

sns.barplot(neighbourhood_counts.index, neighbourhood_counts.values, palette='RdBu')

plt.xlabel('Neighbourhood', fontsize=12)

plt.ylabel('Counts', fontsize=12)

plt.title('Popular Neighbourhood', fontsize=14);
df.minimum_nights.describe()

plt.figure(figsize=(8, 6))

sns.countplot(data=df, x='minimum_nights')

plt.xlim(0, 40)

plt.ylabel('Percentage', fontsize=12)

plt.xlabel('Minimum Stayed', fontsize=12)

tick = [1,5,10,15,20,25,30,35,40]

plt.xticks(tick, tick)

plt.title('Minimum Stay Distribution', fontsize=14);
plt.figure(figsize=(8, 6))

sns.regplot(data=df, x='reviews_per_month', y='price', color=sns.color_palette()[0], 

                x_jitter = 0.04, y_jitter = 0.04, fit_reg=False, scatter_kws={'alpha': 0.5})

plt.yscale('log')

plt.yticks([100, 200, 500, 1000, 10000], ['100', '200', '500', '1k', '10k'])

plt.xlabel('Review Per Month', fontsize=12)

plt.ylabel('Price (dollar)', fontsize=12)

plt.xlim(0, 20)

plt.title('Review VS. Price', fontsize=14);
plt.figure(figsize=(8, 6))

sns.barplot(data=df, x='room_type', y='price', color=sns.color_palette()[0]);

plt.xlabel('Room Type', fontsize=12)

plt.ylabel('Price (dollar)', fontsize=12)

plt.title('Price for Different Room Type', fontsize=14);
df['price'].describe()

# segment price into two groups 

bin_edges = [0, 106, 10000]

bin_name = ['low', 'high']

df['price_bin'] = pd.cut(df['price'], bins=bin_edges, labels=bin_name)



# get the post content for each price group

low = ''

df_low = df[df['price_bin'] == 'low']['name'].astype(str)

for i in range(len(df_low)):

    words = df_low.iloc[i].split(' ')

    for word in words:

        low += word+' '

        

high = ''

df_high = df[df['price_bin'] == 'high']['name'].astype(str)

for i in range(len(df_high)):

    words = df_high.iloc[i].split(' ')

    for word in words:

        high += word+' '
# Get the most popular 5 words for each price group 

from collections import Counter

low_Counter = Counter(low.lower().split(' '))

low_occur = low_Counter.most_common(5)



high_Counter = Counter(high.lower().split(' '))

high_occur = high_Counter.most_common(5)



low_df = pd.DataFrame(low_occur, columns=['word', 'count'])

low_df = low_df.iloc[1:,:]



high_df = pd.DataFrame(high_occur, columns=['word', 'count'])

high_df = high_df.iloc[1:,:]
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

ax1 = sns.barplot(data=low_df, x='word', y='count', palette = "BuGn_d")

plt.title('Lower Price Group', fontsize=12);



plt.subplot(1, 2, 2)

ax2 = sns.barplot(data=high_df, x='word', y='count', palette = "BuGn_d")

plt.title('Higher Price Group', fontsize=12)

plt.suptitle('Popular Words in Different Price Group');
plt.figure(figsize=(10, 8))

sns.barplot(data=df, x='neighbourhood_group', y='price', hue='room_type', palette=sns.cubehelix_palette(8))

plt.xlabel('Neighbourhood', fontsize=12)

plt.ylabel('Price (dollar)', fontsize=12)

plt.title('Price for Different Neighbourhood and Room Types', fontsize=14)

plt.legend(loc=1, framealpha=0, title='Room Type');
df2 = df[df['price'] <= 200]

fig, ax = plt.subplots(figsize=(12,12))



# Show Background image

img=plt.imread('../input/new-york-city-airbnb-open-data/New_York_City_.png', 0)

coordenates_to_extent = [-74.258, -73.7, 40.49, 40.92]

ax.imshow(img, zorder=0, extent=coordenates_to_extent)



# Plotting

scatter_map = plt.scatter(data=df2, x='longitude', y='latitude', c='price', alpha=0.3)

plt.colorbar(shrink = 0.5)

ax.grid(True)

plt.title('Price Map', fontsize=14)

plt.show()
plt.figure(figsize=(12, 12))

sns.heatmap(df.corr(), square=True, annot=True, fmt = '.2f', cmap = 'vlag_r', center=0);
# Trim the data for prediction

cols = ['host_id', 'neighbourhood_group', 'longitude', 'room_type', 'price', 'minimum_nights', 'reviews_per_month', 'availability_365']

df_clean = df.loc[:, cols]

df_clean.head()
# get dummy value

le = preprocessing.LabelEncoder()

le.fit(df_clean['neighbourhood_group'])

df_clean['neighbourhood_group']=le.transform(df_clean['neighbourhood_group'])



le.fit(df_clean['room_type'])

df_clean['room_type']=le.transform(df_clean['room_type'])
# build model

lm = LinearRegression()

X = df_clean.drop(['host_id', 'price'], axis=1)

y = df_clean['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lm.fit(X_train, y_train)



y_pred = lm.predict(X_test)
# Evaluate

mse = metrics.mean_squared_error(y_test, y_pred)

r_square = metrics.r2_score(y_test, y_pred)



print('Mean absolute error is {}'.format(mse))

print('R^2 is {}'.format(r_square))
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):

    plt.figure(figsize=(10, 8))



    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)

    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)

    plt.xlabel('Price (dollars)')

    plt.show()

    plt.close()
# Visualize the results

DistributionPlot(y_test, y_pred, 'Actual Values (Train)', 'Predicted Values (Train)', 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution')
Rcross = cross_val_score(lm, X, y, cv=4)

print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
# test polynomial orders

R_square = []

order = [1, 2, 3, 4, 5, 6, 7]

for n in order:

    pr = PolynomialFeatures(degree=n)    

    X_train_pr = pr.fit_transform(X_train) 

    X_test_pr = pr.fit_transform(X_test)    

    

    lm.fit(X_train_pr, y_train)

    

    R_square.append(lm.score(X_test_pr, y_test))



plt.plot(order, R_square)

plt.xlabel('order')

plt.ylabel('R^2')

plt.ylim(0, 0.2)

plt.title('R^2 Using Test Data')

plt.text(4, 0.17, 'Maximum R^2');    
# fit the data with forth order polynomial

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pr = PolynomialFeatures(degree=4)

X_train_pr = pr.fit_transform(X_train) 

X_test_pr = pr.fit_transform(X_test)  

lm.fit(X_train_pr, y_train)

print('R^2 is {}'.format(lm.score(X_test_pr, y_test)))

y_pred = lm.predict(X_test_pr)
# plot the new fitting

DistributionPlot(y_test, y_pred, 'Actual Values (Train)', 'Predicted Values (Train)', 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution')
RR_square = []

RR_train = []

dummy = []

alpha = [0, 0.001, 0.01, 1, 10, 100]

for a in alpha:

    RigeModel = Ridge(alpha=a) 

    RigeModel.fit(X_train_pr, y_train)

    RR_square.append(RigeModel.score(X_test_pr, y_test))

    RR_train.append(RigeModel.score(X_train_pr, y_train))
plt.figure(figsize=(8, 5))

plt.plot(alpha,RR_square, label='validation data')

plt.plot(alpha,RR_train, 'r', label='training Data')

plt.xlabel('alpha')

plt.ylabel('R^2')

plt.ylim(0, 0.2)

plt.legend();
# Best bridge model

RR = Ridge(alpha=0.15)

RR.fit(X_train_pr, y_train)

y_RR = RR.predict(X_test_pr)

print('R^2 is {}'.format(metrics.r2_score(y_test, y_RR)))
DistributionPlot(y_test, y_pred, 'Actual Values (Train)', 'Predicted Values (Train)', 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution')
plt.figure(figsize=(10,5))

sns.regplot(x=y_test, y=y_pred, color=sns.color_palette()[0])

plt.xlim(0, 2000)

plt.title('Predict Model', fontsize=14)

plt.xlabel('Test Data')

plt.ylabel('Predictions');