import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/movie_metadata.csv")
data.head()
data.describe()
data['imdb_score'].describe()
data.columns
data['imdb_score'].hist(bins=30)
corr = data.select_dtypes(include = ['float64','int64']).iloc[:,1:].corr()

plt.figure(figsize=(7, 7))

sns.heatmap(corr,vmax=1,square= True)
sns.regplot(x='num_voted_users',y='imdb_score',data=data)

data.info()
plt.figure(figsize=(1000,1000))

sns.pairplot(data,x_vars=['num_voted_users','num_user_for_reviews','duration','movie_facebook_likes','title_year','gross','director_facebook_likes','cast_total_facebook_likes','facenumber_in_poster','actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes'],y_vars=['imdb_score'])
plt.figure(figsize=(50,30))

sns.stripplot(x=data['language'],y=data['imdb_score'])
data2=data[['language','imdb_score']]

data2=data2.dropna()
plt.figure(figsize=(12,6))

sns.countplot(x='language',data=data2)

plt.figure(figsize=(12,6))

sns.countplot(x='content_rating',data=data)
plt.figure(figsize=(100,30))

sns.countplot(x='country',data=data)
plt.figure(figsize=(12,6))

sns.boxplot(x='content_rating',y='imdb_score',data=data)

plt.figure(figsize=(20,6))

sns.boxplot(x='language',y='imdb_score',data=data)
data.head()
y = data['imdb_score']

x = data.drop('imdb_score',1)
x.shape
#check missing value 

missing_data = data.isnull().sum(axis=0).reset_index()

missing_data.columns=['column_name','missing_count']

missing_data = missing_data.loc[missing_data['missing_count']>0]

missing_data = missing_data.sort_values(by='missing_count')



ind = np.arange(missing_data.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_data.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_data.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
#fill missing value of budget

budgetmean=data['budget'].mean()

data['budget']=data['budget'].fillna(budgetmean)

#fill missing value of aspect

aspect=data['aspect_ratio'].mean()

data['aspect_ratio']=data['aspect_ratio'].fillna(aspect)

#fill missing value of num_user_for_reviews

num_user_review = data['num_user_for_reviews'].mean()

data['num_user_for_reviews']=data['num_user_for_reviews'].fillna(num_user_review)

#fill missing value of duration

duration = data['duration'].mean()

data['duration']=data['duration'].fillna(num_user_review)#fill missing value of budget

#fill missing value of num_critic_for_reviews

num_critic_for_reviews = data['num_critic_for_reviews'].mean()

data['num_critic_for_reviews']=data['num_critic_for_reviews'].fillna(num_critic_for_reviews)

data.head()

y.shape
x = data[['actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',

       'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',

       'num_user_for_reviews', 'budget', 'title_year',

       'actor_2_facebook_likes', 'aspect_ratio',

       'movie_facebook_likes']].fillna(0)
y = data['imdb_score']
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
x_train.head()
regression = LinearRegression()

regression.fit(x_train,y_train)
pred=regression.predict(x_test)
error=y_test - pred

sns.distplot(error,bins=50);
plt.scatter(y_test,pred)

plt.xlabel('true value')

plt.ylabel('prediction value')
coeffecients = pd.DataFrame(regression.coef_,x.columns)

coeffecients.columns = ['Coeffecient']

coeffecients