# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.cross_validation import KFold # use for cross validation

from sklearn.model_selection import GridSearchCV# for tuning parameter

from sklearn.ensemble import RandomForestClassifier # for random forest classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm # for Support Vector Machine

from sklearn import metrics # for the check the error and accuracy of the model

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. I like it most for plot

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/movie_metadata.csv')

type(data)
#Replace every nan values with 0

data.fillna(value=0,axis=1,inplace=True)
df_genres = pd.DataFrame(data['genres'])

df_genres = pd.DataFrame(df_genres.genres.str.split('|').tolist(),columns=['Genre_'+str(i) for i in range(0,8)])
data.drop('genres',axis=1,inplace=True)
data = pd.concat([data,df_genres],axis=1)
data.director_name.nunique()
#Select only numeric columns - Integers and floats as this would simplify the data handling

data = data.select_dtypes(include=['int64','float'])
#Define features and target

features = ['actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',

       'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',

       'num_user_for_reviews', 'budget', 'title_year',

       'actor_2_facebook_likes', 'aspect_ratio',

       'movie_facebook_likes']

target = ['imdb_score']
#Split the data into training and test data sets

train, test = train_test_split(data,test_size=0.30)
#Fill the training and test data with require information

train_x = train[features] 

train_y = train[target]

test_x = test[features]

test_y = test[target]
#Standardize the feature set

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

train_x_std = pd.DataFrame(sc.fit_transform(train_x),columns=features)

test_x_std = pd.DataFrame(sc.transform(test_x), columns=features)
train_x_std.head(2)
#draw correlation matrix of features

corr = train_x_std.corr()

#draw the heatmap of correlation to identify any features that can be dropped

plt.figure(figsize=(18,18))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

            cmap= 'YlGnBu') 
from sklearn import linear_model

model = linear_model.LinearRegression()

model.fit(train_x_std,train_y)

predicted_rating = model.predict(test_x_std)
#Calculate cross valiation scores

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, train_x_std, train_y, cv=5)

print("Linear regression accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#Calculate score of the model

r2score  = metrics.r2_score(test_y,predicted_rating)

print('R2 score is:{} best score is 1.00 and worst is 0.00'.format(round(r2score,2)))
high_score_data= data.loc[data['imdb_score']>=8] #only count of movies with score>= 8.0

sns.countplot(high_score_data['imdb_score'],label='imdb_score')
#Introducing RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)

#Convert target into a list as per the expectation of the algorithm

train_y = np.asarray(train_y, dtype="|S6")

test_y = np.asarray(test_y,dtype="|S6")

rf.fit(train_x_std,np.ravel(train_y))

predictions = rf.predict(test_x_std)
#Calculate cross valiation scores

from sklearn.model_selection import cross_val_score

print(test_y.shape)

scores = cross_val_score(rf, test_x_std,np.ravel(test_y), cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('Maximum score in cross valuation is: %0.2f'%(scores.max()))
ser = pd.Series(rf.feature_importances_,index=features)

ser.sort_values(ascending=False,inplace=True)

ser.plot(kind='bar')
#Find list of all parameters for the model. This would help us determine which ones we want to tune

rf.get_params()
from sklearn.model_selection import GridSearchCV

estimator_grid = [{'n_estimators':[100,150,200,250,500]}]

grid = GridSearchCV(rf,param_grid=estimator_grid)

grid.fit(train_x_std,np.ravel(train_y))

print('Best Params as per Grid Search:{}'.format(grid.best_params_))

print('Best score as per Grid Search{}'.format(grid.best_score_))
sns.regplot(x='gross', y='imdb_score', data=data, lowess=True)
from bs4 import BeautifulSoup

html_doc = """

<html><head><title>The Dormouse's story</title></head>

<body>

<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were

<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,

<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and

<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;

and they lived at the bottom of a well.</p>

<p class="story">...</p>

"""

soup = BeautifulSoup(html_doc, 'html.parser')
print(soup.prettify())