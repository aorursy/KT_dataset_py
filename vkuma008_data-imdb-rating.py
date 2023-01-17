import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Imputer,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings(action='ignore',message='^internal gelsd', module='scipy')

le = LabelEncoder()
scl = StandardScaler()
movies = pd.read_csv('movie_metadata.csv')
movies.head()
nf = movies._get_numeric_data().columns.values.tolist()
tf = movies.columns.values.tolist()
tf = [i for i in tf if i not in nf]
df1 = movies[nf]
df2 = movies[tf]
df1 = df1.apply(lambda x:x.fillna(x.value_counts().index[0]))
df1.head()
df1[nf] = scl.fit_transform(df1[nf])
df1 = df1[nf]
df1.head()
df2 = df2.apply(lambda x:x.fillna(x.value_counts().index[0]))
df2 = df2.apply(le.fit_transform)
df2[tf] = scl.fit_transform(df2[tf])
df2 = df2[tf]
df2.head()
df = pd.concat([df1,df2],axis=1)
df.head()
df = df[["movie_title","color","num_critic_for_reviews","movie_facebook_likes","duration","director_name","director_facebook_likes","actor_3_name","actor_3_facebook_likes","actor_2_name","actor_2_facebook_likes","actor_1_name","actor_1_facebook_likes","gross","genres","num_voted_users","cast_total_facebook_likes","facenumber_in_poster","plot_keywords","movie_imdb_link","num_user_for_reviews","language","country","content_rating","budget","title_year","aspect_ratio","imdb_score"]]
X = df.iloc[:,:-1].astype(int)
y = df.iloc[:,-1].astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 0)
clf = RandomForestClassifier(n_estimators=5)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred
metrics.accuracy_score(y_test,y_pred)
y_pred
