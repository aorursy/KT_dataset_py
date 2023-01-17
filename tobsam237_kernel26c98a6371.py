# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from google.colab import files
import seaborn as sns

%matplotlib inline
#fifa19_data = files.upload()
df=pd.read_csv('../input/fifa19/data.csv')
df.head()
df.describe()
df.columns
df= df.rename(columns={"Loaned From":"Loaned_From"})
df.isna().sum()
df.Loaned_From.isnull().sum()
df=df.drop('Loaned_From', axis=1)
df.shape
df= df.dropna()
df.shape
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), )

#Comparative Vizualiztion of players count by top 40 nationalities
plt.figure(figsize=(30,7))
plt.bar(df.groupby('Nationality')['ID'].count().sort_values(ascending = False).head(40).index, df.groupby('Nationality')['ID'].count().sort_values(ascending=False).head(40), width=0.3)
df.Nationality.value_counts()[:35]
#Total number of players counts by nationality
plt.figure(figsize=(10,10))
plt.pie(df.Nationality.value_counts()[:15], labels= df.Nationality.value_counts()[:15].index, autopct='%1.2f%%', )
plt.title("By Nationality")
#sns.pairplot(df_Nig)
df_Nig = df[df['Nationality']== 'Nigeria']
df_Nig.describe()
df_Nig25= df_Nig[['Name', 'Age', 'Overall', 'Potential', 'Position', 'Club', 'Special',
       'International Reputation', 'Weak Foot', 'Skill Moves', 'Jersey Number',
       'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']].head(25)
df_Nig25[['Name', 'Age', 'Overall', 'Potential']]
# Best Nigeria Players by 'Overall' feature. These ones should be on the international team.
df_Nig25[['Age', 'Overall', 'Potential', 'Dribbling', 'Finishing', 'Crossing', 'HeadingAccuracy', 'Curve', 'Acceleration']].hist(figsize=(10,7), color ='k' )
plt.tight_layout()
#Age Skewed to right, most are quite young so, high resiliency in team
#Finishing skewed, the team quite good near the goal post
#Overall skewed to left, thus not worldclass players.
plt.scatter(df_Nig['Age'], df_Nig['Overall'])
#shows the best players in team Nig are above 21years of age
(df_Nig.groupby(['Preferred Foot'])['Name'].count())
#We have 13 left legged Nigerian players and 96 rightlegged Nigerian players
#fastest players on team Nigeria are MUSA, EJUKE, OMOH
df_NigFast=df_Nig[['Name', 'Acceleration']].sort_values('Acceleration', ascending=False)
df_NigFast.head()
#strongest players on team Nigeria are MATTHEW, NDIDI, OLAYINKA 
df_Nig[['Name', 'Stamina']].sort_values('Stamina', ascending=False).head(3)

plt.figure(figsize=(15, 7))
plt.plot(df_NigFast.head(15)["Name"],df_NigFast.head(15)["Acceleration"])
plt.ylabel("Acceleration")
plt.xlabel("Name")

plt.title("fastest Nigerian players ", fontsize = 20)
plt.show()
#visualizes acceleration differences among fastest Nigerian players
df1= df[['Name', 'Age', 'Overall', 'Potential', 'Position', 'Club', 'Special',
       'International Reputation', 'Weak Foot', 'Skill Moves', 'Jersey Number',
       'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
#df1.shape
df1.head()
df1.isna().sum().sum()
#'Overall' is our y
#df.drop overall is our x 

# df. drop Nigeria team is our train 
#df. Nig team is or test... 
#Drop unnecessary cokums and One Hot Encode other catgorical variables
df2 = pd.get_dummies(df1.drop('Name', axis=1))

#split dataset to independent and dependent datasets
x= df2.drop('Overall', axis=1)
y= df2.Overall
#We training the whole dataset for testing on Nigerian Team, we dont need the random selctive train_test_split clasiifier

x_train = x.drop(index= df_Nig.index)
y_train= y.drop(index= df_Nig.index)

x_test = df2.loc[df_Nig.index].drop('Overall', axis=1)
y_test = df2.loc[df_Nig.index].Overall
#We are ready to biuld our Linear Regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.score(x_test,y_test)
from sklearn.utils.testing import all_estimators
from sklearn import base

estimators = all_estimators()

for name, class_ in estimators:
    if issubclass(class_, base.RegressorMixin):
        print(name+"()")
n_r=0.6 
r_s=42 
np.random.seed(seed=r_s)

from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge,RidgeCV,BayesianRidge,LinearRegression,Lasso,LassoCV,ElasticNet,RANSACRegressor,HuberRegressor,PassiveAggressiveRegressor,ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import CCA
from sklearn.neural_network import MLPRegressor



my_regressors=[ 
               #ElasticNet(alpha=0.001,l1_ratio=0.70,max_iter=100,tol=0.01, random_state=r_s),
               #ElasticNetCV(l1_ratio=0.9,max_iter=100,tol=0.01,random_state=r_s),
               #GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber',random_state =r_s),
               #RandomForestRegressor(random_state=r_s),
               AdaBoostRegressor(random_state=r_s),
               #ExtraTreesRegressor(random_state=r_s),
               #SVR(C= 20, epsilon= 0.008, gamma=0.0003),
               Ridge(alpha=6),
               RidgeCV(),
               BayesianRidge(),
               DecisionTreeRegressor(),
               LinearRegression(),
               KNeighborsRegressor(),
               Lasso(alpha=0.00047,random_state=r_s),
               LassoCV(),
               #KernelRidge(),
               CCA(),
               MLPRegressor(random_state=r_s),
               HistGradientBoostingRegressor(random_state=r_s),
               HuberRegressor(),
               RANSACRegressor(random_state=r_s),
               PassiveAggressiveRegressor(random_state=r_s)
               #XGBRegressor(random_state=r_s)
              ]

regressors=[]

for my_regressor in my_regressors:
    regressors.append(my_regressor)


scores_val=[]
scores_train=[]
MAE=[]
MSE=[]
RMSE=[]


for regressor in regressors:
    scores_val.append(regressor.fit(x_train,y_train).score(x_test,y_test))
    scores_train.append(regressor.fit(x_train,y_train).score(x_train,y_train))
    y_pred=regressor.predict(x_test)
    MAE.append(mean_absolute_error(y_test,y_pred))
    MSE.append(mean_squared_error(y_test,y_pred))
    RMSE.append(np.sqrt(mean_squared_error(y_test,y_pred)))

    
results=zip(scores_val,scores_train,MAE,MSE,RMSE)
results=list(results)
results_score_val=[item[0] for item in results]
results_score_train=[item[1] for item in results]
results_MAE=[item[2] for item in results]
results_MSE=[item[3] for item in results]
results_RMSE=[item[4] for item in results]


df_results=pd.DataFrame({"Algorithms":my_regressors,"Training Score":results_score_train,"Validation Score":results_score_val,"MAE":results_MAE,"MSE":results_MSE,"RMSE":results_RMSE})
df_results
ranked_models = df_results.sort_values(by="RMSE")
ranked_models
best_model = ranked_models.iloc[0][0]
y_pred = best_model.predict(x_test)
plt.figure(figsize=(10,8))
sns.regplot(y_test, y_pred)