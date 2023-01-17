# This Python 3 environment comes with many helpful analytics libraries installed

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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



sns.set(rc={'figure.figsize':(16,12)})



import pandas_profiling

from scipy import stats

import warnings

warnings.filterwarnings(action="ignore")



from scipy.special import boxcox1p

from mlxtend.regressor import StackingCVRegressor



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, KFold

import xgboost

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import plot_importance

import lightgbm as lgb



from sklearn.preprocessing import StandardScaler, MinMaxScaler,  RobustScaler

from sklearn.pipeline import Pipeline



from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import LinearRegression
df = pd.read_csv("/kaggle/input/Data_Train.csv")

test = pd.read_csv("/kaggle/input/Data_Test.csv")



#Rearranging the way the train data is displayed to keep the dependent variable at the end, its a perfrence of mine 

df = df[['Unique_ID', 'Name', 'Genre', 'Country', 'Song_Name', 'Timestamp', 'Comments', 'Likes', 'Popularity', 'Followers','Views']]
df.head(5)



df.describe()



test.describe()



len(df[df['Views']<6216.500])
#sns.scatterplot(df[df['Views']<6216.500]['Likes'], df[df['Views']<6216.500]['Views'])
df.isnull().sum()
# The 2 lines below remove any non integer charater from the column 

df['Likes'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

df['Popularity'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')



#The 2 lines below convert the data to integer from object 

df['Likes'] = pd.to_numeric(df['Likes'])

df['Popularity'] = pd.to_numeric(df['Popularity'])



# This line is to force pandas to interpret Timestamp column as a datetime object

df['Timestamp'] = pd.to_datetime(df['Timestamp'])



# The 2 lines below remove any non integer charater from the column 

test['Likes'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

test['Popularity'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')



#The 2 lines below convert the data to integer from object 

test['Likes'] = pd.to_numeric(test['Likes'])

test['Popularity'] = pd.to_numeric(test['Popularity'])



# This line is to force pandas to interpret Timestamp column as a datetime object

test['Timestamp'] = pd.to_datetime(test['Timestamp'])
df.dtypes
# Days past since the song is realesed from today 

td = "days_past"

# The below line converts the data to timedelta

df[td] = pd.to_datetime("now") - df['Timestamp']

#since we need only the days let's access just that and conver the column to interger

df[td] = df[td].dt.days

#df[td] = pd.to_numeric(df[td])



test[td] = pd.to_datetime("now") - test['Timestamp']

test[td] = test[td].dt.days
#Pandas Profiling is a powerful library that does most of the EDA for us 



#pandas_profiling.ProfileReport(df)
num_vars = ['Comments',

            'Likes',

            'Popularity',

            'Views',

            'days_past'

]



num_vars_t = ['Comments',

            'Likes',

            'Popularity',

            'days_past'

]
# for i in num_vars: 

#     sns.distplot((df[i]), fit=stats.norm);

#     plt.figure()

#     u = stats.probplot((df[i]), plot=plt)

#     plt.figure()
for i in num_vars:

    df[i] = df[i] + 2



for i in num_vars_t:

    test[i] = test[i] + 2
# for i in num_vars: 

#     sns.distplot(np.log(df[i]), fit=stats.norm);

#     plt.figure()

#     u = stats.probplot(np.log(df[i]), plot=plt)

#     plt.figure()



# sns.set()

# sns.pairplot(df[num_vars], size = 2.5)

# plt.show();



# sns.set()

# sns.pairplot(np.log(df[num_vars]), size = 2.5)

# plt.show();
df.drop(['Unique_ID', 'Country', 'Song_Name','Timestamp','Name'], inplace=True, axis=1)



test.drop(['Unique_ID', 'Country', 'Song_Name','Timestamp','Name'], inplace=True, axis=1)



def mapping(df,col,n=25):

 print(col,n)

 vc = df[col].value_counts()

 replacements = {}

 for col, s in vc.items():

    print(col,s)

    if s<n:

        replacements[col] = 'other'

 return replacements



Genre_dic = mapping(df,'Genre',800)



#Name_dic =  mapping(df,'Name',800)



df['Genre'] = df['Genre'].replace(Genre_dic)

#df['Name'] = df['Name'].replace(Name_dic)





test['Genre'] = test['Genre'].replace(Genre_dic)

#test['Name'] = test['Name'].replace(Name_dic)
def mapping(df,col,n=25):

 print(col,n)

 vc = df[col].value_counts()

 replacements = {}

 for col, s in vc.items():

    print(col,s)

    if s<n:

        replacements[col] = 'other'

 return replacements



Genre_dic = mapping(df,'Genre',800)



#Name_dic =  mapping(df,'Name',800)



df['Genre'] = df['Genre'].replace(Genre_dic)

#df['Name'] = df['Name'].replace(Name_dic)





test['Genre'] = test['Genre'].replace(Genre_dic)

#test['Name'] = test['Name'].replace(Name_dic)
def Pipeline_func(scaler, algo, param_grd=False):

    #steps = [('scaler', StandardScaler()), ('RF', RandomForestRegressor())]

    steps = [('scaler',scaler), ('Algo', algo)]

    

    pipeline = Pipeline(steps)

    if param_grd:

        return GridSearchCV(pipeline, param_grid=param_grd, cv=5)

    else:

        return pipeline

#     Rf_parameter1 = {'RF__min_samples_split' : [2],

#                  'RF__min_samples_leaf': [1]

#                 }



    
# Rf_param = {'Algo__min_samples_split' : [2],

#                  'Algo__min_samples_leaf': [1]

#                 }

# test_Pipeline = Pipeline_func(StandardScaler(), RandomForestRegressor(), param_grd=Rf_param)

# pipeline2 =  Pipeline(steps2)

# GBR_parameter = {'GBR__n_estimators':range(20,81,10),

#                  'GBR__max_depth':range(5,16,2), 

#                  #'min_samples_split':range(200,1001,200),

#                  #'min_samples_split':range(1000,2100,200), 

#                  #'min_samples_leaf':range(30,71,10),

#                  #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],

                 

                 

# }

# grid2 = GridSearchCV(pipeline2, param_grid=GBR_parameter, cv=5)

df = pd.get_dummies(df)

test = pd.get_dummies(test)



# df.drop(['Genre'], inplace=True, axis=1)

# test.drop(['Genre'], inplace=True, axis=1)
# sns.scatterplot(stats.boxcox(df['Comments'])[0], stats.boxcox(df['Views'])[0], color='red')



# sns.scatterplot(df[df['Comments']<25000]['Comments'],(df[df['Comments']<25000]['Views']) )



#df = df[df['Comments']<25000]
def fitmodel(alg,tr_x,tr_y,val_x,val_y):

    alg.fit(tr_x,tr_y)

    print((np.sqrt(mean_squared_error(alg.predict(val_x),val_y))))
def rmse(x,y):

    print(np.sqrt(mean_squared_error(x,y)))

    return np.sqrt(mean_squared_error(x,y))
train_x, valid_x, train_y, valid_y = train_test_split(df.drop(["Views"], axis=1), df['Views'], test_size = 0.10,

                                                      random_state =42 )
print("Training set size", train_x.shape)

print("Validation set size", valid_x.shape)
# eval_set = [(valid_x, valid_y)]

# model.fit(train_x, train_y, eval_metric="rmse", eval_set=eval_set, verbose=True)
gb = GradientBoostingRegressor()

fitmodel(gb, train_x, train_y, valid_x, valid_y)



GB_PP1 = Pipeline_func(scaler=StandardScaler(), algo=gb)

fitmodel(GB_PP1, train_x, train_y, valid_x, valid_y)



GB_PP2 = Pipeline_func(scaler=RobustScaler(), algo=gb)

fitmodel(GB_PP2, train_x, train_y, valid_x, valid_y)
gb.get_params
feat_importances = pd.Series(GB_PP1.steps[1][1].feature_importances_, index=df.drop(['Views'], axis=1).columns)

feat_importances.nlargest(10).plot(kind='barh')

#gb.feature_importances_
br = BaggingRegressor()



#     base_estimator=None, 

#                       n_estimators=50, 

#                       max_samples=1.0, 

#                       max_features=1.0, 

#                       bootstrap=True, 

#                       bootstrap_features=False, 

#                       oob_score=False, 

#                       warm_start=False, 

#                       n_jobs=1, 

#                       random_state=1, 

#                       verbose=0)

fitmodel(br, train_x, train_y, valid_x, valid_y)



BR_PP1 = Pipeline_func(scaler=StandardScaler(), algo=br)

fitmodel(BR_PP1, train_x, train_y, valid_x, valid_y)



BR_PP2 = Pipeline_func(scaler=RobustScaler(), algo=br)

fitmodel(BR_PP2, train_x, train_y, valid_x, valid_y)
br.get_params
# feat_importances = pd.Series(Rf.feature_importances_, index=df.drop(['Views'], axis=1).columns)

# feat_importances.nlargest(10).plot(kind='barh')
Rf = RandomForestRegressor(max_depth=11, n_estimators=15, min_samples_leaf=1,min_samples_split=2,)#verbose=2,)

fitmodel(Rf, train_x, train_y, valid_x, valid_y)



Rf_PP1 = Pipeline_func(scaler=StandardScaler(), algo=Rf)

fitmodel(Rf_PP1, train_x, train_y, valid_x, valid_y)



Rf_PP2 = Pipeline_func(scaler=RobustScaler(), algo=Rf)

fitmodel(Rf_PP2, train_x, train_y, valid_x, valid_y)

#RandomForestRegressor(max_depth=8,n_estimators=40)
Rf.get_params
feat_importances = pd.Series(Rf_PP2.steps[1][1].feature_importances_, index=df.drop(['Views'], axis=1).columns)

feat_importances.nlargest(10).plot(kind='barh')
xg = xgboost.XGBRegressor(objective='reg:squarederror')

fitmodel(xg, train_x, train_y, valid_x, valid_y)



XG_PP1 = Pipeline_func(scaler=StandardScaler(), algo=xg)

fitmodel(XG_PP1, train_x, train_y, valid_x, valid_y)



XG_PP2 = Pipeline_func(scaler=RobustScaler(), algo=xg)

fitmodel(XG_PP2, train_x, train_y, valid_x, valid_y)
sub_file = pd.read_excel("/kaggle/input/Sample_Submission.xlsx")



print(sub_file['Views'])
sub_file['Views'] = XG_PP2.predict(test)



sub_file['Views'] = sub_file['Views'].astype(int)

print(sub_file['Views'])



sub_file.to_excel("sub.xlsx", index = False)