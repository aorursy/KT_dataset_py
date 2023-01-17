import pandas as pd
import math
import re
from sklearn.preprocessing import StandardScaler
import numpy as np
df=pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")
df
df.isnull().sum()
ls=list()
ls=round(100*(df.isnull().sum()/len(df.index)),4)
ls
df=df.drop("society",axis=1)
df["location"].value_counts()
df["location"].fillna("Whitefield",inplace=True)
df['size'].value_counts()
df["size"].fillna("2 BHK",inplace=True)
df.isnull().sum()
df.describe()
df["bath"].fillna(2.692610,inplace=True)
df["balcony"].fillna(1.584376,inplace=True)
df.isnull().sum()
df['size'] = df['size'].str[0:2]
df['size']=df['size'].astype('int')
df
df.info()

df.total_sqft.unique()
def convert_total_sqft(my_list):
    if len(my_list) == 1:
        
        try:
            return float(my_list[0])
        except:
            ls = ['Sq. Meter', 'Sq. Yards', 'Perch', 'Acres', 'Cents', 'Guntha', 'Grounds']
            split_list = re.split('(\d*.*\d)', my_list[0])[1:]
            area = float(split_list[0])
            type_of_area = split_list[1]
            if type_of_area == 'Sq. Meter':
                area_in_sqft = area * 10.7639
            elif type_of_area == 'Sq. Yards':
                area_in_sqft = area * 9.0
            elif type_of_area == 'Perch':
                area_in_sqft = area * 272.25
            elif type_of_area == 'Acres':
                area_in_sqft = area * 43560.0
            elif type_of_area == 'Cents':
                area_in_sqft = area * 435.61545
            elif type_of_area == 'Guntha':
                area_in_sqft = area * 1089.0
            elif type_of_area == 'Grounds':
                area_in_sqft = area * 2400.0
            return float(area_in_sqft)
        
    else:
        return (float(my_list[0]) + float(my_list[1]))/2.0
df['total_sqft'] = df.total_sqft.str.split('-').apply(convert_total_sqft)
df = df.drop(df[df['bath']>5].index)
df = df.drop(df[df['size']>6].index)
df['price_per_sqft'] = df['price']*100000/df['total_sqft']
df
def remove_outlier(df1):
    new_dataframe = pd.DataFrame()
    for key, df2 in df1.groupby('location'):
        m = np.mean(df2.price_per_sqft)
        st = np.std(df2.price_per_sqft)
        reduced_df = df2[(df2.price_per_sqft>(m-st)) & (df2.price_per_sqft<=(m+st))]
        new_dataframe = pd.concat([new_dataframe,reduced_df],ignore_index=True)
    return new_dataframe
    
df = remove_outlier(df)
df.shape
df = remove_outlier(df)
df.drop(columns=['availability','area_type'],inplace = True)
df.location = df.location.str.strip()
location_count = df['location'].value_counts(ascending=False)
location_stats_less_than_8 = location_count[location_count<=8]
df.location = df.location.apply(lambda x: 'other' if x in location_stats_less_than_8 else x)
df = df[df.location != 'other']
df.shape
location = pd.get_dummies(df.location)
df = pd.concat([df,location],axis='columns')
df.shape
df1 = df.drop('location',axis = 1)
df1 = df1.drop(columns=['balcony','price_per_sqft'])
df1
x=df1.drop("price",axis=1)
y=df1["price"]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.4)
x
!pip install lazypredict
from lazypredict.Supervised import LazyRegressor
reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
models,predictions = reg.fit(xtrain, xtest, ytrain, ytest)
models
from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(random_state=1, max_iter=1000).fit(xtrain, ytrain)
print("Train Accuracy ",regr.score(xtrain,ytrain)*100)
print("Test Accuracy ",regr.score(xtest, ytest)*100)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
model_params = {
    'MLP': {
        'model': MLPRegressor(),
        'params' : {
            'max_iter':[1000],
            'random_state':[1]
        }  
    },
    'random_forest': {
        'model': RandomForestRegressor(),
        'params' : {
            'n_estimators': [1,5,10,20,25]
        }
    },
    'GradientBoosting':{
        'model':GradientBoostingRegressor(random_state=0),
        'params':{
            'n_estimators':[1500]
        }
    }
}
scores=[]
for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(xtrain,ytrain)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
df2 = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df2
