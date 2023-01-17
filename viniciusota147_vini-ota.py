import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import datetime as dt
from matplotlib import pylab
plt.rcParams["figure.figsize"] = [16,9]
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder
from sklearn import base
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from h2o.automl import H2OAutoML

## Importando biblioteca h2o para utilizar o Target Encoder:
import h2o
h2o.init()
from h2o.estimators import H2OTargetEncoderEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/train.csv.zip")
features_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/features.csv.zip")
stores_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/stores.csv")
test_df = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/test.csv.zip")
def columns_to_lower(data):
    return data.columns.str.lower()


train_df.columns = columns_to_lower(train_df)
features_df.columns = columns_to_lower(features_df)
stores_df.columns = columns_to_lower(stores_df)
test_df.columns = columns_to_lower(test_df)
dataset = pd.DataFrame()

print(train_df.shape)
dataset = train_df.merge(features_df.drop("isholiday",axis = 1) , how = "left" , on = ["date","store"])
print(dataset.shape)
dataset = dataset.merge(stores_df,how='left', on = ["store"])
print(dataset.shape)

dataset['store'] = dataset['store'].astype('object')
dataset['dept']  = dataset['dept'].astype('object')
dataset['year'] = dataset['date'].str[:4]
dataset['date']  = pd.to_datetime(dataset['date'])
dataset['month'] = dataset['date'].dt.month

features_df['date'] = pd.to_datetime(features_df['date'])
features_df['week'] = features_df['date'].dt.isocalendar()['week']
features_df['year'] = features_df['date'].dt.isocalendar()['year']
pd.DataFrame({'missing_quantity' : dataset.isnull().sum().sort_values(ascending = False),
              'missing_pct' : dataset.isnull().mean().sort_values(ascending = False)})
dataset['weekly_sales'].describe().T.to_frame()
def target_description_plot(df):

    plt.figure(figsize = (25,8))

    plt.subplot(131)
    sns.lineplot(x = 'date' , y = 'weekly_sales' , data = df )
    plt.title("Time Series from Weekly Sales")

    plt.subplot(132)
    sns.distplot(df['weekly_sales'],hist = True )
    plt.title("Density from Weekly Sales")

    plt.subplot(133)
    sns.boxplot(y = df['weekly_sales'])
    plt.title("Boxplot from Weekly Sales") ; 
target_description_plot(dataset)
dataset.groupby(['date'])['weekly_sales'].mean().plot()
dataset.groupby(['date'])['weekly_sales'].median().plot()
plt.legend(['Media','Mediana']) 
plt.title("Media e Mediana de vendas ao longo do tempo")
plt.ylabel("Vendas"); 

for i in ['temperature','fuel_price', 'cpi', 'unemployment']:

    plt.figure(figsize = (30,6))

    plt.subplot(131)
    sns.lineplot(x = 'date', y =  i , data = features_df,ci=None)
    plt.title( i + " ao longo do tempo"  )
    plt.xticks(rotation = 90)
    
    plt.subplot(132)
    sns.boxplot(y =  i , data = features_df)
    plt.title("Boxplot " + i)
   

    plt.subplot(133)
    sns.distplot(features_df[i].dropna())

    plt.show() ; 
stores_df['type'].value_counts().plot(kind='bar') 
plt.title("Quantidade de cada tipo de loja")  
plt.ylabel("Quantidade") ; 
plt.figure(figsize = (30,6))

plt.subplot(121)
sns.boxplot(y = stores_df['size'])
plt.title("Boxplot - Tamanho da Loja")

plt.subplot(122)
sns.distplot(stores_df['size'])
plt.title("Densidade - Tamanho da loja"); 
num_columns = dataset.select_dtypes(['int','float64']).columns
# multicolinearidade
plt.figure(figsize = (20,5))
sns.heatmap(data = dataset[num_columns].corr() , annot=True) ;
def plot_scatter( var ):
    if var != 'weekly_sales':
        sns.scatterplot(x = var , y = 'weekly_sales' , data = dataset)
        plt.title(var + " weekly_sales")
        plt.show() ; 
for i in num_columns:
    plot_scatter(i) ; 
def cat_plot(var_cat):
    sns.boxplot(x = var_cat , y = 'weekly_sales' , data = dataset) 
    plt.title("Boxplot weekly sales por " + var_cat) 
    plt.show(); 
categorical_features = dataset.select_dtypes(['object']).columns
for i in categorical_features:
    cat_plot(i)
plt.figure(figsize = (25,8))
plt.subplot(121)
sns.lineplot(x = 'date',y = 'weekly_sales',hue='type',data=dataset.groupby(['date','store','type'])['weekly_sales'].median().reset_index(),ci = None)


plt.subplot(122)
sns.boxplot(x = 'type',
            y = 'size' ,
            data = dataset) 
plt.title("Boxplot: Tamanho da loja por tipo de loja "); 
def getSeason(x):
    
    if (x > 11 or x <= 3):
        return "winter"
    elif ( x == 4 or x == 5):
        return "spring"
    elif (x >=6 and x <= 9):
        return "summer"
    else:
        return "fall"
    
dataset['season'] = dataset['month'].apply(getSeason)
"Como visto, a variável departamento apresenta {} categorias distintas para tentar melhorar isso, vamos tentar agrupa-las de alguma maneira".format(dataset['dept'].nunique())
train = dataset.query("date < '2011-10-01'")
test  = dataset.query("date >= '2011-10-01'") 
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
train.groupby(['dept'])['weekly_sales'].median().sort_values().plot(kind='bar') ; 
def group_dept(x):
    if x < 10000:
        return 'C'
    elif x <20000:
        return 'B'
    else:
        return 'A'

    
    
train_dept = train.groupby(['dept'])['weekly_sales'].median().sort_values().reset_index()
train_dept['grupo_dept'] = train_dept['weekly_sales'].apply(group_dept)
train = train.merge(train_dept[['dept','grupo_dept']].drop_duplicates(),on = 'dept')
test  = test.merge(train_dept[['dept','grupo_dept']].drop_duplicates(),on='dept')
def WMAE(data, real, predicted):
    weights = data['isholiday_True'].apply(lambda x: 5 if x == 1 else 1)

    return np.round(np.sum(weights*abs(real-predicted))/(np.sum(weights)), 2)



# Erro personalizado para utilizar no grid search:
erro = make_scorer(WMAE,greater_is_better = False)
def prepdata(data ,categorical_var,continuous_var):
    
    
    onehot = OneHotEncoder(handle_unknown='error',sparse = False,drop='first')
    onehot.fit(data[categorical_var])
    df_cat = pd.DataFrame( onehot.transform( data[categorical_var] ) , columns = onehot.get_feature_names(categorical_var) )
    
    Standard = StandardScaler()
    Standard.fit(train[numerical_features]) 
    df_standard = pd.DataFrame( Standard.transform(data[numerical_features]) , columns = list(numerical_features) )
    
    data_train = pd.concat( [df_standard,df_cat] , axis = 1 )
    data_train['target'] = data[target]
    
    X_train = data_train.drop('target',axis=1)
    y_train = data_train['target']
    
    
    return X_train,y_train

def fit_model(X,y):
    
    fit_LR = LinearRegression().fit( X , np.ravel(y) )
    fit_knn = KNeighborsRegressor( n_neighbors = 10 ).fit( X, np.ravel(y) )
    fit_Rf = RandomForestRegressor(min_samples_split=50,max_depth=10).fit( X , np.ravel(y) )
    fit_GB = GradientBoostingRegressor(min_samples_split=50,max_depth=10).fit( X , np.ravel(y) )
    
    
    return fit_LR,fit_knn,fit_Rf,fit_GB


def test_model(data ,categorical_var,continuous_var):
    
    onehot = OneHotEncoder(handle_unknown='error',sparse = False,drop='first')
    onehot.fit(train[categorical_var])
    df_cat = pd.DataFrame( onehot.transform( data[categorical_var] ) , columns = onehot.get_feature_names(categorical_var) )
    
    Standard = StandardScaler()
    Standard.fit(data[numerical_features]) 
    df_standard = pd.DataFrame( Standard.transform(data[numerical_features]) , columns = list(numerical_features) )
    
    data_test = pd.concat( [df_standard,df_cat] , axis = 1 )
    data_test['target'] = data[target]
    
       
    return data_test

categorical_features = ['store','isholiday','grupo_dept','type']
numerical_features = ['temperature','fuel_price','markdown1','markdown2','markdown3','markdown5','cpi','unemployment','size']

target = 'weekly_sales'
X_train,y_train = prepdata(train,categorical_features,numerical_features)


fitted_models = []
fitted_models = fit_model(X_train,y_train)

##################################################################################################################################

test_df = test_model(test,categorical_features,numerical_features)
X_test = test_df.drop("target",axis=1)
y_test = test_df.drop("target",axis=1)

##################################################################################################################################

list_erro = []
list_model = []
for i in fitted_models:
        list_erro.append( WMAE(test_df,test_df['target'],i.predict(X_test) ))
        list_model.append( str(i) )


        
display( pd.DataFrame(list_erro,list_model) )
pd.DataFrame(fitted_models[2].feature_importances_,X_train.columns).sort_values(by = 0 , ascending = False).plot(kind = 'bar') ; 

categorical_features = ['store','isholiday','season','grupo_dept','type']
numerical_features = ['temperature','fuel_price','cpi','unemployment','size']
target = 'weekly_sales'

##################################################################################################################################

X_train,y_train = prepdata(train,categorical_features,numerical_features)
fitted_models = []
fitted_models = fit_model(X_train,y_train)

##################################################################################################################################

test_df = test_model(test,categorical_features,numerical_features)
X_test = test_df.drop("target",axis=1)
y_test = test_df.drop("target",axis=1)

list_erro = []
list_model = []
for i in fitted_models:
        list_erro.append( WMAE(test_df,test_df['target'],i.predict(X_test) ))
        list_model.append( str(i) )


        
display(pd.DataFrame(list_erro,list_model) )

x_columns = ['temperature',
             'fuel_price',
             'markdown1',
             'markdown2',
             'markdown3',
             'markdown5',
             'cpi',
             'unemployment',
             'size',
             'store',
             'isholiday',
             'season',
             'grupo_dept',
             'type']

y_column = 'weekly_sales'
# # Identify predictors and response
# #train = h2o.H2OFrame(train)
# x = x_columns
# y = y_column


# # Run AutoML for 20 base models (limited to 1 hour max runtime by default)
# aml = H2OAutoML(max_models= 8 , seed=1)
# aml.train(x=x, y=y, btraining_frame=train)

# # View the AutoML Leaderboard
# lb = aml.leaderboard
# lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)