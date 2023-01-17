import warnings

warnings.filterwarnings("ignore")



#Data Manipulation and Treatment

import numpy as np

import pandas as pd

from datetime import datetime



#Plotting and Visualizations

import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns

from scipy import stats

import itertools



#Scikit-Learn for Modeling

from sklearn import model_selection

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
def str_to_date(date):

    return datetime.strptime(date, '%Y-%m-%d').date()
#The training Set

df_train = pd.read_csv("../input/ml-training-vlib/train.csv",sep=',', parse_dates=['Date']

                       , date_parser=str_to_date,

                       low_memory = False)





#Additional Information on those stores 

df_store = pd.read_csv("../input/ml-training-vlib/store.csv"

                       , low_memory = False)
df_train.head(20) 
#df_train[df_train["Store"] == 1]
df_train.dtypes,print ("The Train dataset has {} Rows and {} Variables".format(str(df_train.shape[0]),str(df_train.shape[1])))
df_store.head()
df_store.dtypes ,print ("The Store dataset has {} Rows (which means unique Shops) and {} Variables".format(str(df_store.shape[0]),str(df_store.shape[1]))) 
df_train.isna().sum()
print ()

print ("-Durant ces 2 années, {} est le nombre de jours que les magasins ont fermés.".format(df_train[(df_train.Open == 0)].count()[0]))

print ()

print ("-Parmi ces fermetures, {} sont dues aux vacances scolaires. " .format(df_train[(df_train.Open == 0) & (df_train.SchoolHoliday == 1)&(df_train.StateHoliday == '0') ].count()[0]))

print ()

print ("-et {} sont dues aux vacances 'publique'.".format(df_train[(df_train.Open == 0) &

         ((df_train.StateHoliday == 'a') |

          (df_train.StateHoliday == 'b') | 

          (df_train.StateHoliday == 'c'))].count()[0]))

print ()

print ("-Il est intéréssant de noter, {} jours de fermetures que ne sont pas expliqués (aucune vacances d'annoncées).".format(df_train[(df_train.Open == 0) &

         (df_train.StateHoliday == "0")

         &(df_train.SchoolHoliday == 0)].count()[0]))

print ()
df_train[(df_train.Open == 0)].Sales.mean()
df_train = df_train.drop(df_train[(df_train.Open == 0) & (df_train.Sales == 0)].index)
df_train = df_train.reset_index(drop=True) #making sure the indexes are back to [0,1,2,3 etc.] 
print ("Our new training set has now {} rows ".format(df_train.shape[0]))
df_train.Sales.describe() 
# Pourquoi?

df_train[(df_train.Open == 1) & (df_train.Sales == 0)]
df_train=df_train.drop(df_train[(df_train.Open == 1) & (df_train.Sales == 0)].index)

df_train = df_train.reset_index(drop=True) 
fig, axes = plt.subplots(1, 2, figsize=(17,3.5))

axes[0].boxplot(df_train.Sales, showmeans=True,vert=False)

axes[0].set_xlim(0,max(df_train["Sales"]+1000))

axes[0].set_title('Boxplot For Sales Values')

axes[1].hist(df_train.Sales, cumulative=False, bins=20)

axes[1].set_title("Sales histogram")

axes[1].set_xlim((min(df_train.Sales), max(df_train.Sales)))



{"Mean":np.mean(df_train.Sales),"Median":np.median(df_train.Sales)}

fig, axes = plt.subplots(1, 2, figsize=(17,3.5))

axes[0].boxplot(df_train.Sales, showmeans=True,vert=False)

axes[0].set_xlim(0,max(df_train["Sales"]+1000))

axes[0].set_title('Boxplot For Sales Values')

axes[1].hist(np.log(df_train.Sales), cumulative=False, bins=20)

axes[1].set_title("Sales histogram")

axes[1].set_xlim((min(np.log(df_train.Sales)), max(np.log(df_train.Sales))))



{"Mean":np.mean(df_train.Sales),"Median":np.median(df_train.Sales)}
print ("{0:.2f}% of the time Rossman are actually having big sales day (considered outliers).".format(df_train[df_train.Sales>14000].count()[0]/df_train.shape[0]*100))
df_train.Customers.describe()    
fig, axes = plt.subplots(1, 2, figsize=(17,3.5))

axes[0].boxplot(df_train.Customers, showmeans=True,vert=False)

axes[0].set_xlim(0,max(df_train["Customers"]+100))

axes[0].set_title('Boxplot For Customer Values')

axes[1].hist(df_train.Customers, cumulative=False, bins=20)

axes[1].set_title("Customers histogram")

axes[1].set_xlim((min(df_train.Customers), max(df_train.Customers)))



{"Mean":np.mean(df_train.Customers),"Median":np.median(df_train.Customers)}
print ("{0:.2f}% of the time Rossman are actually having customers more than usual (considered outliers).".format(df_train[df_train.Customers>1500].count()[0]/df_train.shape[0]*100))
df_train[df_train.Customers>7000]
stats.pearsonr(df_train.Customers, df_train.Sales)[0]
df_store.count(0)/df_store.shape[0] * 100
df_store[pd.isnull(df_store.CompetitionDistance)] 
df_store_check_distribution=df_store.drop(df_store[pd.isnull(df_store.CompetitionDistance)].index)

fig, axes = plt.subplots(1, 2, figsize=(17,3.5))

axes[0].boxplot(df_store_check_distribution.CompetitionDistance, showmeans=True,vert=False,)

axes[0].set_xlim(0,max(df_store_check_distribution.CompetitionDistance+1000))

axes[0].set_title('Boxplot For Closest Competition')

axes[1].hist(df_store_check_distribution.CompetitionDistance, cumulative=False, bins=30)

axes[1].set_title("Closest Competition histogram")

axes[1].set_xlim((min(df_store_check_distribution.CompetitionDistance), max(df_store_check_distribution.CompetitionDistance)))

{"Mean":np.nanmean(df_store.CompetitionDistance),"Median":np.nanmedian(df_store.CompetitionDistance),"Standard Dev":np.nanstd(df_store.CompetitionDistance)}#That's what i thought, very different values, let's see why 
df_store['CompetitionDistance'].fillna(df_store['CompetitionDistance'].median(), inplace = True)
# remplir avec la médiane versus 0

df_store.CompetitionOpenSinceMonth.fillna(0, inplace = True)

df_store.CompetitionOpenSinceYear.fillna(0,inplace=True)
df_store[pd.isnull(df_store.Promo2SinceWeek)]
df_store[pd.isnull(df_store.Promo2SinceWeek)& (df_store.Promo2==0)]
df_store.Promo2SinceWeek.fillna(0,inplace=True)

df_store.Promo2SinceYear.fillna(0,inplace=True)

df_store.PromoInterval.fillna(0,inplace=True)
df_store.count(0)/df_store.shape[0] * 100
#Left-join the train to the store dataset since .Why?

#Because you want to make sure you have all events even if some of them don't have their store information ( which shouldn't happen)

df_train_store = pd.merge(df_train, df_store, how = 'left', on = 'Store')

df_train_store.head() 

print ("The Train_Store dataset has {} Rows and {} Variables".format(str(df_train_store.shape[0]),str(df_train_store.shape[1]))) 

df_train_store
df_train_store['SalesperCustomer']= df_train_store['Sales']/df_train_store['Customers']
df_train_store['SalesperCustomer']
fig, axes = plt.subplots(2, 3,figsize=(17,10) )

palette = itertools.cycle(sns.color_palette(n_colors=4))

plt.subplots_adjust(hspace = 0.28)

#axes[1].df_train_store.groupby(by="StoreType").count().Store.plot(kind='bar')

axes[0,0].bar(df_store.groupby(by="StoreType").count().Store.index,df_store.groupby(by="StoreType").count().Store,color=[next(palette),next(palette),next(palette),next(palette)])

axes[0,0].set_title("Number of Stores per Store Type \n Fig 1.1")

axes[0,1].bar(df_train_store.groupby(by="StoreType").sum().Sales.index,df_train_store.groupby(by="StoreType").sum().Sales/1e9,color=[next(palette),next(palette),next(palette),next(palette)])

axes[0,1].set_title("Total Sales per Store Type (in Billions) \n Fig 1.2")

axes[0,2].bar(df_train_store.groupby(by="StoreType").sum().Customers.index,df_train_store.groupby(by="StoreType").sum().Customers/1e6,color=[next(palette),next(palette),next(palette),next(palette)])

axes[0,2].set_title("Total Number of Customers per Store Type (in Millions) \n Fig 1.3")

axes[1,0].bar(df_train_store.groupby(by="StoreType").sum().Customers.index,df_train_store.groupby(by="StoreType").Sales.mean(),color=[next(palette),next(palette),next(palette),next(palette)])

axes[1,0].set_title("Average Sales per Store Type \n Fig 1.4")

axes[1,1].bar(df_train_store.groupby(by="StoreType").sum().Customers.index,df_train_store.groupby(by="StoreType").Customers.mean(),color=[next(palette),next(palette),next(palette),next(palette)])

axes[1,1].set_title("Average Number of Customers per Store Type \n Fig 1.5")

axes[1,2].bar(df_train_store.groupby(by="StoreType").sum().Sales.index,df_train_store.groupby(by="StoreType").SalesperCustomer.mean(),color=[next(palette),next(palette),next(palette),next(palette)])

axes[1,2].set_title("Average Spending per Customer in each Store Type \n Fig 1.6")

plt.show()
StoretypeXAssortment = sns.countplot(x="StoreType",hue="Assortment",order=["a","b","c","d"], data=df_store,palette=sns.color_palette("Set2", n_colors=3)).set_title("Number of Different Assortments per Store Type")

df_store.groupby(by=["StoreType","Assortment"]).Assortment.count()
df_train_store['Month']=df_train_store.Date.dt.month

df_train_store['Year']=df_train_store.Date.dt.year
sns.factorplot(data = df_train_store, x ="Month", y = "Sales", 

               col = 'Promo', # per store type in cols

               hue = 'Promo2',

               row = "Year"

              ,sharex=False)

sns.factorplot(data = df_train_store, x ="Month", y = "SalesperCustomer", 

               col = 'Promo', # per store type in cols

               hue = 'Promo2',

               row = "Year"

              ,sharex=False)
#df_train_store.columns
#sns.factorplot(data = df_train_store, x ="Month", y = "Customers", 

#               col = 'Promo', # per store type in cols

#               hue = 'Promo2',

#               row = "Year"

#              ,sharex=False)
sns.factorplot(data = df_train_store, x ="DayOfWeek", y = "Sales",

                hue='Promo'

              ,sharex=False)
#33 Stores are opened on Sundays

print ("Number of Stores opened on Sundays:{}" .format(df_train_store[(df_train_store.Open == 1) & (df_train_store.DayOfWeek == 7)]['Store'].unique().shape[0]))
len(df_train_store["Store"].unique())
#df_train_store[(df_train_store.Open == 1) & (df_train_store.DayOfWeek == 7)]
df_train_store['CompetitionDist_Cat'] = pd.cut(df_train_store['CompetitionDistance'], 5)
#pd.cut(df_train_store['CompetitionDistance'], 5)
df_train_store.groupby(by="CompetitionDist_Cat").Sales.mean()
df_train_store.groupby(by="CompetitionDist_Cat").Customers.mean()
del df_train_store["CompetitionDist_Cat"]
# Création de variable jour

df_train_store['Day']=df_train_store.Date.dt.day

del df_train_store["Date"]
df_train_store['StoreType'].isnull().any(),df_train_store['Assortment'].isnull().any(),df_train_store['StateHoliday'].isnull().any()
df_train_store["StoreType"].value_counts(),df_train_store["Assortment"].value_counts(),df_train_store["StateHoliday"].value_counts()
df_train_store['StateHoliday'] = df_train_store['StateHoliday'].astype('category')

df_train_store['Assortment'] = df_train_store['Assortment'].astype('category')

df_train_store['StoreType'] = df_train_store['StoreType'].astype('category')

df_train_store['PromoInterval']= df_train_store['PromoInterval'].astype('category')
df_train_store['StateHoliday_cat'] = df_train_store['StateHoliday'].cat.codes

df_train_store['Assortment_cat'] = df_train_store['Assortment'].cat.codes

df_train_store['StoreType_cat'] = df_train_store['StoreType'].cat.codes

df_train_store['PromoInterval_cat'] = df_train_store['PromoInterval'].cat.codes
df_train_store['StateHoliday_cat'] = df_train_store['StateHoliday_cat'].astype('float')

df_train_store['Assortment_cat'] = df_train_store['Assortment_cat'].astype('float')

df_train_store['StoreType_cat'] = df_train_store['StoreType_cat'].astype('float')

df_train_store['PromoInterval_cat'] = df_train_store['PromoInterval_cat'].astype('float')
df_train_store.dtypes
df_correlation=df_train_store[['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'Promo',

        'SchoolHoliday',

       'CompetitionDistance', 'CompetitionOpenSinceMonth',

       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',

       'Promo2SinceYear', 'SalesperCustomer', 'Month', 'Year',

       'Day', 'StateHoliday_cat', 'Assortment_cat', 'StoreType_cat',

       'PromoInterval_cat']]
df_correlation=df_correlation.drop('Open', axis = 1)
upper_triangle = np.zeros_like(df_correlation.corr(), dtype = np.bool)

upper_triangle[np.triu_indices_from(upper_triangle)] = True #make sure we don't show half of the other triangle

f, ax = plt.subplots(figsize = (15, 10))

sns.heatmap(df_correlation.corr(),ax=ax,mask=upper_triangle,annot=True, fmt='.2f',linewidths=0.5,cmap=sns.diverging_palette(10, 133, as_cmap=True))
del df_train_store['Assortment_cat']

del df_train_store['StoreType_cat']

del df_train_store['PromoInterval_cat']
df_train_store['CompetitionOpenSince'] = np.where((df_train_store['CompetitionOpenSinceMonth']==0) & (df_train_store['CompetitionOpenSinceYear']==0) , 0,(df_train_store.Month - df_train_store.CompetitionOpenSinceMonth) + 

                                       (12 * (df_train_store.Year - df_train_store.CompetitionOpenSinceYear)) )
df_train_store['CompetitionOpenSince']
del df_train_store['CompetitionOpenSinceYear']

del df_train_store['CompetitionOpenSinceMonth']

#df_train_store["is_holiday_state"] = df_train_store['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})

#del df_train_store['StateHoliday_cat']

#del df_train_store['StateHoliday']
df_train_store=pd.get_dummies(df_train_store, columns=["Assortment", "StoreType","PromoInterval"], prefix=["is_Assortment", "is_StoreType","is_PromoInteval"])

df_train_store.columns
df_train_store
def rmspe(y, yhat):

    rmspe = np.sqrt(np.mean( (y - yhat)**2 ))

    return rmspe
features = df_train_store.drop(['Customers', 'Sales', 'SalesperCustomer'], axis = 1) 

# Une règle empirique est de transformer ma valeur cible en log si je vois que les valeurs sont très dispersées, ce qui est le cas

# et bien sûr les ramener à leurs vraies valeurs avec np.exp

targets=np.log(df_train_store.Sales)

X_train, X_train_test, y_train, y_train_test = model_selection.train_test_split(features, targets, test_size=0.20, random_state=15)
def function_class(data_train, data_test, clf):

    clf.fit(data_train, y_train)

    yhat  = clf.predict(data_test)

    error = rmspe(y_train_test,yhat)  

    return clf, error
from sklearn.dummy import DummyRegressor

dummmy = DummyRegressor(strategy='median')



clf_dummy, error_dummy = function_class(X_train, X_train_test, dummmy)

error_dummy
rfr = RandomForestRegressor(n_estimators = 10, n_jobs = -1)

clf_rf, error_rf = function_class(X_train, X_train_test, rfr)

error_rf
#X_train
from sklearn.feature_selection import RFECV



# Serialization

from sklearn.externals import joblib 



def neg_rmspe(y, yhat):

    rmspe = np.sqrt(np.mean( (y - yhat)**2 ))*-1

    return rmspe



scoring_new = metrics.make_scorer(neg_rmspe)



def rfecv_step(step):

    rfecv = RFECV(estimator =  RandomForestRegressor(n_estimators = 5, n_jobs = -1), step = step, cv = 2,

                  scoring = scoring_new, n_jobs = -1, verbose = 10)

    x_select = rfecv.fit_transform(X_train, y_train)

    joblib.dump(rfecv, 'selection_features_' + str(step) + '_.pkl')

        

    print("Optimal number of features : %d" % rfecv.n_features_)

    print("Rmpse with the features selected: %f" % rfecv.grid_scores_.max())

    

    return x_select



x_select = rfecv_step(25)
rfecv_1 = joblib.load('selection_features_' + str(25) + '_.pkl')



rfr = RandomForestRegressor(n_estimators = 10, n_jobs = -1)

clf_rf, error_rf = function_class(rfecv_1.transform(X_train), rfecv_1.transform(X_train_test), rfr)

error_rf
params = {'max_depth':(10,20),

         'n_estimators':(10,25)}

scoring_fnc = metrics.make_scorer(rmspe)

grid = model_selection.RandomizedSearchCV(estimator = rfr, param_distributions = params,

                                          cv = 3, verbose = 10, n_jobs = -1, scoring = scoring_new ) 

grid.fit(rfecv_1.transform(X_train), y_train)
grid.best_params_,grid.best_score_

#MY BEST PARAMS ARE :n_estimators=128,max_depth=20,min_samples_split=10
#with the optimal parameters i got let's see how it behaves with the validation set

rfr_val=RandomForestRegressor(n_estimators=20, 

                             criterion='mse', 

                             max_depth=10, 

                             min_samples_split=10, 

                             n_jobs=-1,

                             random_state=35, 

                             verbose=0)

model_RF_test = rfr_val.fit(X_train,y_train)

yhat = model_RF_test.predict(X_train_test)
importances = rfr_val.feature_importances_



std = np.std([rfr_val.feature_importances_ for tree in rfr_val.estimators_],

             axis=0)

indices = np.argsort(importances)

palette1 = itertools.cycle(sns.color_palette())

# Store the feature ranking

features_ranked=[]

for f in range(X_train.shape[1]):

    features_ranked.append(X_train.columns[indices[f]])

# Plot the feature importances of the forest



plt.figure(figsize=(10,15))

plt.title("Feature importances")

plt.barh(range(X_train.shape[1]), importances[indices],

            color=[next(palette1)], align="center")

plt.yticks(range(X_train.shape[1]), features_ranked)

plt.ylabel('Features')

plt.ylim([-1, X_train.shape[1]])

plt.show()
from xgboost import XGBRegressor



xgb = XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist')

clf_xgb, error_xgb = function_class(X_train, X_train_test, xgb)

error_xgb
params = {'n_estimators':(100,200,300,400,500),

         'colsample_bytree':(0.6,0.7,0.8,0.9,1),

         'learning_rate':(0.1,0.01,1,0.001)}

scoring_fnc = metrics.make_scorer(rmspe)

grid = model_selection.RandomizedSearchCV(estimator = XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist'), param_distributions = params,

                                          cv = 3, verbose = 10, n_jobs = -1, scoring = scoring_new ) 

grid.fit(rfecv_1.transform(X_train), y_train)
grid.best_params_,grid.best_score_
xgb = XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist', colsample_bytree=0.9, learning_rate=1, n_estimators = 400 )

clf_xgb, error_xgb = function_class(X_train, X_train_test, xgb)

error_xgb
from catboost import CatBoostRegressor



ctb = CatBoostRegressor(task_type="GPU", devices='0:1', verbose = 0)

clf_ctb, error_ctb = function_class(X_train, X_train_test, ctb)

error_ctb
params = {'iterations':(500, 750, 1000, 2000),

         'depth':(10,20,25),

         'learning_rate':(0.1,0.01,1,0.001)}

scoring_fnc = metrics.make_scorer(rmspe)

grid = model_selection.RandomizedSearchCV(estimator = CatBoostRegressor(task_type="GPU", devices='0:1', verbose = 0), param_distributions = params,

                                          cv = 3, verbose = 10, n_jobs = -1, scoring = scoring_new ) 

grid.fit(rfecv_1.transform(X_train), y_train)
grid.best_params_,grid.best_score_