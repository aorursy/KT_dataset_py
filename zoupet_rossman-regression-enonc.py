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

df_train = pd.read_csv("____",sep=',', parse_dates=['____']

                       , date_parser=____,

                       low_memory = False)





#Additional Information on those stores 

df_store = pd.read_csv("____"

                       , low_memory = False)
df_train.____() 
df_train.dtypes,print ("The Train dataset has {} Rows and {} Variables".format(str(df_train.shape[____]),str(df_train.____[____])))
df_store.____()
df_store.dtypes ,print ("The Store dataset has {} Rows (which means unique Shops) and {} Variables".format(str(df_store.____[____]),str(df_store.____[____]))) 
df_train.____().____()
print ()

print ("-Durant ces 2 années, {} est le nombre de jours que les magasins ont fermés.".format(df_train[(df_train.____ == ____)].count()[0]))

print ()

print ("-Parmi ces fermetures, {} sont dues aux vacances scolaires. " .format(df_train[(df_train.____ == ____) & (df_train.____ == 1)&(df_train.____ == '0') ].count()[0]))

print ()

print ("-et {} sont dues aux vacances 'publique'.".format(df_train[(df_train.____ == 0) &

         ((df_train.StateHoliday == '____') |

          (df_train.StateHoliday == '____') | 

          (df_train.StateHoliday == '____'))].count()[0]))

print ()

print ("-Il est intéréssant de noter, {} jours de fermetures que ne sont pas expliqués (aucune vacances d'annoncées).".format(df_train[(df_train.____ == ____) &

         (df_train.____ == "0")

         &(df_train.____ == 0)].count()[0]))

print ()
df_train[(df_train.____ == ____)].____.mean()
df_train = df_train.drop(df_train[(df_train.____ == ____) & (df_train.Sales == ____)].index)
df_train = df_train.reset_index(drop=True) #making sure the indexes are back to [0,1,2,3 etc.] 
print ("Our new training set has now {} rows ".format(df_train.____[____]))
df_train.Sales.____() 
df_train=df_train.drop(df_train[(df_train.____ == ____) & (df_train.Sales == 0)].index)

df_train = df_train.reset_index(drop=True) 
fig, axes = plt.subplots(1, 2, figsize=(17,3.5))

axes[0].boxplot(df_train.____, showmeans=True,vert=False)

axes[0].set_xlim(0,max(df_train["____"]+1000))

axes[0].set_title('Boxplot For Sales Values')

axes[1].hist(df_train.____, cumulative=False, bins=20)

axes[1].set_title("Sales histogram")

axes[1].set_xlim((min(df_train.____), max(df_train.____)))



{"Mean":np.mean(df_train.____),"Median":np.median(df_train.____)}

print ("{0:.2f}% of the time Rossman are actually having big sales day (considered outliers).".format(df_train[df_train.Sales > ____].count()[0]/df_train.shape[____]*100))
df_train.Customers.____()    
fig, axes = plt.subplots(1, 2, figsize=(17,3.5))

axes[0].boxplot(df_train.____, showmeans=True,vert=False)

axes[0].set_xlim(0,max(df_train["____"]+100))

axes[0].set_title('Boxplot For Customer Values')

axes[1].hist(df_train.____, cumulative=False, bins=20)

axes[1].set_title("Customers histogram")

axes[1].set_xlim((min(df_train.____), max(df_train.____)))



{"Mean":np.mean(df_train.____),"Median":np.median(df_train.____)}
print ("{0:.2f}% of the time Rossman are actually having customers more than usual (considered outliers).".format(df_train[df_train.____ > ____].count()[0]/df_train.shape[____]*100))
df_train[df_train.____ > ____]
stats.pearsonr(df_train.____, df_train.____)[0]
df_store.count(0)/df_store.shape[____] * 100
df_store[pd.isnull(df_store.CompetitionDistance)] 
df_store_check_distribution=df_store.drop(df_store[pd.isnull(df_store.____)].index)

fig, axes = plt.subplots(1, 2, figsize=(17,3.5))

axes[0].boxplot(df_store_check_distribution.____, showmeans=True,vert=False,)

axes[0].set_xlim(0,max(df_store_check_distribution.CompetitionDistance+1000))

axes[0].set_title('Boxplot For Closest Competition')

axes[1].hist(df_store_check_distribution.____, cumulative=False, bins=30)

axes[1].set_title("Closest Competition histogram")

axes[1].set_xlim((min(df_store_check_distribution.____), max(df_store_check_distribution.____)))

{"Mean":np.nanmean(df_store.____),"Median":np.nanmedian(df_store.____),"Standard Dev":np.nanstd(df_store.____)}#That's what i thought, very different values, let's see why 
df_store['CompetitionDistance'].fillna(df_store['____'].median(), inplace = True)
df_store.____.fillna(0, inplace = True)

df_store.____.fillna(0,inplace=True)
df_store[pd.isnull(df_store.Promo2SinceWeek)]
df_store[pd.isnull(df_store.Promo2SinceWeek)& (df_store.Promo2==0)]
df_store.____.fillna(0,inplace=True)

df_store.____.fillna(0,inplace=True)

df_store.____.fillna(0,inplace=True)
df_store.count(0)/df_store.shape[____] * 100
#Left-join the train to the store dataset since .Why?

#Because you want to make sure you have all events even if some of them don't have their store information ( which shouldn't happen)

df_train_store = pd.merge(____, df_store, how = 'left', on = 'Store')

df_train_store.head() 

print ("The Train_Store dataset has {} Rows and {} Variables".format(str(df_train_store.shape[____]),str(df_train_store.shape[____]))) 

df_train_store['SalesperCustomer']= df_train_store['____']/df_train_store['____']
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
StoretypeXAssortment = sns.countplot(x="StoreType",hue="Assortment",order=["____","____","____","d"], data=df_store,palette=sns.color_palette("Set2", n_colors=3)).set_title("Number of Different Assortments per Store Type")

df_store.groupby(by=["StoreType","Assortment"]).Assortment.count()
df_train_store['Month'] = df_train_store.Date.dt.____

df_train_store['Year'] = df_train_store.Date.dt.year
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
sns.factorplot(data = df_train_store, x ="____", y = "Sales",

                hue='Promo'

              ,sharex=False)
#33 Stores are opened on Sundays

print ("Number of Stores opened on Sundays:{}" .format(df_train_store[(df_train_store.Open == ____) & (df_train_store.DayOfWeek == ____)]['Store'].unique().shape[0]))
df_train_store['____']=pd.cut(df_train_store['CompetitionDistance'], ____)
df_train_store.groupby(by="CompetitionDist_Cat").Sales.mean()
df_train_store.groupby(by="____").Customers.mean()
del df_train_store["CompetitionDist_Cat"]
# Création de variable jour

df_train_store['Day'] = df_train_store.Date.dt.____

del df_train_store["Date"]
df_train_store['StoreType'].isnull().any(),df_train_store['Assortment'].isnull().any(),df_train_store['StateHoliday'].isnull().any()
df_train_store["StoreType"].value_counts(),df_train_store["Assortment"].value_counts(),df_train_store["StateHoliday"].value_counts()
df_train_store['StateHoliday'] = df_train_store['____'].astype('category')

df_train_store['Assortment'] = df_train_store['____'].astype('category')

df_train_store['StoreType'] = df_train_store['____'].astype('category')

df_train_store['PromoInterval']= df_train_store['____'].astype('category')
df_train_store['StateHoliday_cat'] = df_train_store['____'].cat.codes

df_train_store['Assortment_cat'] = df_train_store['____'].cat.codes

df_train_store['StoreType_cat'] = df_train_store['____'].cat.codes

df_train_store['PromoInterval_cat'] = df_train_store['____'].cat.codes
df_train_store['StateHoliday_cat'] = df_train_store['____'].astype('float')

df_train_store['Assortment_cat'] = df_train_store['____'].astype('float')

df_train_store['StoreType_cat'] = df_train_store['____'].astype('float')

df_train_store['PromoInterval_cat'] = df_train_store['____'].astype('float')
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

sns.heatmap(df_correlation.____(),ax=ax,mask=upper_triangle,annot=True, fmt='.2f',linewidths=0.5,cmap=sns.diverging_palette(10, 133, as_cmap=True))
del df_train_store['Assortment_cat']

del df_train_store['StoreType_cat']

del df_train_store['PromoInterval_cat']
df_train_store['CompetitionOpenSince'] = np.where((df_train_store['CompetitionOpenSinceMonth']==0) & (df_train_store['CompetitionOpenSinceYear']==0) , 0,(df_train_store.Month - df_train_store.CompetitionOpenSinceMonth) + 

                                       (12 * (df_train_store.Year - df_train_store.CompetitionOpenSinceYear)) )
df_train_store['CompetitionOpenSince']
del df_train_store['____']

del df_train_store['CompetitionOpenSinceMonth']

df_train_store["is_holiday_state"] = df_train_store['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})

del df_train_store['StateHoliday_cat']

del df_train_store['____']
df_train_store=pd.get_dummies(df_train_store, columns=["Assortment", "____","____"], prefix=["is_Assortment", "is_StoreType","is_PromoInteval"])

df_train_store.columns
def rmspe(y, yhat):

    rmspe = np.sqrt(np.mean( (y - yhat)**2 ))

    return rmspe
features = df_train_store.drop(['____', 'Sales', '____'], axis = 1) 

# Une règle empirique est de transformer ma valeur cible en log si je vois que les valeurs sont très dispersées, ce qui est le cas

# et bien sûr les ramener à leurs vraies valeurs avec np.exp

targets=np.log(df_train_store.Sales)

X_train, X_train_test, y_train, y_train_test = model_selection.train_test_split(____, ____, test_size=____, random_state=15)
def function_class(data_train, data_test, ____):

    ____.fit(data_train, y_train)

    yhat  = clf.predict(data_test)

    error = ____(y_train_test,____)  

    return clf, error
from sklearn.____ import ____

dummmy = ____()



clf_dummy, error_dummy = function_class(X_train, X_train_test, dummmy)

error_dummy
rfr = ____(n_estimators = ____, n_jobs = ____)

clf_rf, error_rf = function_class(____, X_train_test, ____)

error_rf
from sklearn.feature_selection import RFECV



# Serialization

from sklearn.externals import joblib 



def neg_rmspe(y, yhat):

    rmspe = np.sqrt(np.mean( (y - yhat)**2 ))*-1

    return rmspe



scoring_new = metrics.make_scorer(neg_rmspe)



def rfecv_step(step):

    rfecv = ____(estimator =  ____(n_estimators = 10, n_jobs = -1), step = step, cv = 2,

                  scoring = scoring_new, n_jobs = -1, verbose = 10)

    x_select = rfecv.fit_transform(X_train, y_train)

    joblib.dump(rfecv, 'selection_features_' + str(step) + '_.pkl')

        

    print("Optimal number of features : %d" % rfecv.n_features_)

    print("Rmpse with the features selected: %f" % rfecv.grid_scores_.max())

    

    return x_select



x_select = ____(1)
# Proposer une alternative à la sélection de variables avec: https://scikit-learn.org/stable/modules/feature_selection.html
rfecv_1 = joblib.load('selection_features_' + str(1) + '_.pkl')



rfr = ____(n_estimators = 10, n_jobs = -1)

clf_rf, error_rf = function_class(____.transform(X_train), ____.transform(X_train_test), rfr)

error_rf
#params = {'max_depth':(10,20),

#         'n_estimators':(10,25)}



#grid = model_selection.RandomizedSearchCV(estimator = ____, param_distributions = ____,

#                                          cv = ____, verbose = 10, n_jobs = ____, scoring = scoring_new ) 

#grid.fit(rfecv_1.transform(X_train), y_train)
grid.best_params_,grid.best_score_

#MY BEST PARAMS ARE :n_estimators=128,max_depth=20,min_samples_split=10
#with the optimal parameters i got let's see how it behaves with the validation set

#rfr_val=RandomForestRegressor(n_estimators=____, 

#                             criterion='____', 

#                             max_depth=____, 

#                             min_samples_split=10, 

#                             n_jobs=4,

#                             random_state=35, 

#                             verbose=0)

#model_RF_test = rfr_val.fit(____,____)

#yhat = model_RF_test.predict(____)
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
from xgboost import ____



xgb = ____(gpu_id = 0, tree_method = 'gpu_hist')

clf_xgb, error_xgb = function_class(____, X_train_test, ____)

error_xgb
params = {'n_estimators':(100,____,____,____,____),

         'colsample_bytree':(0.6,0.7,0.8,0.9,1),

         'learning_rate':(0.1,0.01,1,0.001)}



grid = model_selection.RandomizedSearchCV(estimator = ____(gpu_id = 0, tree_method = 'gpu_hist'), param_distributions = params,

                                          cv = 3, verbose = 10, n_jobs = -1, scoring = scoring_new ) 

grid.fit(rfecv_1.transform(____), ____)
grid.best_params_,grid.best_score_
# Réaliser le même travail pour les modèles linéaires
xgb = XGBRegressor(gpu_id = 0, tree_method = 'gpu_hist', colsample_bytree=____, learning_rate=____, n_estimators = ____ )

clf_xgb, error_xgb = function_class(X_train, X_train_test, ____)

error_xgb
from catboost import ____



ctb = ____(task_type="GPU", devices='0:1', verbose = 0)

clf_ctb, error_ctb = function_class(X_train, X_train_test, ctb)

error_ctb
params = {'iterations':(____, ____, 1000, 2000),

         'depth':(10,20,____),

         'learning_rate':(0.1,0.01,1,0.001)}

scoring_fnc = metrics.make_scorer(rmspe)

grid = model_selection.RandomizedSearchCV(estimator = ____(task_type="GPU", devices='0:1', verbose = 0), param_distributions = params,

                                          cv = 3, verbose = 10, n_jobs = -1, scoring = scoring_new ) 

grid.fit(rfecv_1.transform(X_train), y_train)