
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import scipy


def sortByDate(year_data): # Additional function needed for data-loading. Without it we get unsorted values. For example 4-th day gets thrown into second week and so on...
    
    month_data = year_data.loc[year_data['month'] == 1]
    sorted_df = month_data.sort_values(by = 'day')
    
    for month in range(2,13):
        month_data = year_data.loc[year_data['month'] == month]
        sorted_month_data = month_data.sort_values(by = 'day')
        sorted_df = pd.concat([sorted_df, sorted_month_data],)
    
    return sorted_df
def load_weather_data():
    """ 
    Load all weather data files and combine them into a single Pandas DataFrame.
    Add a week column and a hierarchical index (year, month, week, day)
    
    Returns
    --------
    weather_data: data frame containing the weather data
    """
    
    years_to_load = ['2012','2013','2014','2015','2016','2017','2018']
    path=["../input/sickness-and-weather-data/weather_2012.csv","../input/sickness-and-weather-data/weather_2013.csv","../input/sickness-and-weather-data/weather_2014.csv","../input/sickness-and-weather-data/weather_2015.csv","../input/sickness-and-weather-data/weather_2016.csv","../input/sickness-and-weather-data/weather_2017.csv","../input/sickness-and-weather-data/weather_2018.csv"]
    col = list()
    datalist = list()
    week_col = list()
    load_years=list()

    first_year = True
    for filename in path:
        year_data = pd.read_csv(filename)
        year_data = sortByDate(year_data)
        if first_year:
            weather_data = year_data
            first_year = False
        else:
            weather_data = pd.concat([weather_data, year_data], sort = True)

    year,week,day = datetime.date(int(years_to_load[0]),1,1).isocalendar()
    count = 8-day
    rows, cols = weather_data.shape
    first_year=int(years_to_load[0])
    
    
    for i in range(count):
        week_col.append(week)
    if week==52:
        if datetime.date(year,12,28).isocalendar()[1] == 53: # every 4-th year, check it- add it!
            week = 53
        else:
            week = 1
            if year == first_year:
                first_year +=1
    elif week == 53:
        week = 1
        if year == first_year:
            first_year += 1        
        week=1
    else:
        week+=1
        
        
        
    while len(week_col)<rows:
        for i in range(7):
            if len(week_col)<rows:
                week_col.append(week)
        if week==52:
            if datetime.date(int(first_year),12,28).isocalendar()[1] == 53: #53 woche?
                week = 53
            else:
                week = 1
                first_year += 1
        
        elif week == 53:
            week = 1 
            first_year += 1
        
        else:
            week += 1
            
            
            
            
    weather_data.drop("Unnamed: 0",axis=1,inplace=True)
   
    weather_data.insert(loc=0, column='week', value = week_col)
    weather_data.set_index(["year","month","week","day"],inplace=True)
    return weather_data    
data_weather=load_weather_data()
data_weather
def load_influenza_data():
    """ 
    Load and prepare the influenza data file
    
    Returns
    --------
    influenza_data: data frame containing the influenza data
    """
    influenza_data=pd.read_csv('../input/sickness-and-weather-data/influenza.csv')
    influenza_data = pd.DataFrame(influenza_data)
    influenza_data= influenza_data[["Neuerkrankungen pro Woche","Jahr","Kalenderwoche"]]
    new_names = {'Neuerkrankungen pro Woche':'weekly_infections',"Jahr":"year","Kalenderwoche":"week"}
    

    influenza_data.rename(index=str, columns=new_names,inplace=True)
    influenza_data['week'] = influenza_data['week'].str.replace('Woche', '')
    influenza_data['week'] = influenza_data['week'].str.replace('.', '')
    influenza_data['year'] = influenza_data['year'].astype(int)
    influenza_data['week'] = influenza_data['week'].astype(int)
    influenza_data.set_index(["year","week"],inplace=True)
    return influenza_data

data_influenza = load_influenza_data()
data_influenza


data_weather.isnull().any().any()


def handle_missingValues_simple(incomplete_data):
    """ 
    Parameters
    --------
    incomplete_data: data frame containing missing values 
    
    Returns
    --------
    complete_data: data frame not containing any missing values
    """
    #See description
   

    wind=incomplete_data[["wind_mSec"]].interpolate(method='linear')
    temp7h = incomplete_data['temp_7h'].fillna(method='bfill')
    temp14h = incomplete_data['temp_14h'].fillna(method='bfill')
    temp19h = incomplete_data['temp_19h'].fillna(method='bfill')
    hum_missing = incomplete_data[["hum_14h","hum_19h","hum_7h"]]
    hum_missing = hum_missing.reset_index()
    hum_missing = hum_missing.interpolate(method = 'piecewise_polynomial')
    hum_missing.set_index(["year","month","week","day"],inplace = True)
    
    #2012 to 2018_01 are ok (just a bit of NaN)
    wind_degrees_only = incomplete_data[["wind_degrees"]]
    wind_degrees_2012_until_2018_01 = wind_degrees_only.iloc[0:2223]
    wind_degrees_2012_until_2018_01 = wind_degrees_2012_until_2018_01.interpolate(method='linear')
    
    #Tricky part(find the indices before)---description
    wind_degrees_2018_Feb = wind_degrees_only.iloc[2223:2251]     
    wind_degrees_2018_Feb.iloc[:] = wind_degrees_only.iloc[1858:1886].values
    
    wind_degrees_2018_March = wind_degrees_only.iloc[2251:2282] #no NaN
    
    wind_degrees_2018_April = wind_degrees_only.iloc[2282:2312]
    wind_degrees_2018_April.iloc[:] = wind_degrees_only.iloc[1917:1947].values
    
    wind_degrees_2018_May = wind_degrees_only.iloc[2312:2343]
    wind_degrees_2018_May.iloc[:] = wind_degrees_only.iloc[1947:1978].values
    
    wind_degrees_2018_June_July_August = wind_degrees_only.iloc[2343:] #no NaN
    
    skyCover_and_sun = incomplete_data[['skyCover_14h','skyCover_19h','skyCover_7h','sun_hours']].interpolate(method='linear')
    
    
    wind_degrees = pd.concat([wind_degrees_2012_until_2018_01, wind_degrees_2018_Feb, wind_degrees_2018_March,
                              wind_degrees_2018_April,wind_degrees_2018_May, wind_degrees_2018_June_July_August])
   


    complete_data = pd.concat([ incomplete_data[['temp_dailyMax','temp_dailyMean',
                                                'temp_dailyMin','temp_minGround','hum_dailyMean','precip']],hum_missing,wind,temp7h,temp14h,temp19h,skyCover_and_sun], axis = 1, sort = True)

    return complete_data


def handle_missingValues_advanced (incomplete_data):
    """ 
    Parameters
    --------
    data: data frame containing missing values 
    

    Returns
    --------
    data: data frame not containing any missing values
    """
    
    return complete_data
    
data_weather_complete = handle_missingValues_simple(data_weather)
data_weather_complete


data_weather_complete.isnull().any().any()
fig, ax = plt.subplots(4, 2)
    
    
    
ax[0, 0].hist(data_weather_complete.temp_14h, normed=True, bins=30)
ax[1, 0].hist(data_weather_complete.temp_19h, normed=True, bins=30)
ax[0, 1].hist(data_weather_complete.temp_7h, normed=True, bins=30) 
ax[1, 1].hist(data_weather_complete.temp_dailyMax, normed=True, bins=30) 
ax[2, 0].hist(data_weather_complete.temp_dailyMean, normed=True, bins=30)
ax[2, 1].hist(data_weather_complete.temp_dailyMin, normed=True, bins=30)
ax[3, 0].hist(data_weather_complete.temp_minGround, normed=True, bins=30)
# Before excluding certain values in handle_outliers function underneath, we are going to compare two methods and different paramaeters
# All to see which number of outliers seems reasonable, than we are going to exclude entire row that has this outlier
#It will be only a few since we will opt for the most extreme case, where deviation from the mean is really ridiculous.

def out_std(s, nstd=3.0, return_thresholds=False):

    data_mean, data_std = s.mean(), s.std()
    cut_off = data_std * nstd
    lower, upper = data_mean - cut_off, data_mean + cut_off
    if return_thresholds:
        return lower, upper
    else:
        return [False if x < lower or x > upper else True for x in s]
    

    
    
std2 = data_weather_complete.apply(out_std, nstd=1.8)
std3 = data_weather_complete.apply(out_std, nstd=3.0)
std4 = data_weather_complete.apply(out_std, nstd=4.0)

    
    
f, ((ax1, ax2, ax3)) = plt.subplots(ncols=3, nrows=1, figsize=(22, 12));
ax1.set_title('Outliers with 1.8 standard deviations');
ax2.set_title('Outliers using 3 standard deviations');
ax3.set_title('Outliers using 4 standard deviations');

sns.heatmap(std2, cmap='Blues', ax=ax1);
sns.heatmap(std3, cmap='Blues', ax=ax2);
sns.heatmap(std4, cmap='Blues', ax=ax3);


plt.show()

def handle_outliers(noisy_data):
    """ 
    Parameters
    --------
    noisy_data: data frame that contains outliers
    
    Returns
    --------
    cleaned_data: data frame with outliers
    """
    noisy_data=noisy_data[std2]
    noisy_data.loc[noisy_data.temp_14h >= 40, 'temp_14h'] = np.NaN
    
    noisy_data = noisy_data.reset_index()
    noisy_data = noisy_data.interpolate(method = 'piecewise_polynomial')
    noisy_data.set_index(["year","month","week","day"],inplace = True)
    
    cleaned_data = noisy_data
    return cleaned_data
    
data_weather_cleaned = handle_outliers(data_weather_complete)
data_weather_cleaned[["temp_19h"]].plot()
def aggregate_weekly(data):
    """ 
    Parameters
    --------
    data: weather data frame
    
    Returns
    --------
    weekly_stats: data frame that contains statistics aggregated on a weekly basis
    """
    
   
    data=data.reset_index()
    data = data.iloc[1:] #Aggregation with a 2012 1 1 is actually week 52 from 2011 (because of the way they count it) so we should not include it in aggregation
    data.set_index(["year","week"],inplace=True)

    data["temp_weeklyMin"] = data.pivot_table('temp_dailyMin', index=["year",'week'],aggfunc=min)
    data["temp_weeklyMax"] = data.pivot_table('temp_dailyMax', index=["year",'week'],aggfunc=np.mean)
    data["temp_weeklyMean"] = data.pivot_table('temp_dailyMean', index=["year",'week'],aggfunc=np.mean)
    data["temp_7h_weeklyMedian"] = data.pivot_table('temp_7h', index=["year",'week'],aggfunc=np.median)
    data["temp_14h_weeklyMedian"] = data.pivot_table('temp_14h', index=["year",'week'],aggfunc=np.median)
    data["temp_19h_weeklyMedian"] = data.pivot_table('temp_19h', index=["year",'week'],aggfunc=np.median)
    data["hum_weeklyMean"] = data.pivot_table('hum_dailyMean', index=["year",'week'],aggfunc=np.mean)
    data["hum_7h_weeklyMedian"] = data.pivot_table('hum_7h', index=["year",'week'],aggfunc=np.median)
    data["hum_14h_weeklyMedian"] = data.pivot_table('hum_14h', index=["year",'week'],aggfunc=np.median)
    data["hum_19h_weeklyMedian"] = data.pivot_table('hum_19h', index=["year",'week'],aggfunc=np.median)
    data["precip_weeklyMean"] = data.pivot_table('precip', index=["year",'week'],aggfunc=np.mean)
    data["wind_mSec_mean"] = data.pivot_table('wind_mSec', index=["year",'week'],aggfunc=np.mean)
    
    weekly_weather_data = data.drop(["temp_minGround","sun_hours","skyCover_7h","skyCover_19h","skyCover_14h",'temp_dailyMin', 'temp_dailyMax','temp_dailyMean','temp_7h','temp_14h','temp_19h','hum_dailyMean','hum_7h','hum_14h','hum_19h','precip','wind_mSec'], 1)
    ww2012=weekly_weather_data.xs(2012,level='year')
    ww2012=ww2012[~ww2012.index.get_level_values(0).duplicated()]
    ww2013=weekly_weather_data.xs(2013,level='year')
    ww2013=ww2013[~ww2013.index.get_level_values(0).duplicated()]
    ww2014=weekly_weather_data.xs(2014,level='year')
    ww2014=ww2014[~ww2014.index.get_level_values(0).duplicated()]
    ww2015=weekly_weather_data.xs(2015,level='year')
    ww2015=ww2015[~ww2015.index.get_level_values(0).duplicated()]
    ww2016=weekly_weather_data.xs(2016,level='year')
    ww2016=ww2016[~ww2016.index.get_level_values(0).duplicated()]
    ww2017=weekly_weather_data.xs(2017,level='year')
    ww2017=ww2017[~ww2017.index.get_level_values(0).duplicated()]
    ww2018=weekly_weather_data.xs(2018,level='year')
    ww2018=ww2018[~ww2018.index.get_level_values(0).duplicated()]
    weekly_weather_data = pd.concat([ww2012,ww2013,ww2014,ww2015,ww2016,ww2017,ww2018], keys=['2012','2013',"2014","2015","2016","2017","2018"])
    weekly_weather_data.index.names = ['year','week']
    weekly_weather_data.reset_index(inplace=True)
    weekly_weather_data['year'] = pd.to_numeric(weekly_weather_data['year'])
    weekly_weather_data['week'] = pd.to_numeric(weekly_weather_data['week'])
    weekly_weather_data.set_index(["year","week"],inplace=True)
    
    return weekly_weather_data

data_weather_weekly = aggregate_weekly(data_weather_cleaned)
data_weather_weekly

def merge_data(weather_df, influenza_df):
    """ 
    Parameters
    --------
    weather_df: weekly weather data frame
    influenza_df: influenza data frame
    
    Returns
    --------
    merged_data: merged data frame that contains both weekly weather observations and prevalence of influence infections
    """
    merged_data = weather_df.join(influenza_df)
    merged_data=merged_data.reset_index()
    merged_data = merged_data.apply(pd.to_numeric)
    merged_data["weekly_infections"]= merged_data["weekly_infections"].interpolate(method = 'linear')
    
    merged_data.iloc[327:,14] = merged_data.iloc[275:297,14].values
    merged_data.set_index("year","week")
    return merged_data

data_merged = merge_data(data_weather_weekly, data_influenza)
data_merged
data_merged2=data_merged[["temp_weeklyMin","temp_weeklyMax","temp_weeklyMean","temp_7h_weeklyMedian","hum_weeklyMean","hum_7h_weeklyMedian","precip_weeklyMean","wind_mSec_mean","weekly_infections"]].reset_index()

sns_plot2=sns.pairplot(data_merged2)
sns_plot2.savefig("01527395_01.png")
b=data_merged.drop(["weekly_infections"], axis=1).reset_index()
b=pd.DataFrame(b)
melted = pd.melt(b, ['year',"week","index"])
    
melted.drop(['year',"week","index"],axis=1,inplace=True)
melted["value"] = pd.to_numeric(melted["value"])
melted
sns_plot1=sns.boxplot(x="variable", y="value", data=melted)
sns_plot1.set_xticklabels(sns_plot1.get_xticklabels(), rotation = 90, fontsize = 10)
sns_plot1.figure.savefig("01527395_02.png")


data_merged1=data_merged.reset_index()
data_merged1.drop(['year',"week"],axis=1,inplace=True)

# calculate the correlation matrix
corr = data_merged1.corr()

# plot the heatmap
sns_plot2=sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
sns_plot2.figure.savefig("01527395_03.png")

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
import sklearn

data_merged1=data_merged.reset_index()
year=data_merged1["year"]
week=data_merged1["week"]
data_merged1=data_merged1.drop(["year","week"],axis=1)
dm1_col = data_merged1.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data_merged1)
data_merged2=pd.DataFrame(np_scaled,columns =dm1_col)

X=data_merged2.drop(["weekly_infections"], axis=1)
X["year"]=year
X["week"]=week
X_2018=X.loc[X['year'] == 2018]
X=X.loc[X['year'] != 2018]




y=data_merged2["weekly_infections"].reset_index()

# y.drop(["year","week"], axis=1,inplace=True)
y["year"]=year
y["week"]=week
y_2018=y.loc[y['year'] == 2018]
y_2018.drop(["year","week","index"], axis=1,inplace=True)
y=y.loc[y['year'] != 2018]
y.drop(["year","week","index"], axis=1,inplace=True)
X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.4,random_state=42)


from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=123)

def baseline_model():
    model = Sequential()
    model.add(Dense(200, input_dim=17, kernel_initializer='random_uniform', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam())
    return model

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=32, verbose=False)
estimator.fit(X_train, y_train)

prediction = estimator.predict(X_train)
rmse_nn = sqrt(mean_squared_error(y_train, prediction))

prediction1 = estimator.predict(X_test)
rmse_nn1 = sqrt(mean_squared_error(y_test, prediction1))

prediction2 = estimator.predict(X_2018)
rmse_nn2 = sqrt(mean_squared_error(y_2018, prediction2))



print(rmse_nn,rmse_nn1,rmse_nn2)
    
    
    
    
#batch_size = [20, 40, 60, 80, 100]
#epochs = [10, 50, 100,250,500]
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#weight_constraint = [1, 2, 3, 4, 5]
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#neurons = [1, 5, 10, 15, 20, 25, 30]
#param_grid = dict(neurons=neurons, batch_size=batch_size, epochs=epochs, learn_rate=learn_rate, 
#                 momentum=momentum, dropout_rate=dropout_rate, weight_constraint=weight_constraint)
# param_grid = dict(batch_size=batch_size)
# grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1,cv=10)
# grid_result = grid.fit(X_train, y_train)
# summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
from sklearn import *

model2 = ensemble.RandomForestRegressor(n_estimators=25, random_state=11, max_depth=1,
                                        min_weight_fraction_leaf=0.122)
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5


model2.fit(X_train, np.log1p(y_train.values))
print("Model2 trained")
preds2 = model2.predict(X_test)
preds3 = model2.predict(X_2018)
print('RMSE RandomForestRegressor on validation data: ', RMSLE(y_test, preds2))
print('RMSE RandomForestRegressor on test data: ', RMSLE(y_2018, preds3))

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
# parameters for GridSearchCV
# specify parameters and distributions to sample from
param_dist = {"n_estimators": [3,4,5,6,7,8,9,11,15,18,24,25,26,27,28,40,50,80],
              "min_weight_fraction_leaf": [3.03030303e-03,   4.04040404e-03,   5.05050505e-03,
         6.06060606e-03,   7.07070707e-03,   8.08080808e-03,
         9.09090909e-03,   1.01010101e-02,   1.11111111e-02,
         1.21212121e-02,   1.31313131e-02,   1.41414141e-02,
         1.51515152e-02,   1.61616162e-02,   1.71717172e-02,0.0002,0.0003,0.0004,0.0001,0.0009,0.0008],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 30),
              "min_samples_leaf": sp_randint(1, 30),
              "bootstrap": [True, False],
              "max_depth": [3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,40,50,80],
              "random_state": sp_randint(1, 30)
             }
# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(model2, param_distributions=param_dist,
                                   n_iter=n_iter_search,cv=10)
random_search.fit(X_train, np.log1p(y_train.values))
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
report(random_search.cv_results_)

finalmodel = ensemble.RandomForestRegressor(n_estimators=5, random_state=25, max_depth=21,
                                        min_weight_fraction_leaf=0.0131313131,bootstrap=False,max_features=8,min_samples_leaf=1,min_samples_split=12)
def RMSLE(y, pred):
    return metrics.mean_squared_error(y, pred)**0.5


finalmodel.fit(X_train, np.log1p(y_train.values))
print("Model2 trained")
preds2 = finalmodel.predict(X_test)
print('RMSE RandomForestRegressor: ', RMSLE(y_test, preds2))

preds3 = finalmodel.predict(X_2018)
print('RMSE RandomForestRegressor on test data: ', RMSLE(y_2018, preds3))