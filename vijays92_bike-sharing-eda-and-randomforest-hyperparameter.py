#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline 
train=pd.read_csv('../input/bike-sharing-demand/train.csv') ##Loading the training data

test=pd.read_csv("../input/bike-sharing-demand/test.csv") ##Loading the testing data
train.head() 
train.groupby(['workingday']).sum()['count'] 
sns.relplot(x='season',y='count',data=train,hue='workingday') 

sns.relplot(x='weather',y='count',data=train,hue='workingday')
sns.relplot(x='temp',y='count',data=train,hue='workingday')

sns.relplot(x='atemp',y='count',data=train,hue='workingday')
sns.relplot(x='humidity',y='count',data=train,hue='workingday')

sns.relplot(x='windspeed',y='count',data=train,hue='workingday')

sns.relplot(x='casual',y='count',data=train,hue='workingday')
cols = ['temp','atemp','humidity','windspeed','casual','registered']

fig, axes = plt.subplots(2,3,figsize = (10,5))



count=0

for i in range(2):

    for j in range(3):

        x = cols[count+j]

        sns.distplot(train[x].values, ax = axes[i][j],bins = 30)

        axes[i][j].set_title(x,fontsize=15)

        fig.set_size_inches(15,7)

        plt.tight_layout()

    count = count+j+1 

train.info()
train.describe()
print("Shape of training dataset is: ",train.shape)

print("Does the traing dataset have null values ? -",train.isnull().values.any())
visual_df = train.copy()
train['datetime'] = pd.to_datetime(train['datetime'] )#changing the dtype of datetime field to datetime

train['year']=train.datetime.dt.year

train['month']=train.datetime.dt.month

train['day']=train.datetime.dt.day

train['hour']=train.datetime.dt.hour

train['minute']=train.datetime.dt.minute
visual_df['datetime'] = pd.to_datetime(visual_df['datetime'] )#changing the dtype of datetime field to datetime
# method for creating the count plot based on hour for a given year 

def plot_by_month(data,aggre,title):

    d2 = data

    d2['year'] = d2.datetime.dt.year

    d2['month'] = d2.datetime.dt.month

    d2['hour'] = d2.datetime.dt.hour

    

    by_year = d2.groupby([aggre,'year'])['count'].sum().unstack() # groupby hour and working day

    

    return by_year.plot(kind='bar', figsize=(15,5), width=0.9, title=title) # returning the figure grouped by hour



plot_by_month(visual_df,'month', "Seasonal trend: There must be high demand during summer season, when temperature is good enough to ride cycle and low demand during winter.")  

plot_by_month(visual_df,'hour', "Hourly trend: There must be high demand during office timings. Early morning and late evening can have moderate trend (cyclist) and low demand during 10:00 pm to 4:00 am.") 
# method for creating the count plot based on hour for a given year 

def plot_by_hour(data, year):

    d1 = data

    d1['hour'] = d1.datetime.dt.hour

    

    by_hour = d1.groupby(['hour', 'workingday'])['count'].sum().unstack() # groupby hour and working day

    

    return by_hour.plot(kind='bar', figsize=(15,5), width=0.9, title="Hourly pattern based on working days. High trend during hours when peope start to office and leave to home.Year = {0}".format(year)) # returning the figure grouped by hour





plot_by_hour(visual_df, year=2011) # plotting the count plot based on hour for 2011 

plot_by_hour(visual_df, year=2012) # plotting the count plot based on hour for 2012
def plot_hours(data, message):

    d2 = data.copy()

    d2['hour'] = data.datetime.dt.hour # extratcing the hour

    

    hours = {}



    for hour in range(24):

        hours[hour] = d2[ d2.hour == hour ]['count'].values



    

    plt.figure(figsize=(20,10))

    plt.ylabel("Count rent")

    plt.xlabel("Hours")

    plt.title("count vs hours\n" + message)

    plt.boxplot( [hours[hour] for hour in range(24)] )

    plt.grid()

    



plot_hours( visual_df[visual_df.datetime.dt.year == 2011], 'year 2011') # box plot for hourly count for the mentioned year

plot_hours( visual_df[visual_df.datetime.dt.year == 2012], 'year 2012') # box plot for hourly count for the mentioned year

 


plot_hours( visual_df[visual_df.workingday == 0], 'Non-working days') # box plot for hourly count for the mentioned year

plot_hours( visual_df[visual_df.workingday == 1], 'Working days') # box plot for hourly count for the mentioned year
train.columns
from sklearn.linear_model import Lasso,Ridge

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor
train.head()
train['count'] = np.log(train['count']+1)
y_train = train['count'] ## Capture the dependent feature

x_train = train.drop(['datetime','count'],axis=1) ## Capture the independent feature
x_train.head()
test.head(2)
x_train1 = x_train.drop(['casual','registered'],axis=1) # Removing casual and registered as its not available in test data
x_train_pred,x_test_pred,y_train_pred,y_test_pred = train_test_split(x_train1,y_train, test_size=0.3, random_state=42)
x_train_pred.head(2)
models=[RandomForestRegressor(),Lasso(alpha=0.01),DecisionTreeRegressor(),SVR(),KNeighborsRegressor()]

model_names=['RandomForestRegressor','Lasso','DecisionTreeRegressor','SVR','KNeighborsRegressor']

rmse=[]

r_squared=[]

dic={}

for model in range (len(models)):

    alg=models[model]

    alg.fit(x_train_pred,y_train_pred)

    alg_y_pred=alg.predict(x_test_pred)

    rmse.append(np.sqrt(mean_squared_error(y_test_pred,alg_y_pred)))

    r_squared.append(r2_score(y_test_pred,alg_y_pred))

dic={'Modelling Algorithms':model_names,'RMSE':rmse,'R-Squared':r_squared}   

model_performances= pd.DataFrame(dic)



model_performances
plt.figure(figsize = (10,5))

sns.barplot(x='Modelling Algorithms',y='RMSE',data=model_performances)

plt.title("Algorithms vs RMSE")
plt.figure(figsize = (10,5))

sns.barplot(x='Modelling Algorithms',y='R-Squared',data=model_performances)

plt.title("Algorithms vs R-Squared")
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]

# Minimum number of samples required to split a node

min_samples_split = [5, 10]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split}
random_grid
rF_random = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1, scoring='neg_mean_squared_error')

# Fit the random search model

rF_random.fit(x_train_pred,y_train_pred)
rF_random.best_params_
# Performance Comparison

best_fit= rF_random.best_estimator_.predict(x_test_pred)

print("RMSE and R-Squared after hyperparameter tuning :\n")

print("Modelling algorithm: RandomForestRegressor ")

print("RMSE value is: ",np.sqrt(mean_squared_error(y_test_pred,best_fit)))

print("R-Squared value is ",r2_score(y_test_pred,best_fit))
test.head(2)
test['datetime'] = pd.to_datetime(test['datetime'])

test['year']=test.datetime.dt.year

test['month']=test.datetime.dt.month

test['day']=test.datetime.dt.day

test['hour']=test.datetime.dt.hour

test['minute']=test.datetime.dt.minute
test_val = test.drop(['datetime'],axis=1)
test_val.head(2)
predictions = rF_random.best_estimator_.predict(test_val)
predictions_exp = np.exp(predictions)-1
predictions_exp
submission = pd.DataFrame({'datetime':test['datetime'],'count': predictions_exp})
submission.head()
submission_viz=submission.copy()
submission_viz['datetime'] = pd.to_datetime(submission_viz['datetime'])
def plot_by_month_pred(data,aggre,title):

    d2 = data

    d2['year'] = d2.datetime.dt.year

    d2['month'] = d2.datetime.dt.month

    d2['hour'] = d2.datetime.dt.hour

    

    by_year = d2.groupby([aggre,'year'])['count'].sum().unstack() # groupby hour and working day

    

    return by_year.plot(kind='bar', figsize=(15,5), width=0.9, title=title) # returning the figure grouped by hour





plot_by_month_pred(submission_viz,'month', "Testing Data - Seasonal trend: There must be high demand during summer season,\n when temperature is good enough to ride cycle and low demand during winter.")  

plot_by_month_pred(submission_viz,'hour', "Testing Data - Hourly trend: There must be high demand during office timings.\n Early morning and late evening can have moderate trend (cyclist) and low demand during 10:00 pm to 4:00 am.") 
submission.to_csv("sampleSubmission.csv",index=False)