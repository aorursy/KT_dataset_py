import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

%matplotlib inline

sns.set(style='whitegrid',color_codes=True)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# sampleSubmission = pd.read_csv("../input/bike-sharing-demand/sampleSubmission.csv")

test = pd.read_csv("../input/bike-sharing-demand/test.csv")

train = pd.read_csv("../input/bike-sharing-demand/train.csv")

df = train.copy()

test_df = test.copy()
# Let's look at first 5 rows of train data

df.head()
df.describe()
# (safety first)

df.isnull().sum()
# BY SEASON

print("Season:")

df.season.value_counts()

sns.factorplot(x='season',data=df,kind='count',size=3,aspect=1)

# by holiday

print("holiday")

print(df.holiday.value_counts())

sns.factorplot(x='holiday',data=df,kind='count',size=3,aspect=1) # majority of data is for non holiday days.

print("working day")

print(df.workingday.value_counts())

sns.factorplot(x='workingday',data=df,kind='count',size=3,aspect=1) # majority of data is for working days.

print("weather (0 is the clearest)")

print(df.weather.value_counts())

sns.factorplot(x='weather',data=df,kind='count',size=3,aspect=1)  

# distribution and outliers of continuous variables

sns.boxplot(data=df[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])

fig=plt.gcf()

fig.set_size_inches(9,9)

# or maybe let's see the histograms

df.temp.unique()

fig,axes=plt.subplots(2,2)

axes[0,0].hist(x="temp",data=df,edgecolor="black",linewidth=2,color='#ff4125')

axes[0,0].set_title("Variation of temp")

axes[0,1].hist(x="atemp",data=df,edgecolor="black",linewidth=2,color='#ff4125')

axes[0,1].set_title("Variation of atemp")

axes[1,0].hist(x="windspeed",data=df,edgecolor="black",linewidth=2,color='#ff4125')

axes[1,0].set_title("Variation of windspeed")

axes[1,1].hist(x="humidity",data=df,edgecolor="black",linewidth=2,color='#ff4125')

axes[1,1].set_title("Variation of humidity")

fig.set_size_inches(10,10)
cor_mat= df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]

df["day"] = [t.day for t in pd.DatetimeIndex(df.datetime)]

df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]

df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]

df['year'] = df['year'].map({2011:0, 2012:1})

df.head()

# same for test

test_df["hour"] = [t.hour for t in pd.DatetimeIndex(test_df.datetime)]

test_df["day"] = [t.day for t in pd.DatetimeIndex(test_df.datetime)]

test_df["month"] = [t.month for t in pd.DatetimeIndex(test_df.datetime)]

test_df['year'] = [t.year for t in pd.DatetimeIndex(test_df.datetime)]

test_df['year'] = test_df['year'].map({2011:0, 2012:1})

cor_mat= df[:].corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
df.drop('datetime',axis=1,inplace=True)

test_df.drop('datetime',axis=1,inplace=True)

df.drop('temp',axis=1,inplace=True)

test_df.drop('temp',axis=1,inplace=True)

# let's not drop day for now, we shall need it for the split

# df.drop('day',axis=1,inplace=True)

# test_df.drop('day',axis=1,inplace=True)

print(df.head())

print(test_df.head())
valid_df = df[df["day"] < 5]

valid_df.head()
train_df = df[df["day"] >= 5]

train_df.head()
x_train = train_df.copy()

x_train.drop('count',axis=1,inplace=True)

x_train.drop("casual",axis=1,inplace=True)

x_train.drop("registered",axis=1,inplace=True)

x_valid = valid_df.copy()

x_valid.drop('count',axis=1,inplace=True)

x_valid.drop("casual",axis=1,inplace=True)

x_valid.drop("registered",axis=1,inplace=True)

y_train = train_df["count"]

y_valid = valid_df["count"]

print(x_train.head())

print(x_valid.head())

print(y_train.head())

print(y_valid.head())
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 



models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]

model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']

rmsle=[]

d={}

for model in range (len(models)):

    clf=models[model]

    clf.fit(x_train,y_train)

    valid_pred=clf.predict(x_valid)

    rmsle.append(np.sqrt(mean_squared_log_error(valid_pred,y_valid)))

d={'Modelling Algo':model_names,'RMSLE':rmsle}   
rmsle_frame=pd.DataFrame(d)

rmsle_frame
registered_train = train_df["registered"]

casual_train = train_df["casual"]



registered_valid = valid_df["registered"]

casual_valid = valid_df["casual"]
# registered

rmsle = []

d = {}

reg_counts = []

cas_counts = []



for model in range (len(models)):

    clf = models[model]

    clf.fit(x_train,registered_train)

    valid_pred = clf.predict(x_valid)

    reg_counts.append(valid_pred)

    rmsle.append(np.sqrt(mean_squared_log_error(valid_pred,registered_valid)))

d={'Modelling Algo':model_names,'RMSLE':rmsle}   

print(pd.DataFrame(d))

# casual

rmsle = []

d = {}

for model in range (len(models)):

    clf = models[model]

    clf.fit(x_train,casual_train)

    valid_pred = clf.predict(x_valid)

    cas_counts.append(valid_pred)

    rmsle.append(np.sqrt(mean_squared_log_error(valid_pred,casual_valid)))

d={'Modelling Algo':model_names,'RMSLE':rmsle}   

print(pd.DataFrame(d))

del d
# let's see total count error now

rmsle = []

d = {}

for model in range(len(models)):

    total_pred = [reg_c+cas_c for reg_c, cas_c in zip(reg_counts[model],cas_counts[model])]

    rmsle.append(np.sqrt(mean_squared_log_error(total_pred,y_valid)))

d={'Modelling Algo':model_names,'RMSLE':rmsle}   

print("TOTAL COUNTS SCORE")

print(pd.DataFrame(d))

del d
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

no_of_test=[200,300,400,500,600,700]

params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}

clf_rf=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_rf.fit(x_train,y_train)

print("RMSE on training set")

pred_tr=clf_rf.predict(x_train)

print((np.sqrt(mean_squared_log_error(pred_tr,y_train))))

print("RMSE on validation set")

pred=clf_rf.predict(x_valid)

print((np.sqrt(mean_squared_log_error(pred,y_valid))))

print(clf_rf.best_params_)

del pred
# registered

no_of_test=[300] # to simplify computations; experiments show it doesn't affect the score too much

params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}

clf_rf_reg=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_rf_reg.fit(x_train,registered_train)

reg_pred=clf_rf_reg.predict(x_valid)

reg_train_pred = clf_rf_reg.predict(x_train)

print("REGISTERED USER COUNT ERROR")

print((np.sqrt(mean_squared_log_error(reg_pred,registered_valid))))

print(clf_rf_reg.best_params_)

# casual

no_of_test=[500]

params_dict={'n_estimators':no_of_test,'n_jobs':[-1],'max_features':["auto",'sqrt','log2']}

clf_rf_cas=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params_dict,scoring='neg_mean_squared_log_error')

clf_rf_cas.fit(x_train,casual_train)

cas_pred=clf_rf_cas.predict(x_valid)

cas_train_pred = clf_rf_cas.predict(x_train)

print("CASUAL USER COUNT ERROR")

print((np.sqrt(mean_squared_log_error(cas_pred,casual_valid))))

print(clf_rf_cas.best_params_)

# total

total_pred = [cas_pred_single+reg_pred_single for reg_pred_single, cas_pred_single in zip(reg_pred, cas_pred)]

total_train_pred = [cas_pred_single+reg_pred_single for reg_pred_single, cas_pred_single in zip(reg_train_pred, cas_train_pred)]

print("TOTAL COUNTS ERROR - validation")

print((np.sqrt(mean_squared_log_error(total_pred,y_valid))))

print("TOTAL COUNTS ERROR - training")

print((np.sqrt(mean_squared_log_error(total_train_pred,y_train))))
from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(hidden_layer_sizes=(80, 80, 60, 60, 40), activation='relu', solver='adam', alpha=0.001, batch_size=40, learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=20)

clf.fit(x_train,y_train)

train_pred=clf.predict(x_train)

train_pred[train_pred < 0] = 0 #making sure there are no negative values

rmsle_tr = np.sqrt(mean_squared_log_error(abs(train_pred),y_train))

valid_pred=clf.predict(x_valid)

valid_pred[valid_pred < 0] = 0 #making sure there are no negative values

rmsle = np.sqrt(mean_squared_log_error(abs(valid_pred),y_valid))

print("RMSLE on validation")

print(rmsle)

print("RMSLE on training")

print(rmsle_tr)

print("nr of iterations")

print(clf.n_iter_)
# registered

# clf_mlp_reg= MLPRegressor(hidden_layer_sizes=(80, 80, 60,40), activation='relu', solver='adam', alpha=0.0001, batch_size=40, learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

clf_mlp_reg = clf

clf_mlp_reg.fit(x_train,registered_train)

reg_pred=clf_mlp_reg.predict(x_valid)

reg_pred[reg_pred < 0] = 0 #making sure there are no negative values



print("REGISTERED USER COUNT ERROR")

print((np.sqrt(mean_squared_log_error(reg_pred,registered_valid))))

# casual

# clf_mlp_cas=MLPRegressor(hidden_layer_sizes=(80, 80, 60,40), activation='relu', solver='adam', alpha=0.0001, batch_size=40, learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

clf_mlp_cas = clf

clf_mlp_cas.fit(x_train,casual_train)

cas_pred=clf_mlp_cas.predict(x_valid)

cas_pred[cas_pred < 0] = 0 #making sure there are no negative values

print("CASUAL USER COUNT ERROR")

print((np.sqrt(mean_squared_log_error(cas_pred,casual_valid))))

# total

total_pred = [cas_pred_single+reg_pred_single for reg_pred_single, cas_pred_single in zip(reg_pred, cas_pred)]

print("TOTAL COUNTS ERROR")

print((np.sqrt(mean_squared_log_error(total_pred,y_valid))))
best_clf = clf_rf



pred=best_clf.predict(test_df)

d={'datetime':test['datetime'],'count':pred}

ans=pd.DataFrame(d)

ans.to_csv('answer_bestsinglemodel.csv',index=False) # saving to a csv file for predictions on kaggle.



best_cas_clf = clf_rf_cas

best_reg_clf = clf_rf_reg



pred_cas=best_cas_clf.predict(test_df)

pred_reg=best_reg_clf.predict(test_df)

pred = [pred_reg_s + pred_cas_s for pred_reg_s, pred_cas_s in zip(pred_reg, pred_cas)]

d={'datetime':test['datetime'],'count':pred}

ans=pd.DataFrame(d)

ans.to_csv('answer_besttwomodels.csv',index=False) # saving to a csv file for predictions on kaggle.
