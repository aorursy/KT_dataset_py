import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_columns', 500)

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools



from math import sqrt

from numpy import concatenate

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.metrics.pairwise import euclidean_distances

from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV



# multivariate output stacked lstm example

from numpy import array

from numpy import hstack

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
demo = pd.read_csv('../input/train_OwBvO8W/demographics.csv')

print(demo.shape)

demo.head()
event = pd.read_csv('../input/train_OwBvO8W/event_calendar.csv')

print(event.shape)

event['YearMonth']  = pd.to_datetime(event['YearMonth'],format='%Y%m')

event.head()
historical = pd.read_csv('../input/train_OwBvO8W/historical_volume.csv')

print(historical.shape)

historical['YearMonth'] = pd.to_datetime(historical['YearMonth'],format='%Y%m')

historical.head()
soda = pd.read_csv('../input/train_OwBvO8W/industry_soda_sales.csv')

print(soda.shape)

soda['YearMonth'] = pd.to_datetime(soda['YearMonth'],format='%Y%m')

soda.head()
industry = pd.read_csv('../input/train_OwBvO8W/industry_volume.csv')

print(industry.shape)

industry['YearMonth'] = pd.to_datetime(industry['YearMonth'],format='%Y%m')

industry.head()
price = pd.read_csv('../input/train_OwBvO8W/price_sales_promotion.csv')

print(price.shape)

price['YearMonth'] = pd.to_datetime(price['YearMonth'],format='%Y%m')

price.head()
weather = pd.read_csv('../input/train_OwBvO8W/weather.csv')

print(weather.shape)

weather['YearMonth'] = pd.to_datetime(weather['YearMonth'],format='%Y%m')

weather.head()
sku = historical.merge(price,on=['Agency','SKU','YearMonth'],how='left')

sku = sku.merge(soda,on=['YearMonth'],how='left')

sku = sku.merge(industry,on='YearMonth',how='left')

sku = sku.merge(event,on=['YearMonth'],how='left')

print(sku.shape)

sku.head()
sku.describe()
agency = weather.merge(demo,on=['Agency'],how='left')

print(agency.shape)

agency.head()
agency.describe()
df = sku.merge(agency,on=['YearMonth','Agency'],how='left')

print(df.shape)

df.head()
df = pd.get_dummies(df, columns= ['SKU'], dummy_na= False)

df.head()
df.describe()
train_df = df.drop(columns=['Price','Sales','Promotions'])

train_df.set_index('YearMonth',inplace=True)

train_df.head()
test = pd.read_csv('../input/test_8uviCCm/volume_forecast.csv')

print(test.shape)

test.head()
n_date = len(train_df.index.unique())

tes4 = pd.date_range(start='1/1/2013', end='31/12/2017',freq='M')

tes3 = list(tes4)*len(test)

tes1 = list(test.Agency)*len(tes4)

tes2 = list(test.SKU)*len(tes4)
test_df = pd.DataFrame({'Agency':tes1,'SKU':tes2,'Volume':np.nan})

test_df.sort_values(['Agency','SKU'],inplace=True,ascending=False)

test_df.reset_index(inplace=True,drop=True)

test_df.loc[:,'YearMonth'] = tes3

test_df['YearMonth'] = test_df['YearMonth'].dt.floor('d') - pd.offsets.MonthBegin(1)

print(test_df.shape)

test_df.head()
test_df = test_df.merge(weather,on=['YearMonth','Agency'],how='left')

test_df = test_df.merge(demo,on='Agency',how='left')

test_df = test_df.merge(industry,on='YearMonth',how='left')

test_df = test_df.merge(soda,on=['YearMonth'],how='left')

test_df = test_df.merge(event,on=['YearMonth'],how='left')

test_df = pd.get_dummies(test_df, columns= ['SKU'], dummy_na= False)

test_df.set_index('YearMonth',inplace=True)

test_df = test_df[train_df.columns]

print(test_df.shape)

test_df.head()
tes = ['Agency_06','Agency_14']*len(price.SKU.unique())*len(tes4)

tes.sort()

tes5 = list(price.SKU.unique())*2*len(tes4)

df_agen = pd.DataFrame({'Agency':tes,'SKU':tes5,'YearMonth':np.NaN,'Volume':np.NaN})

df_agen.sort_values(['Agency','SKU'],inplace=True)

df_agen.loc[:,'YearMonth'] = list(tes4)*2*25

df_agen.loc[:,'YearMonth'] = df_agen.loc[:,'YearMonth'].dt.floor('d') - pd.offsets.MonthBegin(1)

print(df_agen.shape)

df_agen.head()
df_agen = df_agen.merge(weather,on=['YearMonth','Agency'],how='left')

df_agen = df_agen.merge(demo,on='Agency',how='left')

df_agen = df_agen.merge(industry,on='YearMonth',how='left')

df_agen = df_agen.merge(soda,on=['YearMonth'],how='left')

df_agen = df_agen.merge(event,on=['YearMonth'],how='left')

df_agen = pd.get_dummies(df_agen, columns= ['SKU'], dummy_na= False)

df_agen.set_index('YearMonth',inplace=True)

df_agen = df_agen[train_df.columns]

print(df_agen.shape)

df_agen.head()
# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
# Missing values statistics

missing_values = missing_values_table(train_df)
# Missing values statistics

missing_values = missing_values_table(test_df)

missing_values.head()
corr = train_df[train_df.columns[:18]].corr()

plt.figure(figsize=(12,12))

sns.heatmap(corr,vmin=-1,cmap='coolwarm', annot=True, fmt = ".2f")
train_df.drop(columns=['FIFA U-17 World Cup','Football Gold Cup'],inplace=True)

test_df.drop(columns=['FIFA U-17 World Cup','Football Gold Cup'],inplace=True)

df_agen.drop(columns=['FIFA U-17 World Cup','Football Gold Cup'],inplace=True)
x_call = train_df.columns[2:]

X = train_df[x_call]

y = train_df['Volume']
std_call = ['Soda_Volume','Industry_Volume','Avg_Max_Temp','Avg_Population_2017','Avg_Yearly_Household_Income_2017']

scaller = StandardScaler()

std = pd.DataFrame(scaller.fit_transform(X[std_call]),columns=std_call)

std_test = pd.DataFrame(scaller.transform(test_df[std_call]),columns=std_call)

std_agen = pd.DataFrame(scaller.transform(df_agen[std_call]),columns=std_call)
X_std = X.copy()

X_std.loc[:,std_call] = std.values

test_df_std = test_df.copy() 

df_agen_std = df_agen.copy()

test_df_std.loc[:,std_call] = std_test.values 

df_agen_std.loc[:,std_call] = std_agen.values
print(X_std.shape)

X_std.head()
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.30, random_state = 217,shuffle=True)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# Spot all methods want to be used

models = []

models.append(('RF', RandomForestRegressor()))

models.append(('LR', LinearRegression()))

models.append(('GB', GradientBoostingRegressor()))

models.append(('LG', LGBMRegressor()))

models.append(('KN', KNeighborsRegressor()))

models.append(('XG', XGBRegressor(objective='reg:squarederror')))
results = pd.DataFrame({'Score':['fit_time', 'score_time', 'test_R_Square', 'test_MSE', 'test_MAE']})

for name, model in models:

    # Spot all scorers want to be used

    scorer = {'R_Square' : 'r2',

              'MSE'  : 'neg_mean_squared_error',

              'MAE' : 'neg_mean_absolute_error'}

        

    # Cross Validation Model

    kfold = KFold(n_splits=5, random_state=217,shuffle=True)

    cv_results = cross_validate(model,X_train, y_train,cv=kfold,scoring=scorer)

    cv_results['test_R_Square'] = cv_results['test_R_Square']*100

    cv_results['test_MSE'] = np.log(np.sqrt(np.abs(cv_results['test_MSE'])))*10

    cv_results['test_MAE'] = np.log(np.abs(cv_results['test_MAE']))*10

    results[name] = pd.DataFrame(cv_results).mean().values
results
model_name = ['RandomForest', 'LinearRegression', 'GradientBoosting', 'KNeighbors', 'LGBM', 'XGBRegressor']



fig = go.Figure()

fig.add_trace(go.Bar(

    x=model_name,

    y=results.iloc[2,1:],

    name='R Square',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=model_name,

    y=results.iloc[3,1:],

    name='logRMSE*10',

    marker_color='lightsalmon'

))

fig.add_trace(go.Bar(

    x=model_name,

    y=results.iloc[4,1:],

    name='logMAE*10',

    marker_color='mediumslateblue'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.layout.update(barmode='group', xaxis_tickangle=-45)

fig.show()
model_rf = RandomForestRegressor()



model_rf.fit(X_train, y_train)



predictions = model_rf.predict(X_test)

print("R Square: %.3f" % r2_score(y_test, predictions))

print("RMSE: %f" % np.sqrt(mean_squared_error(y_test, predictions)))

print("MAE: %f" % mean_absolute_error(y_test, predictions))
model_name  = ['Random Forest']

fig = go.Figure()

fig.add_trace(go.Bar(

    x=model_name,

    y=[r2_score(y_test, predictions)*100],

    name='R Square',

    marker_color='indianred'

))

fig.add_trace(go.Bar(

    x=model_name,

    y=np.log([np.sqrt(mean_squared_error(y_test, predictions))])*10,

    name='logRMSE*10',

    marker_color='lightsalmon'

))

fig.add_trace(go.Bar(

    x=model_name,

    y=np.log([mean_absolute_error(y_test, predictions)])*10,

    name='logMAE*10',

    marker_color='mediumslateblue'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.layout.update(barmode='group', xaxis_tickangle=-45)

fig.show()
model_lg = LGBMRegressor()



model_lg.fit(X_train, y_train)



predictions = model_lg.predict(X_test)

print("R Square: %.3f" % r2_score(y_test, predictions))

print("RMSE: %f" % np.sqrt(mean_squared_error(y_test, predictions)))

print("MAE: %f" % mean_absolute_error(y_test, predictions))
model_fix = RandomForestRegressor()

model_fix.fit(X_std, y)
def plot_feature_importances(df,n):

    # Sort features according to importance

    df = df.sort_values('importance', ascending = False).reset_index()

    

    # Normalize the feature importances to add up to one

    df['importance_normalized'] = df['importance'] / df['importance'].sum()



    # Make a horizontal bar chart of feature importances

    plt.figure(figsize = (12, 12))

    ax = plt.subplot()

    

    # Need to reverse the index to plot most important on top

    ax.barh(list(reversed(list(df.index[:n]))), 

            df['importance_normalized'].head(n), 

            align = 'center', edgecolor = 'k')

    

    # Set the yticks and labels

    ax.set_yticks(list(reversed(list(df.index[:n]))))

    ax.set_yticklabels(df['feature'].head(n))

    

    # Plot labeling

    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')

    plt.show()

    

    return df
# Extract feature importances

feature_importance = model_fix.feature_importances_

feature_importances = pd.DataFrame({'feature': x_call, 'importance': feature_importance})
feature_importance = plot_feature_importances(feature_importances,30)
print(test_df.shape)

test_df.head()
pred_test = model_fix.predict(test_df_std[x_call])

test_df.loc[:,'Volume'] = pred_test

print(test_df.shape)

test_df.head()
def split(x):

    x = x.split('_')

    return x[1]+'_'+x[2]



test_df.loc[:,'SKU'] = test_df[test_df.columns[17:]].idxmax(axis=1).apply(split).values
test_df.drop(columns=test_df.iloc[:,17:-1].columns,inplace=True)

test_df.reset_index(inplace=True)

print(test_df.shape)

test_df.head()
pivot = pd.pivot_table(test_df, values='Volume', index='YearMonth', columns=['Agency','SKU'])

print(pivot.shape)

pivot.head()
# split a multivariate sequence into samples

def split_sequences(sequences, n_steps):

    X, y = list(), list()

    for i in range(len(sequences)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the dataset

        if end_ix > len(sequences)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)
dataset = np.array(pivot)

# choose a number of time steps

n_steps = 12

# convert into input/output

X, y = split_sequences(dataset, n_steps)

# the dataset knows the number of features, e.g. 2

n_features = X.shape[2]
# split into train and validating

train_X, train_y = X[:-12, :], y[:-12,:]

val_X, val_y = X[-12:-3, :], y[-12:-3,:]

test_X, test_y = X[-3:, :], y[-3:,:]
# define model

model = Sequential()

model.add(LSTM(1024, activation='relu', return_sequences=True, input_shape=(n_steps, n_features),recurrent_dropout=0.2))

model.add(LSTM(512, activation='relu',return_sequences=True,recurrent_dropout=0.2))

model.add(LSTM(256, activation='relu',return_sequences=True,recurrent_dropout=0.1))

model.add(LSTM(128, activation='relu',recurrent_dropout=0.1))

model.add(Dense(n_features))

model.compile(optimizer='adam', loss='mse')
# fit model

history = model.fit(train_X, train_y, epochs=712, batch_size=32, verbose=0, shuffle=False,validation_data=(val_X, val_y))
model.summary()
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=list(range(1,722)),

                y=history.history['loss'],

                name="Train",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=list(range(1,722)),

                y=history.history['val_loss'],

                name="Test",

                line_color='dimgray',

                opacity=0.8))



# Use date string to set xaxis range

fig.layout.update(title_text="Forecasting Score")

fig.show()
# demonstrate prediction

train_yhat = model.predict(train_X, verbose=0)

print("RMSE of training data : %.3f" % np.sqrt(mean_squared_error(train_y,train_yhat)))
# demonstrate prediction

val_yhat = model.predict(val_X, verbose=0)

print("RMSE of training data : %.3f" % np.sqrt(mean_squared_error(val_y,val_yhat)))
# demonstrate prediction

test_yhat = model.predict(test_X, verbose=0)

print("RMSE of testing data : %.3f" % np.sqrt(mean_squared_error(test_y,test_yhat)))
# demonstrate prediction

x_input = X[len(X)-1,:]

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)
pivot.loc['2018-01-01',:] = yhat

pivot.index = pd.to_datetime(pivot.index)
vol_for = pd.read_csv('../input/test_8uviCCm/volume_forecast.csv')

print(vol_for.shape)

vol_for.head()
def volume(row):

    agen = row.Agency

    cost = row.SKU

    return pivot[agen,cost]['2018-01-01']



vol_for['Volume'] = vol_for.apply(volume,axis=1)

print(vol_for.shape)
vol_for.head(20)
vol_for.tail(20)
df_agen.head()
pred = model_fix.predict(df_agen_std[x_call])

df_agen.loc[:,'Volume'] = pred

print(df_agen.shape)

df_agen.head()
df_agen.loc[:,'SKU'] = df_agen[df_agen.columns[17:]].idxmax(axis=1).apply(split).values

df_agen.drop(columns=df_agen.iloc[:,17:-1].columns,inplace=True)

df_agen.reset_index(inplace=True)

print(df_agen.shape)

df_agen.head()
pivot_agen = pd.pivot_table(df_agen.reset_index(), values='Volume', index='YearMonth', columns=['Agency','SKU'])

print(pivot_agen.shape)

pivot_agen.head()
dataset = np.array(pivot_agen)

# choose a number of time steps

n_steps = 12

# convert into input/output

X, y = split_sequences(dataset, n_steps)

# the dataset knows the number of features, e.g. 2

n_features = X.shape[2]
# split into train and validating

train_X, train_y = X[:-12, :], y[:-12,:]

val_X, val_y = X[-12:-3, :], y[-12:-3,:]

test_X, test_y = X[-3:, :], y[-3:,:]
# define model

model = Sequential()

# define model

model = Sequential()

model.add(LSTM(1024, activation='relu', return_sequences=True, input_shape=(n_steps, n_features),recurrent_dropout=0.2))

model.add(LSTM(512, activation='relu',return_sequences=True,recurrent_dropout=0.2))

model.add(LSTM(256, activation='relu',return_sequences=True,recurrent_dropout=0.1))

model.add(LSTM(128, activation='relu',recurrent_dropout=0.1))

model.add(Dense(n_features))

model.compile(optimizer='adam', loss='mse')
# fit model

history = model.fit(train_X, train_y, epochs=712, batch_size=32, verbose=0, shuffle=False,validation_data=(val_X, val_y))
model.summary()
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=list(range(1,722)),

                y=history.history['loss'],

                name="Train",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=list(range(1,722)),

                y=history.history['val_loss'],

                name="Test",

                line_color='dimgray',

                opacity=0.8))



# Use date string to set xaxis range

fig.layout.update(title_text="Forecasting Score")

fig.show()
# demonstrate prediction

train_yhat = model.predict(train_X, verbose=0)

print("RMSE of training data : %.3f" % np.sqrt(mean_squared_error(train_y,train_yhat)))
# demonstrate prediction

val_yhat = model.predict(val_X, verbose=0)

print("RMSE of training data : %.3f" % np.sqrt(mean_squared_error(val_y,val_yhat)))
# demonstrate prediction

test_yhat = model.predict(test_X, verbose=0)

print("RMSE of testing data : %.3f" % np.sqrt(mean_squared_error(test_y,test_yhat)))
# demonstrate prediction

x_input = X[len(X)-1,:]

x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=0)
pivot_agen.loc['2018-01-01',:] = yhat

pivot_agen.index = pd.to_datetime(pivot_agen.index)

pivot_agen.tail()
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_06','SKU_01'],

                name="Agency_06 with SKU_01",

                line_color='deepskyblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_06','SKU_02'],

                name="Agency_06 with SKU_02",

                line_color='darkgoldenrod',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_06','SKU_03'],

                name="Agency_06 with SKU_03",

                line_color='dimgray',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_06','SKU_04'],

                name="Agency_06 with SKU_04",

                line_color='aquamarine',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_06','SKU_05'],

                name="Agency_06 with SKU_03",

                line_color='lightpink',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_14','SKU_01'],

                name="Agency_14 with SKU_01",

                line_color='cornflowerblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_14','SKU_02'],

                name="Agency_14 with SKU_02",

                line_color='lawngreen',

                opacity=0.8))

fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_14','SKU_04'],

                name="Agency_14 with SKU_04",

                line_color='lightsalmon',

                opacity=0.8))

fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_14','SKU_05'],

                name="Agency_14 with SKU_05",

                line_color='indianred',

                opacity=0.8))



# Use date string to set xaxis range

fig.layout.update(title_text="Top Four Recommendation SKU for Agency_06 & Agency_14")

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_06','SKU_02'],

                name="Agency_06 with SKU_02",

                line_color='darkgoldenrod',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_06','SKU_01'],

                name="Agency_06 with SKU_01",

                line_color='dimgray',

                opacity=0.8))





fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_14','SKU_01'],

                name="Agency_14 with SKU_01",

                line_color='cornflowerblue',

                opacity=0.8))



fig.add_trace(go.Scatter(

                x=pivot_agen.index,

                y=pivot_agen['Agency_14','SKU_02'],

                name="Agency_14 with SKU_02",

                line_color='lawngreen',

                opacity=0.8))



# Use date string to set xaxis range

fig.layout.update(title_text="Recommendation SKU for Agency_06 & Agency_14")

fig.show()
sku_recom = pd.read_csv('../input/test_8uviCCm/sku_recommendation.csv')

tes = pivot_agen.loc['2018-01-01',:].reset_index()

tes.columns = ['Agency','SKU','Volume']

tes_1 = tes[tes.Agency=='Agency_06']

tes_2 = tes[tes.Agency=='Agency_14']

tes_3 = list(tes_1.loc[tes_1['Volume'].nlargest(2).index,'SKU']) + list(tes_2.loc[tes_2['Volume'].nlargest(2).index,'SKU'])

sku_recom.SKU = tes_3

print(sku_recom)