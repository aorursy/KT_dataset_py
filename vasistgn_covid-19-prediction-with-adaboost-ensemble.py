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
#importing requiste libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras import Sequential

from tensorflow.python.keras.layers import LSTM

from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor

import matplotlib.pyplot as plt
class dataImport():

    

    def __init__(self,filename,type, df=None):

        self.df = df

    

    def import_and_describe_data(self,filename,type):

      if type=='csv':

        self.df=pd.read_csv(filename)

        print('------------------------------------DF Head-----------------------------------------')

        print(self.df.head())

        print('-------------------------------------Shape------------------------------------------')

        print(self.df.shape)

        print('--------------------------------------Info------------------------------------------')

        print(self.df.info())

        print('------------------------------DataSet Description-----------------------------------')

        print(self.df.describe())

        return self

    

    #Drop columns with single value.

    def drop_single_value_columns(self):

        drop_cols = list(filter(lambda x : len(self.df[x].unique()) < 2, self.df.columns))        

        print('------------------------------Single Value coulmns----------------------------------')

        print('Columns dropped: ',drop_cols)

        self.df.drop(drop_cols,axis=1, inplace=True)   

        return self



train=(dataImport(f'/kaggle/input/covid19-global-forecasting-week-2/train.csv','csv')

.import_and_describe_data(f'/kaggle/input/covid19-global-forecasting-week-2/train.csv','csv')

.drop_single_value_columns()).df
test=(dataImport(f'/kaggle/input/covid19-global-forecasting-week-2/test.csv','csv')

.import_and_describe_data(f'/kaggle/input/covid19-global-forecasting-week-2/test.csv','csv')

.drop_single_value_columns()).df
temperature_date=(dataImport(f'/kaggle/input/covid19-global-weather-data/temperature_dataframe.csv','csv')

.import_and_describe_data(f'/kaggle/input/covid19-global-weather-data/temperature_dataframe.csv','csv')

.drop_single_value_columns()).df
GDP_Density_data=(dataImport(f'/kaggle/input/countries-of-the-world/countries of the world.csv','csv')

.import_and_describe_data(f'/kaggle/input/countries-of-the-world/countries of the world.csv','csv')

.drop_single_value_columns()).df
#removing spaces in string columns and coverting all textual data to lower case

train=train.apply(lambda x: x.astype(str).str.lower().replace(' ','', regex=True))

test=test.apply(lambda x: x.astype(str).str.lower().replace(' ','', regex=True))

temperature_date=temperature_date.apply(lambda x: x.astype(str).str.lower().replace(' ','', regex=True))

GDP_Density_data=GDP_Density_data.apply(lambda x: x.astype(str).str.lower().replace(' ','', regex=True))

#Province is more or less have Nan values replacing with country's value

train['Province_State']=np.where(train.Province_State=='nan', train.Country_Region, train.Province_State)

test['Province_State']=np.where(test.Province_State=='nan', test.Country_Region, test.Province_State)

temperature_date['province']=np.where(temperature_date.province=='nan', temperature_date.country, temperature_date.province)
#for 177 rows temparature is null replacing with 0

temperature_date=temperature_date.fillna(0)

temperature_date.info()
#3-4 countries/rows have none values will be replaced with 0's.

# Will go for machine learning driven imputation for now will proceed with replacing 0's

GDP_Density_data=GDP_Density_data.fillna(0)

GDP_Density_data.info()
# there are few countries which are represented differently in temparature and covid data set changing them in temparature data set

temperature_date['country']=temperature_date.country.replace("usa","us")

temperature_date['country']=temperature_date.country.replace("uk","unitedkingdom")

temperature_date['country']=temperature_date.country.replace("taiwan","taiwan*")

temperature_date['country']=temperature_date.country.replace("korea","korea,south")

temperature_date['country']=temperature_date.country.replace("uae","unitedarabemirates")
#changing temparature field to float 

temperature_date.tempC=temperature_date.tempC.astype(float)

#taking mean of temparature for every country

temperature_date=temperature_date.groupby(temperature_date['country'])['tempC'].mean().reset_index()
#mereging temperature data set with train/test

train = pd.merge(train,temperature_date[['tempC','country']], how='left',  left_on=['Country_Region'], right_on=['country'])

test = pd.merge(test,temperature_date[['tempC','country']], how='left',  left_on=['Country_Region'], right_on=['country'])

train.head()
#few african countries don't have tempartures and simple google search inidcates temp around 20

# as said earlier will replace with ML imputation technique after looking how this metric influences the model

train.tempC.fillna(20,inplace=True)

test.tempC.fillna(20,inplace=True)

train.info()
#correcting country names in Density data set

GDP_Density_data['Country']=GDP_Density_data.Country.replace("unitedstates","us")

GDP_Density_data['Country']=GDP_Density_data.Country.replace("taiwan","taiwan*")

GDP_Density_data['Country']=GDP_Density_data.Country.replace("korea","korea,south")

GDP_Density_data['Country']=GDP_Density_data.Country.replace("uae","unitedarabemirates")
train = pd.merge(train,GDP_Density_data, how='left',  left_on=['Country_Region'], right_on=['Country'])

test = pd.merge(test,GDP_Density_data, how='left',  left_on=['Country_Region'], right_on=['Country'])

train.info()
#imputing 0's where data is not there

train.fillna(0,inplace=True)

test.fillna(0,inplace=True)

train.head()
#dropping unwanted columns

train=train.drop(['country','Country','Region'], axis = 1)

test=test.drop(['country','Country','Region'], axis = 1)

test.head()
#identifying numerical data and converting to float values

cols=['ConfirmedCases','Fatalities','Population','Area (sq. mi.)','Pop. Density (per sq. mi.)'

      ,'Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)',

     'GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Climate','Birthrate','Deathrate']

cols_test=['Population','Area (sq. mi.)','Pop. Density (per sq. mi.)'

      ,'Coastline (coast/area ratio)','Net migration','Infant mortality (per 1000 births)',

     'GDP ($ per capita)','Literacy (%)','Phones (per 1000)','Climate','Birthrate','Deathrate']

#from locale import atof

train[cols]=train[cols].astype(str).apply(lambda x: x.str.replace(',', '').astype(float), axis=1)

test[cols_test]=test[cols_test].astype(str).apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
#dropping columns which not be required

train=train.drop(['Area (sq. mi.)','Coastline (coast/area ratio)','Net migration','Industry','Phones (per 1000)','Arable (%)','Crops (%)','Other (%)','Climate','Agriculture','Service'], axis = 1)

test=test.drop(['Area (sq. mi.)','Coastline (coast/area ratio)','Net migration','Industry','Phones (per 1000)','Arable (%)','Crops (%)','Other (%)','Climate','Agriculture','Service'], axis = 1)



train.info()
#imputing most commonly applied values

def fillna(col):

    col.fillna(col.value_counts().index[0], inplace=True)

    return col

train=train.apply(lambda col:fillna(col))

test=test.apply(lambda col:fillna(col))

train.info()
#will bin temparature into 9 bins to see if any impact on confirmed cases/fatalities



train['quan']=pd.qcut(train['tempC'], q=[0,.1, .25, .5, .75,.80,.85,.90,.95,.99], labels=[1,2,3,4,5,6,7,8,9])

p=train[['tempC','quan','ConfirmedCases','Fatalities','Pop. Density (per sq. mi.)','Population']][(train.ConfirmedCases>0)]

lists=['Fatalities','Pop. Density (per sq. mi.)']



p.groupby(p['quan'])['tempC'].mean()#.reset_index()



p=p[['quan','ConfirmedCases','tempC','Fatalities','Pop. Density (per sq. mi.)','Population']].groupby(p['quan'])['ConfirmedCases','tempC','Fatalities','Pop. Density (per sq. mi.)','Population'].max().reset_index()

p=pd.DataFrame(p)

#sns.lineplot(x='tempC',y=lists, markers=True,data=p)

ax = p.plot(x="tempC", y="ConfirmedCases", legend=False,color='b')

ax2 = ax.twinx()

p.plot(x="tempC", y=["Pop. Density (per sq. mi.)","Fatalities"],ax=ax2, legend=False,color=['r','y'])

ax.figure.legend()

plt.show()
#dropping quan

train=train.drop(['quan'], axis = 1)

train.head()
#checing for correlation between different I/P parameters

train.corr()
#'tempC' removing for now once ml imputer is implemented will bring this back

train=train.drop(['tempC'], axis = 1)

test=test.drop(['tempC'], axis = 1)

'''

X=train[['Province_State', 'Country_Region', 'Date',  'Population', 'Pop. Density (per sq. mi.)',

       'Infant mortality (per 1000 births)', 'GDP ($ per capita)',

       'Literacy (%)', 'Birthrate', 'Deathrate']]

X1=test[['Province_State', 'Country_Region', 'Date',  'Population', 'Pop. Density (per sq. mi.)',

       'Infant mortality (per 1000 births)', 'GDP ($ per capita)',

       'Literacy (%)', 'Birthrate', 'Deathrate']]   

'''

X=train[['Province_State', 'Country_Region', 'Date']]

X1=test[['Province_State', 'Country_Region', 'Date']]   



Y_ConfirmedCases=train['ConfirmedCases']

Y_Fatalities=train['Fatalities']
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler

autoscaler = LabelEncoder()

X['Province_State']=autoscaler.fit_transform(X['Province_State'])

X['Country_Region']=autoscaler.fit_transform(X['Country_Region'])

X['Date']=autoscaler.fit_transform(X['Date'])

X1['Province_State']=autoscaler.fit_transform(X1['Province_State'])

X1['Country_Region']=autoscaler.fit_transform(X1['Country_Region'])

X1['Date']=autoscaler.fit_transform(X1['Date'])



minmaxscale=StandardScaler()



minmaxscale.fit(pd.concat([X,X1]))

                              

train_X=minmaxscale.fit_transform(X)

test_X=minmaxscale.fit_transform(X1)



tuned_models = [

               RandomForestRegressor(n_estimators= 8,

                               criterion= 'mse',

                               max_features = 'log2',#log2

                               min_samples_split = 60,

                               random_state = 40)]  

tuned_parameters = {    'base_estimator':tuned_models,

                        'loss' : ['exponential']#exponential

                        ,'random_state' : [43]

                        ,'learning_rate' : [0.1]

                         }

#exponential

clf_ConfirmedCases = GridSearchCV(AdaBoostRegressor(), tuned_parameters, cv=4)

clf_ConfirmedCases.fit(train_X,Y_ConfirmedCases)



clf_Fatalities = GridSearchCV(AdaBoostRegressor(), tuned_parameters, cv=4)

clf_Fatalities.fit(train_X,Y_Fatalities)
pred_xgbrf_ConfirmedCases = clf_ConfirmedCases.predict(train_X)

metrics.r2_score(Y_ConfirmedCases,pred_xgbrf_ConfirmedCases)
pred_xgbrf_Fatalities = clf_Fatalities.predict(train_X)

metrics.r2_score(Y_Fatalities,pred_xgbrf_Fatalities)
train_X1=train

train_X1=pd.concat([pd.DataFrame(pred_xgbrf_ConfirmedCases,columns=['ConfirmedCases_Predicted']),train_X1],axis=1)

train_X1=pd.concat([pd.DataFrame(pred_xgbrf_Fatalities,columns=['Fatalities_Predicted']),train_X1],axis=1)

train_X1[train_X1.Country_Region=='india']


clf_ConfirmedCases_pred_test=clf_ConfirmedCases.predict(test_X)



clf_Fatalities_pred_test=clf_Fatalities.predict(test_X)


test_X1=test

test_X1=pd.concat([pd.DataFrame(clf_ConfirmedCases_pred_test,columns=['ConfirmedCases_Predicted']),test_X1],axis=1)

test_X1=pd.concat([pd.DataFrame(clf_Fatalities_pred_test,columns=['Fatalities_Predicted']),test_X1],axis=1)

test_X1

test_X1[test_X1.Country_Region=='spain']
output=test_X1[['ForecastId','ConfirmedCases_Predicted','Fatalities_Predicted']].astype('int')

output.columns=['ForecastId','ConfirmedCases_Predicted','Fatalities_Predicted']

output.to_csv('submission.csv', index=False)