

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pandas as pd

from sklearn.impute import SimpleImputer

import numpy as np

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

        # HANDLING MISSING data

def fill_miessing_values_maually(obj, row, col, value):

    obj.iloc[row, col] = value

# -------------------------------------------------------------



# HANDLING MISSING data



def fill_miessing_values_automatic(obj, p_feature_name,method_type='nan',value =0 ):

    if method_type == 'mean':

           obj[p_feature_name].fillna(value = obj[p_feature_name].mean(), inplace=True)

    else:

         obj[p_feature_name].fillna(value)



# -------------------------------------------------------------

# ignoring tuples

def ignoringtuple(obj, igonoretype='all', value= 0 , colname=np.nan):

    if igonoretype == 'row':

         obj.drop([value], inplace=True, axis=0)

    if igonoretype == 'col':

        obj = obj[pd.notnull(obj[colname])]

    if igonoretype == 'all':  # drop any tuple containning null value

         obj.dropna(axis = 0, how ='any',inplace = True)

  





# -------------------------------------------------------------

def mean(numbers):

    return int(sum(numbers)) / max(len(numbers), 1)





# -----------------------------------------------------------------------------------------------------------------------

def binning_norm_by_mean(obj, frequent_number):

    tuples_count = len(obj)

    freq_stes = int(tuples_count / frequent_number)# dividing list debend on frequent_number

    arr1 = []



    for i in range(0, frequent_number):

        arr = []

        arr1.append([])

        for j in range(i * freq_stes, (i + 1) * freq_stes):

            if j >= tuples_count:

                break

            arr = arr + [obj[j]]

        arr1[i] = arr



    out = []

    for i in range(0, len(arr1)):

        out.append(mean(arr1[i]))



    return out



# Any results you write to the current directory are saved as output.
#reading data

data = pd.read_csv('/kaggle/input/MELBOURNE_HOUSE_PRICES.csv')

data.head()
print(data.isnull().sum())
#data['Price'] = data['Price'] / 1000 

fill_miessing_values_automatic(data, 'Price','mean')

print(data.isnull().sum())
# drop any field containing null value

print(len(data))

ignoringtuple(data,'all')

data.shape

print(len(data))

# remove duplications

data.drop_duplicates()

data.shape

# extract year and mounts from date 

data['Date'] = pd.to_datetime(data['Date'])

data['year'], data['month'] = data['Date'].dt.year, data['Date'].dt.month

data


#dropping columns

data.drop(['Address'  , 'SellerG','Suburb','Date'], axis=1, inplace=True)

print(data.describe())



region = {'Eastern Metropolitan':1,'Eastern Victoria':2,'Northern Metropolitan':3,'Northern Metropolitan':4,'South-Eastern Metropolitan':5,'South-Eastern Metropolitan':6,'Western Metropolitan':7,'Western Victoria':8,'Southern Metropolitan':9,'Northern Victoria':10}

data.Regionname = [region[item] for item in data.Regionname]

htype = {'h': 1,'t': 2,'u':3}

data.Type = [htype[item] for item in data.Type]

# convert method to integer 



method = {'PI': 1,'PN': 2,'S':3,'SA':4,'SN':5,'SP':6,'SS':7,'VB':8,'W':9}

data.Method = [method[item] for item in data.Method]



councilArea = {'Banyule City Council':1,'Bayside City Council':2,'Boroondara City Council':3,'Brimbank City Council':4,'Cardinia Shire Council':5,'Casey City Council':6,'Darebin City Council':7,'Frankston City Council':8,'Glen Eira City Council':9,'Greater Dandenong City Council':10,

'Hobsons Bay City Council':11,'Hume City Council':12,'Kingston City Council':13,'Knox City Council':14,'Macedon Ranges Shire Council':15,'Manningham City Council':16,'Manningham City Council':17,

'Maroondah City Council':18,'Melbourne City Council':19,'Melton City Council':20,'Mitchell Shire Council':21,

'Monash City Council':22,'Moonee Valley City Council':23,'Moorabool Shire Council':24,'Moreland City Council':25,'Murrindindi Shire Council':26,'Murrindindi Shire Council':27,'Port Phillip City Council':28,'Stonnington City Council':29,'Whitehorse City Council':30,'Whittlesea City Council':31,

'Wyndham City Council':32,'Yarra City Council':33,'Yarra Ranges Shire Council':34,'Maribyrnong City Council':35,'Nillumbik Shire Council':36}

data.CouncilArea = [councilArea[item] for item in data.CouncilArea]

data
import matplotlib.pyplot as plt

data.plot.scatter(x='Rooms', y='Price')

data.plot.scatter(x='year', y='Price')

data.plot.scatter(x='month', y='Price')

data.plot.scatter(x='Method', y='Price')

#X ( features )Data

X = data.drop(['Price'], axis=1, inplace=False)

#print('X Data is \n' , X.head())

#print('X shape is ' , X.shape)



#y ( label ) Data

y = data['Price']

#print('y Data is \n' , y.head())

#print('y shape is ' , y.shape)
print('X Data is \n' , X[:10])



#y Data

print('y Data is \n' , y[:10])
# for correlation coefficient test which measures the linear relationship between two datasets

from scipy.stats import pearsonr

from scipy import stats

stats.pearsonr(X['Rooms'],y)

stats.pearsonr(X['Distance'],y)

stats.pearsonr(X['Postcode'],y)

stats.pearsonr(X['Propertycount'],y)

from sklearn.feature_selection import GenericUnivariateSelect, chi2,f_regression,f_classif

# score_func = SelectKBest , chi2 ,f_regression ,SelectPercentile ,SelectKBest ,f_classif

#{‘percentile’, ‘k_best’, ‘fpr’, ‘fdr’, ‘fwe’} for featuer selection mode 



#allows to perform univariate feature selection with a configurable strategy. 

#This allows to select the best univariate selection strategy with hyper-parameter search estimator.

#These objects take as input a scoring function that returns univariate scores and p-values 

#(or only scores for SelectKBest and SelectPercentile):



#For regression: f_regression, mutual_info_regression

#For classification: chi2, f_classif, mutual_info_classif

    

FeatureSelection = GenericUnivariateSelect(f_regression, 'k_best', param=10)

X = FeatureSelection.fit_transform(X, y)

X[:10]
#Splitting data



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50, shuffle =True)



#Splitted Data

print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)

X_train