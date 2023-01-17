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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sqlalchemy import create_engine

import warnings



warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
#load data

lifeexpectancy = pd.read_csv('/kaggle/input/life-expectancy-who/Life Expectancy Data.csv', delimiter=',')

lifeexpectancy.dataframeName = 'Life Expectancy Data.csv'


lifeexpectancy.head()
#Looking for missing values, data types



def eda(dataframe):

    print('MISSING VALUES\n', dataframe.isnull().sum())

    print('\n DATA TYPES \n', dataframe.dtypes)

    print('\n DATA SHAPE \n', dataframe.shape)

    print('\n DATA DESCRIBE \n', dataframe.describe())

    for item in dataframe:

        print('\n UNIQUE VALUE TOTALS \n',item)

        print(dataframe[item].nunique())

        print(dataframe[item].value_counts())     

eda(lifeexpectancy)
#Assignin missing columns

missing_columns = list(lifeexpectancy.columns[lifeexpectancy.isnull().any()])

missing_columns
#filling missing columns with interpolation method. 



for col in missing_columns:

    lifeexpectancy.loc[:, col] = lifeexpectancy.loc[:, col].interpolate()
#Yay! No more missing values!

lifeexpectancy.info()
xx = lifeexpectancy.iloc[:,3:]

#xx.head



plt.figure(figsize=(20,30))



for col_names in list(xx.columns):

    

    plt.subplot(5,4,(list(xx.columns).index(col_names)+1))

    plt.boxplot(lifeexpectancy[col_names], whis=1.5)

    plt.title(col_names)

    

plt.show()
# NumPy's percentile() method returns the 

# values of the given percentiles. In our case,

# we give 75 and 25 as parameters which corresponds 

# to the third and the first quartile.



for variable in list(xx.columns):

   q75, q25 = np.percentile(lifeexpectancy[variable], [75 ,25])

   iqr = q75 - q25



   min_val = q25 - (iqr*1.5)

   max_val = q75 + (iqr*1.5)

   print("Number of outliers and percentage of it in {} : {} and {}".format(variable,

                                                                             len((np.where((lifeexpectancy[variable] > max_val) | 

                                                                                           (lifeexpectancy[variable] < min_val))[0])),len((np.where((lifeexpectancy[variable] > max_val) | 

                                                                                           (lifeexpectancy[variable] < min_val))[0]))*100/2938))  
from scipy.stats.mstats import winsorize



#applying winsorize techniqu to cap the outliers and adding the new winsorized column to our dataset

#After adding the new winsorized column, I plot the before and after versions of columns with a boxplot



lifeexpectancy['winsorized_infant_deaths'] = winsorize(lifeexpectancy["infant deaths"], (0, 0.11))



plt.figure(figsize=(10,60))



plt.subplot(19,2,1)

plt.boxplot(lifeexpectancy["infant deaths"])

plt.title("Box Plot of Infant Deaths Before Winsorize")



plt.subplot(19,2,2)

plt.boxplot(lifeexpectancy['winsorized_infant_deaths'])

plt.title("Box Plot of Infant Deaths After Winsorize")



##



lifeexpectancy['winsorized_percentage_expenditure'] = winsorize(lifeexpectancy["percentage expenditure"], (0, 0.1325))



plt.subplot(19,2,3)

plt.boxplot(lifeexpectancy["percentage expenditure"])

plt.title("Box Plot of Percentage Expenditure Before Winsorize")



plt.subplot(19,2,4)

plt.boxplot(lifeexpectancy['winsorized_percentage_expenditure'])

plt.title("Box Plot of Percentage Expenditure After Winsorize")



##



lifeexpectancy['winsorized_hepatitis_B'] = winsorize(lifeexpectancy["Hepatitis B"], (0.10, 0))



#plt.figure(figsize=(12,3))



plt.subplot(19,2,5)

plt.boxplot(lifeexpectancy["Hepatitis B"])

plt.title("Box Plot of Hepatitis B Before Winsorize")



plt.subplot(19,2,6)

plt.boxplot(lifeexpectancy['winsorized_hepatitis_B'] )

plt.title("Box Plot of Hepatitis B After Winsorize")



##



lifeexpectancy['winsorized_Measles'] = winsorize(lifeexpectancy["Measles "], (0, 0.1846))

#plt.figure(figsize=(12,3))

plt.subplot(19,2,7)

plt.boxplot(lifeexpectancy["Measles "])

plt.title("Box Plot of Measles Before Winsorize")



plt.subplot(19,2,8)

plt.boxplot(lifeexpectancy['winsorized_Measles'])

plt.title("Box Plot of Measles After Winsorize")



##



lifeexpectancy['winsorized_under'] = winsorize(lifeexpectancy['under-five deaths '], (0, 0.1342))



plt.subplot(19,2,9)

plt.boxplot(lifeexpectancy["under-five deaths "])

plt.title("Box Plot of under-five deaths Before Winsorize")



plt.subplot(19,2,10)

plt.boxplot(lifeexpectancy['winsorized_under'])

plt.title("Box Plot of under-five deaths After Winsorize")



##



lifeexpectancy['winsorized_polio'] = winsorize(lifeexpectancy['Polio'], (0.10, 0))



plt.subplot(19,2,11)

plt.boxplot(lifeexpectancy["Polio"])

plt.title("Box Plot of Polio Before Winsorize")



plt.subplot(19,2,12)

plt.boxplot(lifeexpectancy['winsorized_polio'])

plt.title("Box Plot of Polio After Winsorize")



##



lifeexpectancy['winsorized_total_expenditure'] = winsorize(lifeexpectancy['Total expenditure'], (0, 0.012))



plt.subplot(19,2,13)

plt.boxplot(lifeexpectancy["Total expenditure"])

plt.title("Box Plot of Total expenditure Before Winsorize")



plt.subplot(19,2,14)

plt.boxplot(lifeexpectancy['winsorized_total_expenditure'])

plt.title("Box Plot of Total expenditure After Winsorize")



##

 



lifeexpectancy['winsorized_Diphtheria'] = winsorize(lifeexpectancy['Diphtheria '], (0.11, 0))



plt.subplot(19,2,15)

plt.boxplot(lifeexpectancy["Diphtheria "])

plt.title("Box Plot of Diphtheria Before Winsorize")



plt.subplot(19,2,16)

plt.boxplot(lifeexpectancy['winsorized_Diphtheria'])

plt.title("Box Plot of Diphtheria After Winsorize")



##



lifeexpectancy['winsorized_Population'] = winsorize(lifeexpectancy['Population'], (0, 0.125))



plt.subplot(19,2,17)

plt.boxplot(lifeexpectancy["Population"])

plt.title("Box Plot of Population Before Winsorize")



plt.subplot(19,2,18)

plt.boxplot(lifeexpectancy['winsorized_Population'])

plt.title("Box Plot of Population After Winsorize")



##



lifeexpectancy['winsorized_HIV_AIDS'] = winsorize(lifeexpectancy[' HIV/AIDS'], (0, 0.185))



plt.subplot(19,2,19)

plt.boxplot(lifeexpectancy[" HIV/AIDS"])

plt.title("Box Plot of HIV/AIDS Before Winsorize")



plt.subplot(19,2,20)

plt.boxplot(lifeexpectancy['winsorized_HIV_AIDS'])

plt.title("Box Plot of HIV/AIDS After Winsorize")

##



lifeexpectancy['winsorized_GDP'] = winsorize(lifeexpectancy['GDP'], (0, 0.12))



plt.subplot(19,2,21)

plt.boxplot(lifeexpectancy["GDP"])

plt.title("Box Plot of Diphtheria Before Winsorize")



plt.subplot(19,2,22)

plt.boxplot(lifeexpectancy['winsorized_GDP'])

plt.title("Box Plot of Diphtheria After Winsorize")



## thinness  1-19 years



lifeexpectancy['winsorized_thinness1_19'] = winsorize(lifeexpectancy[' thinness  1-19 years'], (0, 0.031))



plt.subplot(19,2,23)

plt.boxplot(lifeexpectancy[" thinness  1-19 years"])

plt.title("Box Plot of thinness 1-19 Before Winsorize")



plt.subplot(19,2,24)

plt.boxplot(lifeexpectancy['winsorized_thinness1_19'])

plt.title("Box Plot of thinness 1-19 After Winsorize")



##



lifeexpectancy['winsorized_thinness5_9'] = winsorize(lifeexpectancy[' thinness 5-9 years'], (0, 0.03302))



plt.subplot(19,2,25)

plt.boxplot(lifeexpectancy[" thinness 5-9 years"])

plt.title("Box Plot of thinness 5-9 Before Winsorize")



plt.subplot(19,2,26)

plt.boxplot(lifeexpectancy['winsorized_thinness5_9'])

plt.title("Box Plot of thinness 5-9 After Winsorize")



##



lifeexpectancy['winsorized_income'] = winsorize(lifeexpectancy['Income composition of resources'], (0.05, 0.0425))



plt.subplot(19,2,27)

plt.boxplot(lifeexpectancy["Income composition of resources"])

plt.title("Box Plot of income Before Winsorize")



plt.subplot(19,2,28)

plt.boxplot(lifeexpectancy['winsorized_income'])

plt.title("Box Plot of income After Winsorize")



##



lifeexpectancy['winsorized_schooling'] = winsorize(lifeexpectancy['Schooling'], (0.1804,0.0011))



plt.subplot(19,2,29)

plt.boxplot(lifeexpectancy["Schooling"])

plt.title("Box Plot of schooling Before Winsorize")



plt.subplot(19,2,30)

plt.boxplot(lifeexpectancy['winsorized_schooling'])

plt.title("Box Plot of schooling After Winsorize")



##



lifeexpectancy['winsorized_lifeexpenctancy'] = winsorize(lifeexpectancy['Life expectancy '], (0.0409, 0))



plt.subplot(19,2,31)

plt.boxplot(lifeexpectancy['Life expectancy '])

plt.title("Box Plot of life expec. Before Winsorize")



plt.subplot(19,2,32)

plt.boxplot(lifeexpectancy['winsorized_lifeexpenctancy'] )

plt.title("Box Plot of life expec. After Winsorize")



##



lifeexpectancy['winsorized_adult_mortality'] = winsorize(lifeexpectancy['Adult Mortality'], (0, 0.028))



plt.subplot(19,2,33)

plt.boxplot(lifeexpectancy['Adult Mortality'])

plt.title("Box Plot of adult mortality Before Winsorize")



plt.subplot(19,2,34)

plt.boxplot(lifeexpectancy['winsorized_adult_mortality'])

plt.title("Box Plot of adult mortality After Winsorize")





####



lifeexpectancy['winsorized_BMI'] = winsorize(lifeexpectancy[' BMI '], (0.10, 0.10))

#making a new dataframe for winsorized columns.

winsorized_lifeexpec = lifeexpectancy.iloc[:,22:]



#checking the situation of outliers. 

for variable in list(winsorized_lifeexpec.columns):

   q75, q25 = np.percentile(winsorized_lifeexpec[variable], [75 ,25])

   iqr = q75 - q25



   min_val = q25 - (iqr*1.5)

   max_val = q75 + (iqr*1.5)

   print("Number of outliers and percentage of it in {} : {} and {}".format(variable,

                                                                             len((np.where((winsorized_lifeexpec[variable] > max_val) | 

                                                                                           (winsorized_lifeexpec[variable] < min_val))[0])),len((np.where((winsorized_lifeexpec[variable] > max_val) | 

                                                                                           (winsorized_lifeexpec[variable] < min_val))[0]))*100/2938))  
#target = life expectancy

#using univariate and multivariate exploration techniques



lifeexpectancy.describe()
winsorized_lifeexpec.describe()
# descriptive statistics for just text-based variables

lifeexpectancy.describe(include=['O'])
plot_list = [

       'Life expectancy ',  'winsorized_lifeexpenctancy',

       'Adult Mortality','winsorized_adult_mortality',

       'infant deaths', 'winsorized_infant_deaths',

       #'Alcohol', 

       'percentage expenditure', 'winsorized_percentage_expenditure',

       'Hepatitis B','winsorized_hepatitis_B',

       'Measles ',  'winsorized_Measles',

       ' BMI ', 'winsorized_BMI',

       'under-five deaths ', 'winsorized_under',

       'Polio','winsorized_polio', 

       'Total expenditure','winsorized_total_expenditure',

       'Diphtheria ', 'winsorized_Diphtheria',

       ' HIV/AIDS', 'winsorized_HIV_AIDS',

       'GDP', 'winsorized_GDP',

       'Population',   'winsorized_Population',

       ' thinness  1-19 years', 'winsorized_thinness1_19',

       ' thinness 5-9 years', 'winsorized_thinness5_9',

       'Income composition of resources','winsorized_income',  

       'Schooling','winsorized_schooling'

]



plt.figure(figsize=(20,45))



for col_names in plot_list:

    

    plt.subplot(9,4,(plot_list.index(col_names)+1))

    plt.hist(lifeexpectancy[col_names])

    plt.title(col_names)

    

plt.show()







plt.figure(figsize=(20,45))



plt.subplot(9,4,36)

plt.hist(lifeexpectancy['Alcohol'])

plt.title('Alcohol')

plt.show()

plt.barh( lifeexpectancy["Status"].unique(),

          lifeexpectancy.groupby(["Status"])["Country"].nunique(),

          color = ["red", "green"],

          tick_label = ["Developed", "Developing"]

        )
aa = lifeexpectancy[lifeexpectancy["Status"] == "Developed"].groupby(["Year"])['winsorized_lifeexpenctancy'].mean()

ab = lifeexpectancy[lifeexpectancy["Status"] == "Developing"].groupby(["Year"])['winsorized_lifeexpenctancy'].mean()



plt.figure(figsize = (10,5))

plt.plot(aa)

plt.plot(ab)

plt.legend(["Developed countries", "Developing Countries"])

plt.title("Life Expectancy Improvement Between Developed and Developing Countries")

plt.ylabel("Life Expectancy")
import scipy.stats as stats

stats.ttest_ind(lifeexpectancy.loc[lifeexpectancy['Status']=='Developed','Life expectancy '],lifeexpectancy.loc[lifeexpectancy['Status']=='Developing','Life expectancy '])
sorted_le = (lifeexpectancy.groupby('Country')['winsorized_lifeexpenctancy'].mean()).sort_values()

plt.figure(figsize=(20,6))

xx = sns.barplot(x = sorted_le.index, y = sorted_le)

xx.set_xticklabels(xx.get_xticklabels(), rotation=90, fontsize=8)

xx.set_title('Countries with respect to life expectancy')

plt.show()
plt.figure(figsize=(20,30))



for col_names in winsorized_lifeexpec.columns:

    

    plt.subplot(5,4,(list(winsorized_lifeexpec.columns).index(col_names)+1))

    plt.scatter(lifeexpectancy['winsorized_lifeexpenctancy'], lifeexpectancy[col_names])

    plt.title(col_names)

    

plt.show()
plt.figure(figsize=(12,12))

sns.heatmap(winsorized_lifeexpec.corr(), square=True, annot=True, linewidths=.5)

plt.title("correlation matrix (winsorized_lifeexpec)")
# append dummies to lifeexpectancy dataframe

lifeexpectancy = pd.concat([lifeexpectancy, pd.get_dummies(lifeexpectancy['Status'], drop_first=True)], axis =1)

lifeexpectancy
#creating a new variable from highly correlated 3 variable by using PCA



from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



X = lifeexpectancy[["winsorized_hepatitis_B", "winsorized_polio",

                  "winsorized_Diphtheria"]]



X = StandardScaler().fit_transform(X)



sklearn_pca = PCA(n_components=1)

lifeexpectancy["pca_1"] = sklearn_pca.fit_transform(X)



print(

    'The percentage of total variance in the dataset explained by each',

    'component from Sklearn PCA.\n',

    sklearn_pca.explained_variance_ratio_

)



X_check = lifeexpectancy[["winsorized_hepatitis_B", "winsorized_polio",

                  "winsorized_Diphtheria", "winsorized_lifeexpenctancy", "pca_1"]]

X_check.corr() 
lifeexpectancy_possible_features = lifeexpectancy[['winsorized_percentage_expenditure','winsorized_under', 'winsorized_Population', 'winsorized_HIV_AIDS',

'winsorized_lifeexpenctancy','Developing', 'pca_1']]
#machine learning models assume that all features have values in the same range (e.g., a min of 0 and a max of 1) or they exhibit normal statistical 

#properties. For some techniques, features that vary in range can result in incorrect estimates and results. To be able to apply these techniques and 

#methods, we need to rescale our variables to fit a limited range, or standardize our variables to exhibit some regular statistical patterns.



from sklearn.preprocessing import normalize



# normalize the winsorized variables

lifeexpectancy_possible_features["norm_winsorized_percentageexpenditure"] = normalize(np.array(lifeexpectancy_possible_features['winsorized_percentage_expenditure']).reshape(1,-1)).reshape(-1,1)

lifeexpectancy_possible_features["norm_winsorized_under"] = normalize(np.array(lifeexpectancy_possible_features['winsorized_under']).reshape(1,-1)).reshape(-1,1)

lifeexpectancy_possible_features["norm_winsorized_population"] = normalize(np.array(lifeexpectancy_possible_features['winsorized_Population']).reshape(1,-1)).reshape(-1,1)

lifeexpectancy_possible_features["norm_winsorized_HIV_AIDS"] = normalize(np.array(lifeexpectancy_possible_features['winsorized_HIV_AIDS']).reshape(1,-1)).reshape(-1,1)

lifeexpectancy_possible_features["norm_winsorized_lifeexpectancy"] = normalize(np.array(lifeexpectancy_possible_features['winsorized_lifeexpenctancy']).reshape(1,-1)).reshape(-1,1)

lifeexpectancy_possible_features["norm_developing"] = normalize(np.array(lifeexpectancy_possible_features['Developing']).reshape(1,-1)).reshape(-1,1)

lifeexpectancy_possible_features["pca_deseases"] = normalize(np.array(lifeexpectancy_possible_features['pca_1']).reshape(1,-1)).reshape(-1,1)



lifeexpectancy_possible_features = lifeexpectancy_possible_features.iloc[:,7:]
plt.figure(figsize=(10,6))

sns.scatterplot(lifeexpectancy_possible_features['norm_winsorized_lifeexpectancy'], lifeexpectancy_possible_features['norm_winsorized_population']).set_title('Life expectancy vs Population')

plt.figure(figsize=(12,12))

sns.heatmap(lifeexpectancy_possible_features.corr(), square=True, annot=True, linewidths=.5)

plt.title("correlation matrix (winsorized_lifeexpec)")