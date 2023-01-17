#Importing Libraries

import numpy as np #NumPy is a general-purpose array-processing package.

import pandas as pd #It contains high-level data structures and manipulation tools designed to make data analysis fast and easy.

import matplotlib.pyplot as plt #It is a Plotting Library

import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib.

from sklearn.linear_model import LogisticRegression #Logistic Regression is a Machine Learning classification algorithm

from sklearn.linear_model import LinearRegression #Linear Regression is a Machine Learning classification algorithm

from sklearn.model_selection import train_test_split #Splitting of Dataset

from sklearn.metrics import classification_report 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score
#reading the dataset

zomato_orgnl=pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")

zomato_orgnl.head() #This function returns the first n rows for the object based on position.
#Deleting Unnnecessary Columns

zomato=zomato_orgnl.drop(['url','dish_liked','phone'],axis=1) 
#Removing the Duplicates

zomato.duplicated().sum()

zomato.drop_duplicates(inplace=True)
#Remove the NaN values from the dataset

zomato.isnull().sum()

zomato.dropna(how='any',inplace=True)

zomato.info()
#Changing the Columns Names

zomato.columns

zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',

                                  'listed_in(city)':'city'})

zomato.columns
#Some Transformations

zomato['cost'] = zomato['cost'].astype(str)

zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.'))

zomato['cost'] = zomato['cost'].astype(float)

zomato.info()
#Removing '/5' from Rates

zomato['rate'].unique()

zomato = zomato.loc[zomato.rate !='NEW']

zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)

remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x

zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')

zomato['rate'].head()
#Encode the input Variables

def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:

        zomato[column] = zomato[column].factorize()[0]

    return zomato



zomato_en = Encode(zomato.copy())
#Get Correlation between different variables

corr = zomato_en.corr(method='kendall')

plt.figure(figsize=(15,8))

sns.heatmap(corr, annot=True)

zomato_en.columns
#Defining the independent variables and dependent variables

x = zomato_en.iloc[:,[2,3,5,6,7,8,9,11]]

y = zomato_en['rate']

#Getting Test and Training Set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)

x_train.head()

y_train.head()
#Prepare a Linear REgression Model

reg=LinearRegression()

reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
#Prepairng a Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)

DTree=DecisionTreeRegressor(min_samples_leaf=.0001)

DTree.fit(x_train,y_train)

y_predict=DTree.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_predict)
#Preparing Random Forest REgression

from sklearn.ensemble import RandomForestRegressor

RForest=RandomForestRegressor(n_estimators=500,random_state=329,min_samples_leaf=.0001)

RForest.fit(x_train,y_train)

y_predict=RForest.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_predict)
#Restaurants delivering Online or not

sns.countplot(zomato['online_order'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Restaurants delivering online or Not')
#Restaurants allowing table booking or not

sns.countplot(zomato['book_table'])

fig = plt.gcf()

fig.set_size_inches(10,10)

plt.title('Restaurants allowing table booking or not')
#Table booking Rate vs Rate

plt.rcParams['figure.figsize'] = (13, 9)

Y = pd.crosstab(zomato['rate'], zomato['book_table'])

Y.div(Y.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])

plt.title('table booking vs rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
# Location

sns.countplot(zomato['city'])

sns.countplot(zomato['city']).set_xticklabels(sns.countplot(zomato['city']).get_xticklabels(), rotation=90, ha="right")

fig = plt.gcf()

fig.set_size_inches(13,13)

plt.title('Location')
#Location and Rating

loc_plt=pd.crosstab(zomato['rate'],zomato['city'])

loc_plt.plot(kind='bar',stacked=True);

plt.title('Location - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Location',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');

plt.legend().remove();
#Type and Rating

type_plt=pd.crosstab(zomato['rate'],zomato['type'])

type_plt.plot(kind='bar',stacked=True);

plt.title('Type - Rating',fontsize=15,fontweight='bold')

plt.ylabel('Type',fontsize=10,fontweight='bold')

plt.xlabel('Rating',fontsize=10,fontweight='bold')

plt.xticks(fontsize=10,fontweight='bold')

plt.yticks(fontsize=10,fontweight='bold');