import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from matplotlib import rcParams

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



zomato = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')

zomato.head()
zomato.shape
zomato.describe()
zomato.info()
#Checking the null values

zomato.isnull().sum()
#Removing the null values

zomato.dropna(how = 'any', inplace = True)
zomato.isnull().sum()
zomato.columns
#Dropping the columns which are not relevant

zomato=zomato.drop(['url','dish_liked','phone'],axis=1)
#Upadating few columns 

zomato = zomato.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type',

                                  'listed_in(city)':'city'})

zomato.columns
zomato.head()
#Checking for duplicate values

zomato.duplicated().sum()
#Dropping the duplicate values

zomato.drop_duplicates(inplace = True)
zomato.shape
#Simplifying the ratings column

#Removing '/5' from Rates

zomato['rate'].unique()

zomato = zomato.loc[zomato.rate !='NEW']

zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)

remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x

zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')
zomato['rate'].unique()
rcParams['figure.figsize'] = (20,10)

sb.countplot(zomato['city'])

sb.countplot(zomato['city']).set_xticklabels(sb.countplot(zomato['city']).get_xticklabels(), rotation=90, ha="right")

plt.title('Number of Restaurants in Every Area', size = 20)

plt.show()
rcParams['figure.figsize'] = (10,10)

sb.countplot(zomato['online_order'])

plt.title('Count of Restaurants taking Online orders', size = 20)

plt.show()
rcParams['figure.figsize'] = (10,10)

sb.countplot(zomato['book_table'])

plt.title('Table Booking', size = 20)

plt.show()
rcParams['figure.figsize'] = (20,10)

sb.countplot(zomato['rest_type'])

sb.countplot(zomato['rest_type']).set_xticklabels(sb.countplot(zomato['rest_type']).get_xticklabels(), rotation=90, ha="right")

plt.title('Type of Restaurant', size = 20)

plt.show()
rcParams['figure.figsize'] = (20,10)

sb.countplot(zomato['type'])

sb.countplot(zomato['type']).set_xticklabels(sb.countplot(zomato['type']).get_xticklabels(), rotation=90, ha="right")

plt.title('Type of Service', size = 20)

plt.show()
rcParams['figure.figsize'] = (20,10)

sb.countplot(zomato['cost'])

sb.countplot(zomato['cost']).set_xticklabels(sb.countplot(zomato['cost']).get_xticklabels(), rotation=90, ha="right")

plt.title('Count of Restaurants based on Cost', size = 20)

plt.show()
rcParams['figure.figsize'] = (20,10)

sb.countplot(zomato['location'])

sb.countplot(zomato['location']).set_xticklabels(sb.countplot(zomato['location']).get_xticklabels(), rotation=90, ha="right")

plt.title('Count of Restaurant in Every Location', size = 20)

plt.show()
rcParams['figure.figsize'] = (20,10)

top = zomato['name'].value_counts()[:20]

sb.barplot(x = top, y = top.index, palette = 'cool_r')

plt.xlabel("Number of outlets",size=15)

plt.title('Top 20 Restaurants', size = 20)

plt.show()
zomato.head()
#Cost Transformations

zomato['cost'] = zomato['cost'].astype(str)

zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',','.'))

zomato['cost'] = zomato['cost'].astype(float)
zomato.cost.unique()
#Other Transformations

zomato.name = zomato.name.apply(lambda x:x.title())

zomato.online_order.replace(('Yes','No'),(True, False),inplace=True)

zomato.book_table.replace(('Yes','No'),(True, False),inplace=True)
#Encoding of input Variables

def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'cost', 'votes'])]:

        zomato[column] = zomato[column].factorize()[0]

    return zomato



zomato_en = Encode(zomato.copy())
rcParams['figure.figsize'] = (15,10)

sb.heatmap(zomato_en.corr(), annot=True)

plt.title('Correlation Heatmap', size = 20)

plt.show()
#Defining the independent variables and dependent variables

x = zomato_en.iloc[:,[2,3,5,6,7,8,9,11]]

y = zomato_en['rate']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2)
#Preparing Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

random = RandomForestRegressor(n_estimators=500)

random.fit(x_train,y_train)
print(" Accuracy :{}" .format(random.score(x_test, y_test)))
# Importing Intensity Analyser to check the intensity status of the given review

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Selecting 5 sample reviews to analyse the intensity

restaurant_rev = zomato['reviews_list'].value_counts()

restaurant_rev.sample(5)
# Picking random reviews from the samples to analyse

# fitting the review as sentence for analysis

rev = ["Best place to take your family or the one you have a comfortable relationship with. A no-frills place serving good quality but not extraordinary food.",

      "Awesome place with good ambience and service food was superb , rabbidi was finger licking... A must try ",

      "Visited this place for lunch.. we opted ala carte.. let me remind you the place is quite expensive for the amount it serves. We ordered 2 naan, one malai kofta, 2 shikanji and the bill was 1250 bucks.",

      "I really did not like it.. it is far from fine dining.. it will just burn a hole in your pocket for no reason.",

      "I love the food, but when it comes cleanliness it is zero. Take away is good. But being a budget friendly place, i love it. I am not saying the food is amazing , but yeah it is good. They definitely need a revamp."]
sid = SentimentIntensityAnalyzer()

for sentence in rev:

    print(sentence)

    ss = sid.polarity_scores(sentence)

    for k in ss:

        print('{0}: {1}, '.format(k, ss[k]), end = '\n')

    print()