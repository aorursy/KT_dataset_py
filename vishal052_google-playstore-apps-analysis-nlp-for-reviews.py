import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



import warnings

warnings.filterwarnings('ignore')



from pylab import rcParams



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

data.head()
data.shape
data.columns
#Finding the missing data

sb.heatmap(pd.isnull(data))
#Evaluating the missing values

missing_values = data.isnull().sum().sort_values(ascending = False)

missing_values
#Dropping the Missing values

data.dropna(how = 'any', inplace = True)

missing_values = data.isnull().sum().sort_values(ascending = False)

missing_values
sb.heatmap(pd.isnull(data))
data.shape
#Evaluating the data for Rating field

data['Rating'].describe()
plt.rcParams['figure.figsize'] = (15, 10)

count_graph = sb.countplot(data['Rating'])

count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)

count_graph

plt.title('Count of Apps ', size = 20)
plt.rcParams['figure.figsize'] = (15, 10)

sb.distplot(data.Rating, color = 'red', hist = False)

plt.title('Rating Distribution', size = 20)
#Categorical Evaluation of Apps

print(data['Category'].unique())

print('\n', len(data['Category'].unique()), 'Categories')
plt.rcParams['figure.figsize'] = (15, 10)

count_graph = sb.countplot(data['Category'])

count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)

count_graph

plt.title('Count of Apps in each Category', size = 20)
plt.rcParams['figure.figsize'] = (15, 10)

graph = sb.boxplot(y = data['Rating'], x = data['Category'])

sb.despine(left = True)

graph.set_xticklabels(graph.get_xticklabels(), rotation = 90)

graph

plt.title('Box Plot of Rating VS Category', size = 20)

plt.show()
#Evaluating the data for Reviews

data['Reviews'].describe()
plt.rcParams['figure.figsize'] = (15, 10)

sb.distplot(data.Reviews, color = 'red', hist = False)

plt.title('Reviews Distribution', size = 20)

plt.show()
#Convertings the Reviews object data into int type to plot comparision graph 

data['Reviews'] = data['Reviews'].apply(lambda x: int(x))
rcParams['figure.figsize'] = (15, 10)

sb.jointplot(data = data, x = "Reviews", y = "Rating", size = 10)
rcParams['figure.figsize'] = (15, 10)

sb.regplot(data = data, x = 'Reviews', y = 'Rating')

plt.title('Reviews VS Rating', size = 20)
data['Size'].head()
data['Size'].unique()
len(data[data.Size == 'Varies with device'])
data['Size'].replace('Varies with device', np.nan , inplace = True)
#Converting the object data type into int 

data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \

             data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)

            .fillna(1)

            .replace(['k','M'], [10**3, 10**6]).astype(int))
data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'), inplace = True)
rcParams['figure.figsize'] = (15, 10)

sb.jointplot(x = 'Size', y = 'Rating', data = data, size = 10 )
data['Installs'].head()
data['Installs'].unique()
plt.rcParams['figure.figsize'] = (15, 10)

count_graph = sb.countplot(data['Installs'])

count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)

count_graph

plt.title('Count of Apps ', size = 20)
#Converting the Object data into interger data

data.Installs = data.Installs.apply(lambda x: x.replace(',',''))

data.Installs = data.Installs.apply(lambda x: x.replace('+',''))

data.Installs = data.Installs.apply(lambda x: int(x))
data['Installs'].unique()
#Sorting the values

sorted_value = sorted(list(data['Installs'].unique()))
data['Installs'].replace(sorted_value, range(0, len(sorted_value), 1), inplace = True)

rcParams['figure.figsize'] = (15, 10)

sb.regplot(x = 'Installs', y = 'Rating', data = data)

plt.title("Ratings VS Installs", size = 20)
data['Type'].unique()
plt.rcParams['figure.figsize'] = (15, 10)

count_graph = sb.countplot(data['Type'])

count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)

count_graph

plt.title('Count of Apps ', size = 20)
labels = data['Type'].value_counts(sort = True).index

size = data['Type'].value_counts(sort = True)



explode = (0.1, 0)



rcParams['figure.figsize'] = (10, 10)



plt.pie(size, explode = explode, labels = labels, autopct = '%.2f%%', shadow = True)



plt.title("Perceantage of Free Apps in Playstore", size = 20)

plt.show()
#For Evaluation of Paid Apps only, I will consider the all the free apps as a single record

data['Free'] = data['Type'].map(lambda s :1  if s =='Free' else 0)

data.drop(['Type'], axis=1, inplace=True)
data['Price'].unique()
data.Price = data.Price.apply(lambda x: x.replace('$',''))

data['Price'] = data['Price'].apply(lambda x: float(x))
data['Price'].describe()
rcParams['figure.figsize'] = (15, 10)

sb.regplot(x = 'Price', y = 'Rating', data = data)

plt.title(" Price VS Rating", size = 20)
data['Content Rating'].unique()
plt.rcParams['figure.figsize'] = (15, 10)

count_graph = sb.countplot(data['Content Rating'])

count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)

count_graph

plt.title('Count of Apps ', size = 20)
rcParams['figure.figsize'] = (15, 10)

sb.boxplot(x = 'Content Rating', y = 'Rating', data = data)

plt.title("Content Rating VS Rating", size = 20)
data['Genres'].unique()
len(data['Genres'].unique())
data.Genres.value_counts()
#Grouping to ignore sub-genre

data['Genres'] = data['Genres'].str.split(';').str[0]
print(data['Genres'].unique())

print('\n', len(data['Genres'].unique()), 'genres')
plt.rcParams['figure.figsize'] = (20, 10)

count_graph = sb.countplot(data['Genres'])

count_graph.set_xticklabels(count_graph.get_xticklabels(), rotation = 90)

count_graph

plt.title('Count of Apps', size = 20)
rcParams['figure.figsize'] = (20,10)

graph = sb.boxplot(x = 'Genres', y = 'Rating', data = data)

graph.set_xticklabels(graph.get_xticklabels(), rotation = 90)

graph

plt.title('Rating VS Genres', size = 20)
rev = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv')

rev.head()
rev = pd.concat([rev.Translated_Review, rev.Sentiment], axis = 1)

rev.dropna(axis = 0, inplace = True)

rev.head(10)
rev.Sentiment.value_counts()
rev.Sentiment = [0 if i == 'Positive' else 1 if i == 'Negative' else 2 for i in rev.Sentiment]

rev.head(10)
# Removing characters that are not letters & converting them to lower case

import re

first_text = rev.Translated_Review[0]

text = re.sub('[^a-zA-Z]',' ', first_text)

text = text.lower()
print(rev.Translated_Review[0])

print(text)
#Tokenize to seperate each word

import nltk as nlp

from nltk.corpus import stopwords

text = nlp.word_tokenize(text)

text
#Lemmatization to convert words to their root forms

lemma = nlp.WordNetLemmatizer()

text = [lemma.lemmatize(i) for i in text]

text = " ".join(text)

text
text_list = []

for i in rev.Translated_Review:

    text = re.sub('[^a-zA-Z]',' ', i)

    text = text.lower()

    text = nlp.word_tokenize(text)

    lemma = nlp.WordNetLemmatizer()

    text = [lemma.lemmatize(i) for i in text]

    text = " ".join(text)

    text_list.append(text)

    

text_list[:10]    
from sklearn.feature_extraction.text import CountVectorizer

max_features = 200000

cou_vec = CountVectorizer(max_features = max_features, stop_words = 'english')

sparce_matrix = cou_vec.fit_transform(text_list).toarray()

all_words = cou_vec.get_feature_names()

print('Most used words :', all_words[:100])
y = rev.iloc[:,1].values

x = sparce_matrix
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
from sklearn.ensemble import RandomForestClassifier

random = RandomForestClassifier(n_estimators = 10)

random.fit(x_train, y_train)
print("Accuracy: ",random.score(x_test,y_test))
y_pred = random.predict(x_test)

y_true = y_test
#Confusion Matrix

from sklearn.metrics import confusion_matrix

names = ['Positive', 'Negative', 'Neutral']

cm = confusion_matrix(y_true, y_pred)

f, ax = plt.subplots(figsize = (5,5))

sb.heatmap(cm, annot = True, fmt = '0.2f')

plt.xlabel('y_pred')

plt.ylabel('y_true')

ax.set_xticklabels(names)

ax.set_yticklabels(names)