# Import libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#from sklearn_pandas import DataFrameMapper # Notice that this is actually Sklearn-Pandas library

%matplotlib inline
# Load dataset

data = pd.read_csv('../input/gender-classifier-DFE-791531.csv', encoding='latin-1')



# Drop unnecessary columns/features

data.drop (columns = ['_unit_id',

                      '_last_judgment_at',

                      'user_timezone',

                      'tweet_coord',

                      'tweet_count',

                      'tweet_created', 

                      'tweet_id',

                      'tweet_location',

                      'profileimage',

                      'created'], inplace = True)



data.info()
data.head(3)
data['gender'].value_counts()

# We can see that there are 1117 unknown genders, so get rid of them
drop_items_idx = data[data['gender'] == 'unknown'].index



data.drop (index = drop_items_idx, inplace = True)



data['gender'].value_counts()
print ('profile_yn information:\n',data['profile_yn'].value_counts())



data[data['profile_yn'] == 'no']['gender']
drop_items_idx = data[data['profile_yn'] == 'no'].index



data.drop (index = drop_items_idx, inplace = True)



print (data['profile_yn'].value_counts())



data.drop (columns = ['profile_yn','profile_yn:confidence','profile_yn_gold'], inplace = True)
# Double check the data 

print (data['gender'].value_counts())



print ('---------------------------')

data.info()
print ('Full data items: ', data.shape)

print ('Data with label-confidence < 100%: ', data[data['gender:confidence'] < 1].shape)
drop_items_idx = data[data['gender:confidence'] < 1].index



data.drop (index = drop_items_idx, inplace = True)



print (data['gender:confidence'].value_counts())



data.drop (columns = ['gender:confidence'], inplace = True)
data.drop (columns = ['_golden','_unit_state','_trusted_judgments','gender_gold'], inplace = True)



# Double check the data 

print (data['gender'].value_counts())



print ('---------------------------')

data.info()
from collections import Counter



twit_vocab = Counter()

for twit in data['text']:

    for word in twit.split(' '):

        twit_vocab[word] += 1

        

# desc_vocab = Counter()

# for twit in data['description']:

#     for word in twit.split(' '):

#         desc_vocab[word] += 1

        

twit_vocab.most_common(20)

# desc_vocab.most_common(20)
import nltk



nltk.download('stopwords')
from nltk.corpus import stopwords

stop = stopwords.words('english')



twit_vocab_reduced = Counter()

for w, c in twit_vocab.items():

    if not w in stop:

        twit_vocab_reduced[w]=c



twit_vocab_reduced.most_common(20)
import re



def preprocessor(text):

    """ Return a cleaned version of text

    """

    # Remove HTML markup

    text = re.sub('<[^>]*>', '', text)

    # Save emoticons for later appending

    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    # Remove any non-word character and append the emoticons,

    # removing the nose character for standarization. Convert to lower case

    text = (re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-', ''))

    

    return text



print(preprocessor('This!!@ twit :) is <b>nice</b>'))
from nltk.stem import PorterStemmer



porter = PorterStemmer()



def tokenizer(text):

    return text.split()



def tokenizer_porter(text):

    return [porter.stem(word) for word in text.split()]



print(tokenizer('Hi there, I am loving this, like with a lot of love'))

print(tokenizer_porter('Hi there, I am loving this, like with a lot of love'))
sns.countplot(data['gender'],label="Gender")
sns.barplot (x = 'gender', y = 'fav_number',data = data)
sns.barplot (x = 'gender', y = 'retweet_count',data = data)
male_top_sidebar_color = data[data['gender'] == 'male']['sidebar_color'].value_counts().head(7)

male_top_sidebar_color_idx = male_top_sidebar_color.index

male_top_color = male_top_sidebar_color_idx.values



male_top_color[2] = '000000'

print (male_top_color)

l = lambda x: '#'+x



sns.set_style("darkgrid", {"axes.facecolor": "#F5ABB5"})

sns.barplot (x = male_top_sidebar_color, y = male_top_color, palette=list(map(l, male_top_color)))
female_top_sidebar_color = data[data['gender'] == 'female']['sidebar_color'].value_counts().head(7)

female_top_sidebar_color_idx = female_top_sidebar_color.index

female_top_color = female_top_sidebar_color_idx.values



female_top_color[2] = '000000'

print (female_top_color)



l = lambda x: '#'+x



sns.set_style("darkgrid", {"axes.facecolor": "#F5ABB5"})

sns.barplot (x = female_top_sidebar_color, y = female_top_color, palette=list(map(l, female_top_color)))
male_top_link_color = data[data['gender'] == 'male']['link_color'].value_counts().head(7)

male_top_link_color_idx = male_top_link_color.index

male_top_color = male_top_link_color_idx.values

male_top_color[1] = '009999'

male_top_color[5] = '000000'

print(male_top_color)



l = lambda x: '#'+x



sns.set_style("whitegrid", {"axes.facecolor": "white"})

sns.barplot (x = male_top_link_color, y = male_top_link_color_idx, palette=list(map(l, male_top_color)))
female_top_link_color = data[data['gender'] == 'female']['link_color'].value_counts().head(7)

female_top_link_color_idx = female_top_link_color.index

female_top_color = female_top_link_color_idx.values



l = lambda x: '#'+x



sns.set_style("whitegrid", {"axes.facecolor": "white"})

sns.barplot (x = female_top_link_color, y = female_top_link_color_idx, palette=list(map(l, female_top_color)))
# Firstly, convert categorical labels into numerical ones

# Function for encoding categories

from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

y = encoder.fit_transform(data['gender'])





# split the dataset in train and test

X = data['text']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

#In the code line above, stratify will create a train set with the same class balance than the original set



X_train.head()
from sklearn.linear_model import LogisticRegression



tfidf = TfidfVectorizer(lowercase=False,

                        tokenizer=tokenizer_porter,

                        preprocessor=preprocessor)

clf = Pipeline([('vect', tfidf),

                ('clf', LogisticRegression(multi_class='ovr', random_state=0))])



clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,predictions))

print('Confusion matrix:\n',confusion_matrix(y_test,predictions))

print('Classification report:\n',classification_report(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier

# Plot the correlation between n_estimators and accuracy



# X_train_sample = X_train.head(5000) # this is series

# y_train_sample = y_train[:5000] # this is array



# print (X_train_sample.shape)

# print (y_train_sample.shape)



n = range (1,100,10) #step 10



results = []

for i in n:

    clf = Pipeline([('vect', tfidf),

                ('clf', RandomForestClassifier(n_estimators = i, random_state=0))])

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    results.append(accuracy_score(y_test, predictions))

plt.grid()

plt.scatter(n, results)
tfidf = TfidfVectorizer(lowercase=False,

                        tokenizer=tokenizer_porter,

                        preprocessor=preprocessor)

clf = Pipeline([('vect', tfidf),

                ('clf', RandomForestClassifier(n_estimators = 40, random_state=0))])



clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,predictions))

print('Confusion matrix:\n',confusion_matrix(y_test,predictions))

print('Classification report:\n',classification_report(y_test,predictions))
# the SVM model

from sklearn.svm import SVC



tfidf = TfidfVectorizer(lowercase=False,

                        tokenizer=tokenizer_porter,

                        preprocessor=preprocessor)

clf = Pipeline([('vect', tfidf),

                ('clf', SVC(kernel = 'linear'))])

clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,predictions))

print('Confusion matrix:\n',confusion_matrix(y_test,predictions))

print('Classification report:\n',classification_report(y_test,predictions))
data.head(3)
#Fill NaN with empty string

data.fillna("", inplace = True)



# Concatenate text with description, add white space between. 

# By using Series helper functions Series.str()

data['text_description'] = data['text'].str.cat(data['description'], sep=' ')



data['text_description'].isnull().value_counts() # Check if any null values, True if there is at least one.
# split the dataset in train and test

X = data['text_description']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

#In the code line above, stratify will create a train set with the same class balance than the original set



X_train.head()

X_train.isnull().values.any() # Check if any null values, True if there is at least one.
from sklearn.linear_model import LogisticRegression



tfidf = TfidfVectorizer(lowercase=False,

                        tokenizer=tokenizer_porter,

                        preprocessor=preprocessor)

clf = Pipeline([('vect', tfidf),

                ('clf', LogisticRegression(multi_class='ovr', random_state=0))])



clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,predictions))

print('Confusion matrix:\n',confusion_matrix(y_test,predictions))

print('Classification report:\n',classification_report(y_test,predictions))
# Plot the correlation between n_estimators and accuracy



# X_train_sample = X_train.head(5000) # this is series

# y_train_sample = y_train[:5000] # this is array



# print (X_train_sample.shape)

# print (y_train_sample.shape)



n = range (1,120,10) #step 10



results = []

for i in n:

    clf = Pipeline([('vect', tfidf),

                ('clf', RandomForestClassifier(n_estimators = i, random_state=0))])

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    results.append(accuracy_score(y_test, predictions))

plt.grid()    

plt.scatter(n, results)
from sklearn.ensemble import RandomForestClassifier



tfidf = TfidfVectorizer(lowercase=False,

                        tokenizer=tokenizer_porter,

                        preprocessor=preprocessor)

clf = Pipeline([('vect', tfidf),

                ('clf', RandomForestClassifier(n_estimators = 80, random_state=0))])



clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,predictions))

print('Confusion matrix:\n',confusion_matrix(y_test,predictions))

print('Classification report:\n',classification_report(y_test,predictions))
# the SVM model

from sklearn.svm import SVC



tfidf = TfidfVectorizer(lowercase=False,

                        tokenizer=tokenizer_porter,

                        preprocessor=preprocessor)

clf = Pipeline([('vect', tfidf),

                ('clf', SVC(kernel = 'linear'))])

clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,predictions))

print('Confusion matrix:\n',confusion_matrix(y_test,predictions))

print('Classification report:\n',classification_report(y_test,predictions))
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(multi_class='ovr', random_state=0)

clf2 = RandomForestClassifier(n_estimators = 80, random_state=0)

clf3 = SVC(kernel = 'linear',probability = True, random_state=0)



ensemble_clf = VotingClassifier(estimators=[

        ('lr', clf1), ('rf', clf2), ('svm', clf3)], voting='soft')



clf = Pipeline([('vect', tfidf),

                ('clf', ensemble_clf)])



clf.fit(X_train, y_train)



# ensemble_clf.fit(X_train, y_train)



predictions = clf.predict(X_test)

print('Accuracy:',accuracy_score(y_test,predictions))

print('Confusion matrix:\n',confusion_matrix(y_test,predictions))

print('Classification report:\n',classification_report(y_test,predictions))