#Imports:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#Code:

# reading the csv file into pandas dataframes

df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df.head()
df.shape
df.isnull().sum()
df.drop(['keyword','location'], axis=1, inplace=True)

df.head()
df['target'].value_counts()
#calculating basline accuracy

df['target'].value_counts(normalize=True)
from wordcloud import WordCloud



# word cloud for words in dataframe

text=" ".join(post for post in df.text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words  \n\n',fontsize=18)

plt.axis("off");

# Import Tokenizer

from nltk.tokenize import RegexpTokenizer



df.loc[:,'text'] = df.text.apply(lambda x : str.lower(x))#changing the contents of text to lowercase



df['text']=df['text'].str.replace('http://t.co/+\w+', '',regex = True) #removing hyper link 

df['text']=df['text'].str.replace('û.*.*|Û.*.*', '',regex = True) # removing latin characters 



tokenizer = RegexpTokenizer(r'(\w+)') #only words 

df['tokens'] = df['text'].map(tokenizer.tokenize) # applying tokenization to entire column
from nltk.corpus import stopwords



#assigning stopwords to a variable

stop = stopwords.words("english")

# adding this stop word to list of stopwords as it appears on frequently occuring word

item=['amp','co']

stop.extend(item)

#removing stopwords from tokens

df['tokens']=df['tokens'].apply(lambda x: [item for item in x if item not in stop])
# Importing lemmatizer 

from nltk.stem import WordNetLemmatizer





# Instantiating lemmatizer 

lemmatizer = WordNetLemmatizer()

lemmatize_words=[]

for i in range (len(df['tokens'])):# loop which runs through entire df

    word=''

    for j in range(len(df['tokens'][i])): # loop which runs through each row

        lemm_word=lemmatizer.lemmatize(df['tokens'][i][j])#lemmatize

        

        word=word + ' '+lemm_word # joining tokens into sentence    

    lemmatize_words.append(word) # store in list

#creating a new column to store the result

df['lemmatized']=lemmatize_words 
#displaying first 5 rows of dataframe

df.head()
# word cloud for words related to Disaster 

text=" ".join(post for post in df[df['target']==1].lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
# word cloud for words related to No Disaster 

text=" ".join(post for post in df[df['target']==0].lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words related to No Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
from sklearn.model_selection import train_test_split



#defining X and y for the model

X = df['lemmatized']

y = df['target']



# Spliting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)



from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression



# this pipeline consists of two stages:

pipe = Pipeline([ 

    ('tfidf', TfidfVectorizer()),  # 1.Instantiating TfidVectorizer

    ('lr', LogisticRegression()) # 2.Instantiating logistic regression model

])



pipe.fit(X_train,y_train) #fitting train data



predictions=pipe.predict(X_test) #predicting on test data


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score #imports





print(f'Accuracy score:{accuracy_score(y_test,predictions)}')# printing accuracy score



print('\nConfusion Matrix')

print(confusion_matrix(y_test,predictions)) #printing confusion matrix



tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()#interpreting confusion matrix



print(f"\nMisclassified: {fp+fn:{11}}")

print(f"Correctly classified: {tp+tn}\n\n")





print('Classification Report')

print(classification_report(y_test,predictions)) # Printing a classification report

# Importing model

from sklearn.naive_bayes import MultinomialNB





pipe = Pipeline([

    ('tfidf', TfidfVectorizer()), # 1.Instantiating TfidVectorizer

    ('nb', MultinomialNB()) # 2.Instantiating logistic regression model

])



pipe.fit(X_train,y_train) # fitting train data





predictions=pipe.predict(X_test) # predicting on test data
print(f'Accuracy score:{accuracy_score(y_test,predictions)}')# printing accuracy score



print('\nConfusion Matrix')

print(confusion_matrix(y_test,predictions)) #printing confusion matrix



tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()#interpreting confusion matrix



print(f"\nMisclassified: {fp+fn:{11}}")

print(f"Correctly classified: {tp+tn}\n\n")





print('Classification Report')

print(classification_report(y_test,predictions)) # Printing a classification report

#reading the test data

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test.head()
# word cloud for Frequntly occuring words in test dataframe

text=" ".join(post for post in df.text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words in test dataframe \n\n',fontsize=18)

plt.axis("off")

plt.show()
test.loc[:,'text'] = test.text.apply(lambda x : str.lower(x))#changing the contents of text to lowercase



test['text']=test['text'].str.replace('http://t.co/+\w+', '',regex = True) #removing hyper link 

test['text']=test['text'].str.replace('û.*.*|Û.*.*', '',regex = True) # removing latin characters 



tokenizer = RegexpTokenizer(r'(\w+)') #only words 

test['tokens'] = test['text'].map(tokenizer.tokenize) # applying tokenization to entire column
#assigning stopwords to a variable

stop = stopwords.words("english")

# adding this stop word to list of stopwords as it appears on frequently occuring word

item=['amp','co']

stop.extend(item)

#removing stopwords from tokens

test['tokens']=test['tokens'].apply(lambda x: [item for item in x if item not in stop])
# Instantiating lemmatizer 

lemmatizer = WordNetLemmatizer()

lemmatize_words=[]

for i in range (len(test['tokens'])):# loop which runs through entire df

    word=''

    for j in range(len(test['tokens'][i])): # loop which runs through each row

        lemm_word=lemmatizer.lemmatize(test['tokens'][i][j])#lemmatize

        

        word=word + ' '+lemm_word # joining tokens into sentence    

    lemmatize_words.append(word) # store in list

#creating a new column to store the result

test['lemmatized']=lemmatize_words 
#displaying first 5 rows of dataframe

test.head()
predictions_kaggle = pipe.predict(test['lemmatized'])
# Creating an empty data frame

submission_kaggle = pd.DataFrame()
# Assigning values to the data frame-submission_kaggle

submission_kaggle['Id'] = test.id

submission_kaggle['target'] = predictions_kaggle
# Head of submission_kaggle

submission_kaggle.head()
# saving data as  final_kaggle.csv

submission_kaggle.loc[ :].to_csv('final_kaggle.csv',index=False)