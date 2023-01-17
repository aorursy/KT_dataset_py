#Importing the required libraries to read,visualize and model the givn dataset files

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import missingno as msno 

import warnings

warnings.filterwarnings("ignore")

import re

import re

import string

import nltk

from nltk.corpus import stopwords

from sklearn import model_selection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import RepeatedStratifiedKFold

import nltk

from nltk.tokenize import word_tokenize,RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from platform import python_version

print (python_version())
# Read the dataset csv files and create pandas datframes

train_df=pd.read_csv("../input/nlp-getting-started/train.csv")

test_df=pd.read_csv("../input/nlp-getting-started/test.csv")

print("Train and Test data sets are imported successfully")

#X=train_df.text

#y=train_df.target
# Define a function to explore the train and test dataframes

def explore_data(df):

    

    '''Input- df= pandas dataframes to be explored

       Output- print shape, info and first 5 records of the dataframe 

    '''

    

    print("-"*50)

    print('Shape of the dataframe:',df.shape)

    print("Number of records in train data set:",df.shape[0])

    print("Information of the dataset:")

    df.info()

    print("-"*50)

    print("First 5 records of the dataset:")

    return df.head()

    print("-"*50)
# Lets use explore_data() function to explore train data

explore_data(train_df)
# Lets use explore_data() function to explore test data

explore_data(test_df)
#Calculate count and percentage of missing values in the dataframe



def missing_values(df):

    

    '''Input- df=pandas dataframe

       Output- print missing records count and % of the input dataframe and visualize using MSNO

    '''

    

    print("Number of records with missing location:",df.location.isnull().sum())

    print("Number of records with missing keywords:",df.keyword.isnull().sum())

    print('{}% of location values are missing from Total Number of Records.'.format(round((df.location.isnull().sum())/(df.shape[0])*100),2))

    print('{}% of keywords values are missing from Total Number of Records.'.format(round((df.keyword.isnull().sum())/(df.shape[0])*100),2))

    msno.matrix(df);
# Lets use missing_values function to explore train dataset

missing_values(train_df)
# Lets use missing_values function to explore train dataset

missing_values(test_df);
#Lets visulaize dataframe features using charts

def feature_viz(df,feature):

    

    '''Input- df=pandas dataframe

              feature= column to be charted

       Output- bar and scatter chart using plotly       

    

    '''

    #Visualize the feature

    if feature=='target':

        sns.countplot(feature, data=df)

        print('Target of 0 is {} % of total'.format(round(df[feature].value_counts()[0]/len(df[feature])*100)))

        print('Target of 1 is {} % of total'.format(round(df[feature].value_counts()[1]/len(df[feature])*100)))

    else:

        #Distinct keywords in train dataset

        feat=df[feature].value_counts()

        print(feat.head())

        fig = px.scatter(feat, x=feat.values, y=feat.index,size=feat.values)

        fig.show()
#Lets use feature_viz function to create charts for 'target' column

feature_viz(train_df,'target')
#Lets use feature_viz function to create charts for 'keyword' column

feature_viz(train_df,'keyword')
train_df.loc[train_df['text'].str.contains('disaster', na=False, case=False)].target.value_counts()
#Lets use feature_viz function to create charts for 'location' column

feature_viz(train_df,'location')
# Replacing the ambigious locations name with standard names and create a dictionary object



loc_dict={'United States':'USA','New York':'USA',"London":'UK',"Los Angeles, CA":'USA',"Washington, D.C.":'USA',

          "California":'USA',"Chicago, IL":'USA',"Chicago":'USA',"New York, NY":'USA',"California, USA":'USA',

          "FLorida":'USA',"Nigeria":'Africa',"Kenya":'Africa',"Everywhere":'Worldwide',"San Francisco":'USA',

          "Florida":'USA',"United Kingdom":'UK',"Los Angeles":'USA',"Toronto":'Canada',"San Francisco, CA":'USA',

          "NYC":'USA',"Seattle":'USA',"Earth":'Worldwide',"Ireland":'UK',"London, England":'UK',"New York City":'USA',

          "Texas":'USA',"London, UK":'UK',"Atlanta, GA":'USA',"Mumbai":"India"}



train_df['location'].replace(loc_dict,inplace=True)



#Create barchart for top 10 locations using seaborn

sns.barplot(y=train_df['location'].value_counts()[:10].index,x=train_df['location'].value_counts()[:10],

            orient='h');
# Drop the column 'location' from the training dataset

train_df=train_df.drop(['location'],axis=1)
# Lets find the length of the 'text' for each row and add a new cloumn to train dataframe 

train_df['text_length'] = train_df['text'].apply(lambda x : len(x))

train_df.head(4)
train_df.text_length.describe()
#Create distribution chart to visualize text length distribution

ax=sns.distplot(train_df['text_length']).set_title('Distribution of the tweet lengths');

plt.grid(True)
#Create visualization of the distribution of text length in comparision to target feature

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True,figsize=(10,6))

sns.distplot(train_df[(train_df['target'] == 1)]['text_length'], ax=ax1, kde=False, color='green',label='Disater Tweets')

sns.distplot(train_df[(train_df['target'] == 0)]['text_length'],ax=ax2, kde=False, color='red',label='Non-Disater Tweets');

f.suptitle('Tweet length distribution')

f.legend(loc='upper right')

ax1.grid()

ax2.grid()

plt.show()
#Create visualization of the distribution of the word counts in comparision to target feature

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

dis_tweet=train_df[train_df['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(dis_tweet,color='blue')

ax1.set_title('Disaster tweets')

ax1.grid()

nondis_tweet=train_df[train_df['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(nondis_tweet,color='red')

ax2.set_title('Non-disaster tweets')

ax2.grid()

fig.suptitle('Words in a tweet')

plt.show()
# A disaster tweet exmaple

train_df[train_df['target']==1]['text'][10:20]
#A non-disaster tweet example

train_df[train_df['target']==0]['text'][10:20]
#Lets have a quick look of the text data

train_df['text'][:5]
# Create a function to clean the text



def clean_text(text):



    '''

    Input- 'text' to be cleaned

       

       Output- Convert input 'text' to lowercase,remove square brackets,links,punctuation

       and words containing numbers. Return clean text.

    

    '''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
# Lets apply the clean_text function to both test and training datasets copies

train_df1=train_df.copy()

test_df1=test_df.copy()

train_df1['text'] = train_df1['text'].apply(lambda x: clean_text(x))

test_df1['text'] = test_df1['text'].apply(lambda x: clean_text(x))
#Lets look cleaned text data

def text_after_preprocess(before_text,after_text):

    

    '''

    Input- before_text=text column before cleanup

              after_text= text column after cleanup

       Output- print before and after text to compare how it looks after cleanup

       

    '''

    print('-'*60)

    print('Text before cleanup')

    print('-'*60)

    print(before_text.head(5))

    print('-'*60)

    print('Text after cleanup')

    print('-'*60)

    print(after_text.head(5))
text_after_preprocess(train_df.text,train_df1.text)
text_after_preprocess(test_df.text,test_df1.text)
test_df.text[1]
# Example how tokenization of text works

text = "Heard about #earthquake is different cities, stay safe everyone."

tokenizer1 = nltk.tokenize.WhitespaceTokenizer()

tokenizer2 = nltk.tokenize.TreebankWordTokenizer()

tokenizer3 = nltk.tokenize.WordPunctTokenizer()

tokenizer4 = nltk.tokenize.RegexpTokenizer(r'\w+')

print("-"*100)

print("Example Text: ",text)

print("-"*100)

print("Tokenization by whitespace:- ",tokenizer1.tokenize(text))

print("Tokenization by words using Treebank Word Tokenizer:- ",tokenizer2.tokenize(text))

print("Tokenization by punctuation:- ",tokenizer3.tokenize(text))

print("Tokenization by regular expression:- ",tokenizer4.tokenize(text))
# Lets Tokenize the training and the test dataset copies with RegEx tokenizer

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train_df1['text'] = train_df1['text'].apply(lambda x: tokenizer.tokenize(x))

test_df1['text'] = test_df1['text'].apply(lambda x: tokenizer.tokenize(x))
#Lets checy tokenized text

train_df1['text'].head()
test_df1['text'].head()
#Create a funtion to remove stopwords

def remove_stopwords(text):

    

    """

    Input- text=text from which english stopwprds will be removed

    Output- return text without english stopwords 

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words
train_df1['text'] = train_df1['text'].apply(lambda x : remove_stopwords(x))

test_df1['text'] = test_df1['text'].apply(lambda x : remove_stopwords(x))
train_df1.text.head()
test_df1.text.head()
# Stemming and Lemmatization examples

text =  "ran deduced dogs talking studies"



tokenizer = nltk.tokenize.TreebankWordTokenizer()

tokens = tokenizer.tokenize(text)



# Stemmer

stemmer = nltk.stem.PorterStemmer()

print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))



# Lemmatizer

lemmatizer=nltk.stem.WordNetLemmatizer()

print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))
# Lets combine text after processing it

def combine_text(text):

    

    '''

    Input-text= list cleand and tokenized text

    Output- Takes a list of text and returns combined one large chunk of text.

    

    '''

    all_text = ' '.join(text)

    return all_text
train_df1['text'] = train_df1['text'].apply(lambda x : combine_text(x))

test_df1['text'] = test_df1['text'].apply(lambda x : combine_text(x))
train_df1.head()
# Create a function to pre-process the tweets

def pre_process_text_combined(text):

    

    """

    Input- text= text to be pre-processed

    

    Oputput- return cleaned and combined text to be vectrorized for Machine learning.



    """

    #Initiate a tokenizer

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    # Clean the text using clean_text function

    cleaned_txt = clean_text(text)

    tokenized_text = tokenizer.tokenize(cleaned_txt)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    combined_text = ' '.join(remove_stopwords)

    return  combined_text


# Create a function to pre-process the tweets

def pre_process_text(text):

    """

    Input- text= text to be pre-processed

    

    Oputput- return cleaned text to be vectrorized for Machine learning.



    """

    #Initiate a tokenizer

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    # Clean the text using clean_text function

    cleaned_txt = clean_text(text)

    tokenized_text = tokenizer.tokenize(cleaned_txt)

    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]

    return remove_stopwords
# Text before pre-processing

train_df.text.head()
# Lets pre-process train data text

train_df2=train_df.copy()

train_df2['text'] = train_df2['text'].apply(lambda x : pre_process_text_combined(x))
# Text after pre-processing the text column

train_df2.head()
# Lets pre-process test data text

test_df2=test_df.copy()

test_df2['text'] = test_df2['text'].apply(lambda x : pre_process_text_combined(x))
# Text after pre-processing the text column

test_df2.head(10)
# Lets pre-process train data text

train_df3=train_df.copy()

train_df3['text'] = train_df3['text'].apply(lambda x : pre_process_text(x))
train_df3.head()
# Lets pre-process train data text

test_df3=test_df.copy()

test_df3['text'] = test_df3['text'].apply(lambda x : pre_process_text(x))
test_df3.head()
#Lets have a quick look of the tweets in wordcloud

from wordcloud import WordCloud

fig, ax = plt.subplots(figsize=[10, 6])

wordcloud = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(train_df2.text))

ax.imshow(wordcloud)

ax.axis('off')

ax.set_title('Disaster Tweets',fontsize=40);
# Vectorize the text using CountVectorizer

count_vectorizer = CountVectorizer()

train_cv = count_vectorizer.fit_transform(train_df2['text'])

test_cv = count_vectorizer.transform(test_df2["text"])



## Keeping only non-zero elements to preserve space 

print(train_cv[0].todense())
# Vectorize the text using TFIDF

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

train_tf = tfidf.fit_transform(train_df2['text'])

test_tf = tfidf.transform(test_df2["text"])
#Split the CountVector vectorized data into train and test datasets for model training and testing

X_train_cv, X_test_cv, y_train_cv, y_test_cv =train_test_split(train_cv,train_df.target,test_size=0.2,random_state=2020)
#Define a function to fit and predict on training and test data sets

def fit_and_predict(model,X_train,y_train,X_test,y_test):

    

    '''Input- model=model to be trained

              X_train, y_train= traing data set

              X_test,  y_test = testing data set

       Output- Print accuracy of model for training and test data sets   

    '''

    

    # Fitting a simple Logistic Regression on Counts

    clf = model

    clf.fit(X_train, y_train)

    predictions=clf.predict(X_test)

    confusion_matrix(y_test,predictions)

    print(classification_report(y_test,predictions))

    print('-'*50)

    print("{}" .format(model))

    print('-'*50)

    print('Accuracy of classifier on training set:{}%'.format(round(clf.score(X_train, y_train)*100)))

    print('-'*50)

    print('Accuracy of classifier on test set:{}%' .format(round(accuracy_score(y_test,predictions)*100)))

    print('-'*50)
# Create a list of the regression models to be used

models=[LogisticRegression(C=1.0),SVC(),MultinomialNB(),DecisionTreeClassifier(),

        KNeighborsClassifier(n_neighbors=5),RandomForestClassifier()]
# Loop through the list of models and use 'fit_and_predict()' function to trian and make predictions

for model in models:

    fit_and_predict(model,X_train_cv, y_train_cv,X_test_cv,y_test_cv)
# Split the TFDIF vectorized data into train and test datasets for model training and testing

X_train_tf, X_test_tf, y_train_tf, y_test_tf =train_test_split(train_tf,train_df.target,test_size=0.2,random_state=2020)
# Loop through the list of models and use 'fit_and_predict()' function to train and make predictions on the TFDIF vectororized data

for model in models:

    fit_and_predict(model,X_train_tf, y_train_tf,X_test_tf,y_test_tf)
# Printing model performance results.

results_dict={'Classifier':['Logistic regression', 'SVC', 'MultinomialNB', 'DecisionTreeClassifier',

                            'KNeighborsClassifier','RandomForestClassifier'],

              'F1-Score':[0.81, 0.40, .80, .75,0.65,0.76],'Accuracy':['81%', '56%', '80%','75%','69%','77%']} 

results=pd.DataFrame(results_dict)

results
# Fitting 'LogisticRegression()' with CountVectorizer() fit dataset

clf_logreg = LogisticRegression(C=1.0)

clf_logreg.fit(X_train_cv, y_train_cv)

pred=clf_logreg.predict(X_test_cv)

confusion_matrix(y_test_cv,pred)

print(classification_report(y_test_cv,pred))

print('Accuracy of classifier on training set:{}%'.format(round(clf_logreg.score(X_train_cv, y_train_cv)*100)))

print('Accuracy of classifier on test set:{}%' .format(round(accuracy_score(y_test_cv,pred)*100)))
clf_logreg
# Create the list of various hyper parameters to try

solvers = ['newton-cg', 'lbfgs', 'liblinear']

penalty = ['l2']

c_values = [100, 10, 1.0, 0.1, 0.01]

logreg= LogisticRegression()



# Define and fit grid search

grid = dict(solver=solvers,penalty=penalty,C=c_values)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

grid_search = GridSearchCV(estimator=logreg, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)

grid_result = grid_search.fit(X_train_cv, y_train_cv)



# Summarize and print results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Lets apply pre-processing function to clean and pre-process text data before vectorizing

test_df_final=test_df.copy()

test_df_final['text'] = test_df_final['text'].apply(lambda x : pre_process_text_combined(x))
# Lets fit the test data with Countvectorizer() method to vectroize the data

test_vector = count_vectorizer.transform(test_df_final["text"])
# Define a function to generate predictions and store in a.csv file for submission on Kaggle

def submission(sub_file,model,test_vector):

    

    '''Input- sub_file=Location of the file submission file

              model=final fit model to be used for predictions

              test_vector=pre-processed and vectorized test dataset

       Output- submission file in .csv format with predictions       

    

    '''

    sub_df = pd.read_csv(sub_file)

    sub_df["target"] = model.predict(test_vector)

    sub_df.to_csv("submission.csv", index=False)
# Use Submission() function to generate submission file for posting on Kaggle

sub_file = "../input/nlp-getting-started/sample_submission.csv"

test_vector=test_vector

submission(sub_file,clf_logreg,test_vector)