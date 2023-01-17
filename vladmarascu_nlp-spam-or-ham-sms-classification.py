import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import numpy as np
import matplotlib.pyplot as plt
import nltk
%matplotlib inline
messages = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
messages.head()
messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
messages.head()
messages.describe()
messages.groupby("label").describe()
messages['length']=messages['text'].apply(len)
messages.head()
messages.info()
messages.describe()
def print_validation_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy : "+ str(acc_sc))
def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    #fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  
                cmap="Blues", cbar=False)
    #  square=True,
    plt.ylabel('true label')
    plt.xlabel('predicted label')
messages["label"].value_counts().plot(kind = 'pie', figsize = (8, 8), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()
messages.hist(column='length',by='label',bins=50, figsize=(20,6))
ham  = messages[messages['label'] == 'ham'].copy()
spam = messages[messages['label'] == 'spam'].copy()

ham.head()
import wordcloud

def show_wordcloud(data, title):
    text = ' '.join(data['text'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color='white',
                    colormap='viridis', width=800, height=600).generate(text)
    
    plt.figure(figsize=(10,7), frameon=True)
    plt.imshow(fig_wordcloud, interpolation='bilinear')  
    plt.axis('off')
    plt.title(title, fontsize=20 )
    plt.show()
show_wordcloud(ham, "Ham top words")
show_wordcloud(spam, "Spam top words")
messages.head()
import string
from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words
def remove_punct_stop(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
messages['text'].apply(remove_punct_stop)
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['text'], messages['label'], test_size=0.3)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

pipeline_NB = Pipeline([
    ('bow', CountVectorizer(analyzer=remove_punct_stop)),  # strings to token integer counts | use the DATA CLEANING FUNCTION PREDEFINED
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ CHOSEN ML MODEL (CAN BE CHANGED)
])
pipeline_NB.fit(msg_train,label_train)
predictions = pipeline_NB.predict(msg_test)
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, fbeta_score
print(fbeta_score(predictions,label_test, beta=0.5, pos_label='ham'))
plot_confusion_matrix(predictions,label_test)
print_validation_report(predictions,label_test)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

pipeline_KNN = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punct_stop) ),
                   ('tfidf'   , TfidfTransformer()),
                   ('clf_KNN' , KNeighborsClassifier() )
                    ])

parameters_KNN = {'clf_KNN__n_neighbors': (8,15,20), }

grid_KNN = GridSearchCV(pipeline_KNN, parameters_KNN, cv=5,refit=True, verbose=3)

grid_KNN.fit(msg_train,label_train)
grid_KNN.best_params_
grid_KNN.best_score_
predictions = grid_KNN.predict(msg_test)
print(fbeta_score(predictions,label_test, beta=0.5, pos_label='ham'))
plot_confusion_matrix(predictions,label_test)
print_validation_report(predictions,label_test)
from sklearn.svm import SVC

pipeline_SVC = Pipeline([ ('bow'  , CountVectorizer(analyzer = remove_punct_stop) ),
                   ('tfidf'   , TfidfTransformer()),
                   ('clf_SVC' , SVC(gamma='auto', C=1000)),
                    ])


parameters_SVC = dict(tfidf=[None, TfidfTransformer()], clf_SVC__C=[500, 1000,1500])

grid_SVC = GridSearchCV(pipeline_SVC, parameters_SVC, cv=5, refit=True, verbose=1)

grid_SVC.fit(msg_train, label_train)
grid_SVC.best_params_
grid_SVC.best_estimator_
grid_SVC.best_score_
predictions = grid_SVC.predict(msg_test)
print(fbeta_score(predictions,label_test, beta=0.5, pos_label='ham'))
plot_confusion_matrix(predictions,label_test)
print_validation_report(predictions,label_test)