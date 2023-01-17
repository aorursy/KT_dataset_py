# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import re
import string
import numpy as np 
import random
import pandas as pd 

%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import nltk
from nltk.corpus import stopwords
import nltk as nlp

from tqdm import tqdm
import os





import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

train_df.columns
train_df.head()
train_df.describe()
print(train_df.shape)
print(test_df.shape)
train_df.info()
train_df.dropna(inplace=True)
temp = train_df.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
temp.style.background_gradient(cmap='Reds')
def bar_plot(variable):
   
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
plt.figure(figsize=(10,8))
sns.countplot(x='sentiment',data=train_df)
fig = go.Figure(go.Funnelarea(
    text =temp.sentiment,
    values = temp.text,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()
lens = [len(x) for x in train_df.text]
plt.figure(figsize=(12, 5));

print ("Max length:", max(lens))
print ("Min length:", min(lens))
print ("Mean length:", np.mean(lens))

sns.distplot(lens);
plt.title('Text length distribution')
lens = [len(x) for x in train_df.selected_text]
plt.figure(figsize=(12, 5));
print ("Max length:", max(lens))
print ("Min length:", min(lens))
print ("Mean length:", np.mean(lens))
sns.distplot(lens);
plt.title('Text length distribution')
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
results_jaccard=[]

for ind,row in train_df.iterrows():
    sentence1 = row.text
    sentence2 = row.selected_text

    jaccard_score = jaccard(sentence1,sentence2)
    results_jaccard.append([sentence1,sentence2,jaccard_score])
jaccard = pd.DataFrame(results_jaccard,columns=["text","selected_text","jaccard_score"])
train_df = train_df.merge(jaccard,how='outer')
train_df['Num_words_ST'] = train_df['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
train_df['Num_word_text'] = train_df['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
train_df['difference_in_words'] = train_df['Num_word_text'] - train_df['Num_words_ST'] #Difference in Number of words text and Selected Text
train_df.head() 
#Duygulara gore Jaccard scrore ortalama degerleri
train_df.groupby('sentiment').mean()['jaccard_score']
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
train_df['text'] = train_df['text'].apply(lambda x:clean_text(x))
train_df['selected_text'] = train_df['selected_text'].apply(lambda x:clean_text(x))
train_df['sentiment'] = train_df['sentiment'].map({'positive': 1, 'negative': 2, 'neutral':0})

train_df.head()
def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]

#remove stopwords - selected text

train_df['selected_text_clear'] = train_df['selected_text'].apply(lambda x:str(x).split())

train_df['selected_text_clear'] = train_df['selected_text_clear'].apply(lambda x:remove_stopword(x))
#remove stopwords - text

train_df['text_clear'] = train_df['text'].apply(lambda x:str(x).split())

train_df['text_clear'] = train_df['text_clear'].apply(lambda x:remove_stopword(x))
lemma = nlp.WordNetLemmatizer()

def lemmatizate_word(x):
    return [lemma.lemmatize(word) for word in x]

train_df['selected_text_clear'] = train_df['selected_text_clear'].apply(lambda x:lemmatizate_word(x)) #selected text
train_df['text_clear'] = train_df['text_clear'].apply(lambda x:lemmatizate_word(x)) #text
def ngram(text):    
    return [(text[i],text[i+1]) for i in range(0,len(text)-1)]

train_df['ngram_text'] = train_df['text_clear'].apply(lambda x:str(x).split())
ngram_list = []

    
train_df['ngram_text'] = train_df['ngram_text'].apply(lambda ngram_list:ngram(ngram_list))

train_df.ngram_text
train_df.head()
top = Counter([item for sublist in train_df['selected_text_clear'] for item in sublist])
temp = pd.DataFrame(top.most_common(25))
temp = temp.iloc[1:,:]
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')

fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()
top = Counter([item for sublist in train_df['text_clear'] for item in sublist])
temp = pd.DataFrame(top.most_common(25))
temp = temp.iloc[1:,:]
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()
Positive_sent = train_df[(train_df['sentiment']== 1) ]

top = Counter([item for sublist in Positive_sent['text_clear'] for item in sublist])
temp_positive = pd.DataFrame(top.most_common(25))
temp_positive.columns = ['Common_words','count']
temp_positive.style.background_gradient(cmap='Greens')
Negative_sent = train_df[(train_df['sentiment']== 2) ]

top = Counter([item for sublist in Negative_sent['text_clear'] for item in sublist])
temp_negative = pd.DataFrame(top.most_common(25))
temp_negative = temp_negative.iloc[1:,:] #except 'im'
temp_negative.columns = ['Common_words','count']
temp_negative.style.background_gradient(cmap='Reds')
Neutral_sent = train_df[(train_df['sentiment']== 0) ]

top = Counter([item for sublist in Neutral_sent['text_clear'] for item in sublist])
temp_neutral = pd.DataFrame(top.most_common(25))
temp_neutral = temp_neutral.loc[1:,:] #except 'im'
temp_neutral.columns = ['Common_words','count']
temp_neutral.style.background_gradient(cmap='Greys')
raw_text = [word for word_list in train_df['selected_text_clear'] for word in word_list]

def words_unique(sentiment,numwords,raw_words):
    '''
    Input:
        segment - Segment category (ex. 'Neutral');
        numwords - how many specific words do you want to see in the final result; 
        raw_words - list  for item in train_data[train_data.segments == segments]['temp_list1']:
    Output: 
        dataframe giving information about the name of the specific ingredient and how many times it occurs in the chosen cuisine (in descending order based on their counts)..
    '''
    allother = []
    for item in train_df[(train_df.sentiment != sentiment)]['selected_text_clear']:
        for word in item:
            allother.append(word)
    allother = list(set(allother ))
    
    specificnonly = [x for x in raw_text if x not in allother]
    
    mycounter = Counter()
    
    for item in train_df[(train_df.sentiment == sentiment) ]['selected_text_clear']:
        for word in item:
            mycounter[word] += 1
    keep = list(specificnonly)
    
    for word in list(mycounter):
        if word not in keep:
            del mycounter[word]
    
    Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])
    
    return Unique_words
Unique_Positive= words_unique(1, 10, raw_text)
print("The top 10 unique words in Positive Tweets are:")
Unique_Positive.style.background_gradient(cmap='Greens')
Unique_Negative= words_unique(2, 10, raw_text)
print("The top 10 unique words in Negative Tweets are:")
Unique_Negative.style.background_gradient(cmap='Reds')
Unique_Neutral= words_unique(0, 10, raw_text)
print("The top 10 unique words in Neutral Tweets are:")
Unique_Neutral.style.background_gradient(cmap='Greys')

train_df2 = train_df[train_df['jaccard_score'] > 0.2]
train_df2.head()
temp = train_df2.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
temp.style.background_gradient(cmap='Reds')
selected_text_listt = []
for i in train_df2['selected_text_clear']:
    i = ' '.join(i)
    selected_text_listt.append(i)
    
from sklearn.feature_extraction.text import CountVectorizer 
max_features =500

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(selected_text_listt).toarray()  

y = train_df2.iloc[:,3:4].values     # sentiment
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# %% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

from sklearn.metrics import *
# Predicting the Test set results
y_pred = nb.predict(x_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

nb_02_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
# from sklearn.naive_bayes import MultinomialNB
# nb = MultinomialNB()
# nb.fit(x_train,y_train)

# from sklearn.metrics import *
# # Predicting the Test set results
# y_pred = nb.predict(x_test)


# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

lr_02_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

dt_02_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

rf_02_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

knn_02_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
# LOAD LIBRARIES
from sklearn.svm import SVC
clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

svm_02_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier().fit(x_train, y_train)

# Predicting the Test set results
y_pred = lgbm_model.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

lgbm_02_accuracy = accuracy_score(y_test, y_pred)
train_df8 = train_df[train_df['jaccard_score'] > 0.8]
selected_text_listt = []
for i in train_df8['selected_text_clear']:
    i = ' '.join(i)
    selected_text_listt.append(i)
    
from sklearn.feature_extraction.text import CountVectorizer 
max_features = 500

count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(selected_text_listt).toarray()  

y = train_df8.iloc[:,3:4].values     # sentiment
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# %% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

from sklearn.metrics import *
# Predicting the Test set results
y_pred = nb.predict(x_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

nb_08_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

lr_08_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

dt_08_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

rf_08_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

knn_08_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
# LOAD LIBRARIES
from sklearn.svm import SVC
clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

svm_08_accuracy = accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
            xticklabels='', yticklabels='')
plt.xlabel('true label')
plt.ylabel('predicted label');
from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier().fit(x_train, y_train)

# Predicting the Test set results
y_pred = lgbm_model.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

lgbm_08_accuracy = accuracy_score(y_test, y_pred)
df_accuracies = [lr_02_accuracy,nb_02_accuracy,dt_02_accuracy,rf_02_accuracy,knn_02_accuracy,svm_02_accuracy,lgbm_02_accuracy,lr_08_accuracy,nb_08_accuracy,dt_08_accuracy,rf_08_accuracy,knn_08_accuracy,svm_08_accuracy,lgbm_08_accuracy]

df_accuracies = pd.DataFrame(data = df_accuracies, index=range(len(df_accuracies)),columns=['accuracy'])
df_accuracies['model_name'] = ['logistic regression 02','naive bayes 02','desicion tree 02','random forest 02','knn 02','svm 02','lightgbm 02','logistic regression 08','naive bayes 08','desicion tree 08','random forest 08','knn 08','svm 08','lightgbm 08']
df_accuracies.head(12)

df_accuracies.plot(kind='bar',x='model_name',y='accuracy',figsize=(15,10))

