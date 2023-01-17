

import warnings

warnings.filterwarnings("ignore") # Shhhh

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.pipeline import Pipeline

from sklearn.ensemble import BaggingClassifier

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords

from sklearn.tree import DecisionTreeClassifier

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

from nltk.stem import PorterStemmer

import nltk

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/superbowlads/superbowl-ads.csv',error_bad_lines=False)

df.head(10)
df.isnull().sum()



df = df.dropna()

df['Product Type'].value_counts()

df['Product Type'] = [1 if i == 'Film' else 0 for i in df['Product Type']]
plt.figure(figsize = (12,5))

sns.countplot(x='Product Type',data=df,order= df['Product Type'].value_counts().iloc[:10].index)
df['Company'] = df['Product/Title'].apply(lambda x: x.split("\"")[0])

df
df['Company'].value_counts(normalize = True).head(10)

df['Plot/Notes'] = df['Plot/Notes'].fillna('missing')
df
df.isnull().sum()
ps = PorterStemmer()

df['tokenized_notes'] = [nltk.word_tokenize(doc) for doc in df["Plot/Notes"]]



df['stemmed_notes'] = [[ps.stem(word) for word in row] for row in df['tokenized_notes']]

df['lower_notes'] = [[word.lower() for word in row] for row in df['stemmed_notes']]

df['newtext_notes'] = [" ".join(row) for row in df['lower_notes']]
df.head()
cvec = CountVectorizer(stop_words='english', ngram_range= (1,2))

##fit cvec with title



cvec.fit(df['newtext_notes'])

len_features = len(cvec.get_feature_names())

print(len_features)
sports_df_cv = pd.DataFrame(cvec.transform(df['newtext_notes']).todense(),columns=cvec.get_feature_names())

highest_sports_cv = sports_df_cv.sum(axis=0)

df_cvec = highest_sports_cv.sort_values(ascending = False).head(20)

df_cvec = pd.DataFrame(df_cvec, columns = ['Count_Vectorizer(units)'])

sports_top = highest_sports_cv.to_frame(name='Count_Vectorizer(units)')

sports_top['Word'] = sports_top.index

sports_top.reset_index(drop=True, inplace=True)

cols = ['Word','Count_Vectorizer(units)']

sports_top = sports_top[cols]

sports_top.sort_values(by='Count_Vectorizer(units)',ascending=False, inplace=True)

sports_top.head(10)
##use seaborn package

plt.figure(figsize=(20,10))

plt.title('Count Vectorizer: Sports Reddit Top 20 Words',fontsize=25)

sns.set_style("darkgrid")

sns.barplot(data=sports_top.head(10),x='Count_Vectorizer(units)',y='Word',orient='h')

plt.xlabel('Count Vectorizer Frequency (Units)',fontsize=20)

plt.ylabel('Word(Text)',fontsize=20)

plt.tick_params(labelsize=15)
X = df["newtext_notes"]

y = df["Product Type"]

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle = True)
X_train
tvec = TfidfVectorizer(ngram_range=(1,2), stop_words = "english")

X_train_counts = tvec.fit_transform(X_train)

X_test_counts = tvec.transform(X_test)

print("Number of features:",X_train_counts.shape[1])
df['Product Type'].value_counts(normalize = True).head(10)
log_reg = LogisticRegression(random_state = 42 )

log_reg.fit(X_train_counts, y_train)

print("Train data CV score:", cross_val_score(log_reg, X_train_counts, y_train, cv= 5))

print("Test data score:", log_reg.score(X_test_counts, y_test))
pipe = Pipeline(steps = [('vectorizer',TfidfVectorizer(lowercase = False)),     # first tuple is for first step: vectorizer

                         ('model', LogisticRegression(solver = 'liblinear'))

                         

                        ])    



# Construct Grid Parameters

hyperparams = {'vectorizer__ngram_range': [(1,1), (1,2), (2,2)],

               'vectorizer__stop_words': ['english'],

                                                        # use a single value that isn't built into

                                                        # the defaults (otw: stopwords left in)

               'model__penalty': ['l1', 'l2'],

               'model__C': [3, 10, 1000]

                

              }



 # Perform Grid Search

gs = GridSearchCV(pipe, # pipeline object replaces what we usually had as empty model class

                 param_grid=hyperparams,

                 cv = 3,

                 scoring = 'accuracy')
results_log = gs.fit(X_train, y_train)
train_score_log = results_log.best_score_

print('Best TRAIN accuracy: {:.4f}'.format(train_score_log))

test_score_log = results_log.score(X_test, y_test)

print('Best TEST set accuracy: {:.4f}'.format(test_score_log))
def pretty_confusion_matrix(y_true, y_pred):

    # handling data

    cm = confusion_matrix(y_true, y_pred)

    labels = ['Other(0)', 'Film(1)']

    sns.set(font_scale=3)

    plt.figure(figsize=(14,8))

    sns.heatmap(cm, annot=True, fmt='g', cmap="YlGnBu",xticklabels=labels, yticklabels=labels)

    plt.title("Confusion Matrix")

    plt.xlabel('Predicted Class')

    plt.ylabel('True Class')
pretty_confusion_matrix(y_test, results_log.predict(X_test))