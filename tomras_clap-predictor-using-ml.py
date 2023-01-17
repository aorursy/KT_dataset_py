import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import string
import re
%matplotlib inline
df = pd.read_csv('../input/articles.csv')
df.dtypes
df.head(5)
df['claps'] = df['claps'].apply(lambda x: int(float(x[:-1]) * 1000) if x[-1] == 'K' else int(x))
df.dtypes
df.isnull().any()
df['title_len'] = df['title'].str.len()
df['text_len'] = df['text'].str.len()

df['title'] = df['title'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: x.lower())
df['author'] = df['author'].apply(lambda x: x.lower())

df['title_clean'] = df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
df['text_clean'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))

df['title_clean'] = df['title_clean'].apply(lambda x: re.sub('[' + string.punctuation + '—]', '', x))
df['text_clean'] = df['text_clean'].apply(lambda x: re.sub('[' + string.punctuation + '—]', '', x))

df['title_clean'] = df['title_clean'].apply(lambda x: x.translate(str.maketrans('', '', string.digits)))
df['text_clean'] = df['text_clean'].apply(lambda x: x.translate(str.maketrans('', '', string.digits)))

df['title_clean'] = df['title_clean'].apply(lambda x: re.sub(' +', ' ', x))
df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(' +', ' ', x))

df['title_clean_len'] = df['title_clean'].str.len()
df['text_clean_len'] = df['text_clean'].str.len()

df['full_text'] = df['author'] + ' ' + df['title_clean'] + ' ' + df['text_clean']

df.head(10)
df = df.drop('link', axis=1)
df = df.drop('text', axis=1)
df = df.drop('title', axis=1)
df = df.drop('title_clean', axis=1)
df = df.drop('text_clean', axis=1)
df = df.drop('author', axis=1)

df = df.drop_duplicates()

df.describe(include='all')
df.boxplot(column=['claps', 
                   'text_len', 
                   'text_clean_len'])
plt.show()
df.boxplot(column=['reading_time', 
                   'title_len', 
                   'title_clean_len'])
plt.show()
sns.pairplot(df[['claps', 
                 'reading_time',
                 'title_len', 
                 'title_clean_len',
                 'text_len', 
                 'text_clean_len']], kind='reg')
plt.show()
df[df['claps'] > 18000]
df[df['text_len'] > 28000]
df[df['reading_time'] > 22]
vectorizer = TfidfVectorizer(max_features=None)
full_text_features = vectorizer.fit_transform(df['full_text'])
full_text_features.shape
scaler = StandardScaler()
num_features = scaler.fit_transform(df[['reading_time', 
                                        'title_len',
                                        'text_len',
                                        'title_clean_len',
                                        'text_clean_len']])
num_features.shape
full_text_features = np.concatenate([full_text_features.toarray(), num_features], axis=1)
full_text_features.shape
X_train, X_test, y_train, y_test = train_test_split(full_text_features, df[['claps']].values, test_size=0.3)
X_train.shape
y_test.shape
reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred.shape
r2_score(y_test, y_pred)
df[['claps']].hist()
plt.show()
df['claps_categorical'] = df['claps'].apply(lambda x: 'rising star' if x >= 0 and x <= 10000 else 'star' if x >= 10001 and x <= 20000 else 'super star')
df[['claps', 'claps_categorical']].head(15)
X_train, X_test, y_train, y_test = train_test_split(full_text_features, df[['claps_categorical']].values, test_size=0.3)
X_train.shape
y_test.shape
clf = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred, labels=['rising star', 'star', 'super star'])