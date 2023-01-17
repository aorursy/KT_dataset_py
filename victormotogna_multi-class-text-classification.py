import pandas as pd

df = pd.read_csv("../input/us-consumer-finance-complaints/consumer_complaints.csv")

df.head()
df = df[pd.notnull(df['consumer_complaint_narrative'])]

df.info()
col = ['product', 'consumer_complaint_narrative']

df = df[col]



df.columns
df.columns = ['product', 'consumer_complaint_narrative']
df['category_id'] = df['product'].factorize()[0]

from io import StringIO

category_id_df = df[['product', 'category_id']].drop_duplicates().sort_values('category_id')

category_to_id = dict(category_id_df.values)

id_to_category = dict(category_id_df[['category_id', 'product']].values)
df.head()
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))

df.groupby('product').consumer_complaint_narrative.count().plot.bar(ylim=0)

plt.show()
df.consumer_complaint_narrative.size
df = df.head(n=15000)

df.consumer_complaint_narrative.size
from sklearn.feature_extraction.text import TfidfVectorizer



# sublinear_df is set to True (default) to use a logarithmic form for frequency

# min_df is the minimum number of documents a word must be present in to be kept

# norm is set to l2, to ensure all our feature vectors have a euclidian norm of 1

# encoding 'latin-1', the default for the alphabets of Western European languages

# ngram_range is set to (1, 2) to consider unigrams and bigrams

# stop_words is set to "english" to remove all common pronouns ("a", "the", ...) to reduce the number of noisy features

tfidf = TfidfVectorizer(min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')



features = tfidf.fit_transform(df.consumer_complaint_narrative).toarray()

labels = df.category_id

features.shape
from sklearn.feature_selection import chi2

import numpy as np



# using chi2 (computing chi-square stats between features & classes) to correlate each feature (unigram & bigram) to a class

N = 2

for product, category_id in sorted(category_to_id.items()):

  features_chi2 = chi2(features, labels == category_id)

  indices = np.argsort(features_chi2[0])

  feature_names = np.array(tfidf.get_feature_names())[indices]

  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]

  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

  print("# '{}':".format(product))

  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))

  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB



X_train, X_test, y_train, y_test = train_test_split(df['consumer_complaint_narrative'], df['product'], random_state = 0)

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



clf = MultinomialNB().fit(X_train_tfidf, y_train)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score



models = [

    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),

    LinearSVC(),

    MultinomialNB(),

    LogisticRegression(random_state=0),

]

CV = 5

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

for model in models:

  model_name = model.__class__.__name__

  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)

  for fold_idx, accuracy in enumerate(accuracies):

    entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns



sns.boxplot(x='model_name', y='accuracy', data=cv_df)

sns.stripplot(x='model_name', y='accuracy', data=cv_df, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)

plt.show()
cv_df.groupby('model_name').accuracy.mean()
from sklearn.model_selection import train_test_split



model = LinearSVC()



X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
category_id_df.columns = ['product_data', 'category_id']

category_id_df.product_data.values
from sklearn.metrics import confusion_matrix



conf_mat = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(conf_mat, annot=True, fmt='d',

            xticklabels=category_id_df.product_data.values, yticklabels=category_id_df.product_data.values)

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()