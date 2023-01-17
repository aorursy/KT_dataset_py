import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
    
# NLTK modules
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer

import re

from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, remove_stopwords, strip_numeric, stem_text
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, SVC

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_df = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')
test_df = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')

submission_df = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv')
print(train_df.isnull().sum())
print(train_df.columns)
# Converting binary column to category
target_cols = ['Computer Science', 'Physics', 'Mathematics','Statistics', 'Quantitative Biology', 'Quantitative Finance']
y_data = train_df[target_cols]

# Plot category data
plt.figure(figsize=(10,6))
y_data.sum(axis=0).plot.bar()
plt.show()

# Stemmer object
porter = PorterStemmer()
wnl = WordNetLemmatizer()

class DataPreprocess:
    
    def __init__(self):
        self.filters = [strip_tags,
                       strip_numeric,
                       strip_punctuation,
                       lambda x: x.lower(),
                       lambda x: re.sub(r'\s+\w{1}\s+', '', x),
                       remove_stopwords]
    def __call__(self, doc):
        clean_words = self.__apply_filter(doc)
        return clean_words
    
    def __apply_filter(self, doc):
        try:
            cleanse_words = set(preprocess_string(doc, self.filters))
#             filtered_words = set(wnl.lemmatize(w) if w.endswith('e') else porter.stem(w) for w in cleanse_words)
            filtered_words = set(wnl.lemmatize(word, 'v') for word in cleanse_words)
            return ' '.join(filtered_words)
        except TypeError as te:
            raise(TypeError("Not a valid data {}".format(te)))
train_df['train_or_test'] = 0
test_df['train_or_test'] = 1

feature_col = ['ID', 'TITLE', 'ABSTRACT', 'train_or_test']

# Concat train and test data
combined_set = pd.concat([train_df[feature_col], test_df[feature_col]])

# Combine the Title and Abstract data
combined_set['TEXT'] = combined_set['TITLE'] + combined_set['ABSTRACT']

# Drop unwanted columns
combined_set = combined_set.drop(['TITLE', 'ABSTRACT'], axis=1)
# Invoke data preprocess operation on the text data
combined_set['Processed'] = combined_set['TEXT'].apply(DataPreprocess())
train_set = combined_set.loc[combined_set['train_or_test'] == 0]
test_set = combined_set.loc[combined_set['train_or_test'] == 1]

# Drop key reference column
train_set = train_set.drop('train_or_test', axis=1)
test_set = test_set.drop('train_or_test', axis=1)
# View just 2 row value
train_set[0:2].values
def lsa_reduction(X_train, X_test, n_comp=120):
    svd = TruncatedSVD(n_components=n_comp)
    normalizer = Normalizer()
    
    lsa_pipe = Pipeline([('svd', svd),
                        ('normalize', normalizer)]).fit(X_train)
    
    train_reduced = lsa_pipe.transform(X_train)
    test_reduced = lsa_pipe.transform(X_test)
    return train_reduced, test_reduced

def vectorize(vector, X_train, X_test):
    vector_fit = vector.fit(X_train)
    
    X_train_vec = vector_fit.transform(X_train)
    X_test_vec = vector_fit.transform(X_test)
    
    print("Vectorization is completed.")
    return X_train_vec, X_test_vec
# Hashing Vectorizer calculates the hash value for each term thus keep only the unique words in the vector
def hash_vectorizer(X_train, X_test):
    hasher = HashingVectorizer(ngram_range=(1,2), n_features=25000)
    tfidf_transformer = TfidfTransformer(use_idf=True)
    feature_extractor = Pipeline([('hash', hasher),
                             ('tfidf', tfidf_transformer)]).fit(X_train)
    
    x_train_tf = feature_extractor.transform(X_train)
    x_test_tf = feature_extractor.transform(X_test)
    
    return x_train_tf, x_test_tf


# Hashing Vectorizer performs better than TFIDF
X_train_hashed, X_test_hashed = hash_vectorizer(train_set['Processed'], test_set['Processed'])

# X_train_hashed, X_test_hashed = vectorize(tfidf_vector, train_set['Processed'], test_set['Processed'])

# Dimension reduction
# --------------------------------------------
# Result is not very good after feature reduction
# ---------------------------------------------
# x_train_svd, x_test_svd = lsa_reduction(X_train_hashed, X_test_hashed, 500)
print(X_train_hashed.shape)
# lr = LogisticRegression(C=1.0,class_weight='balanced', 
#                         l1_ratio=0.9, 
#                         solver='saga', 
#                         penalty='l1')
svc = LinearSVC()

# One vs Restclassifier
orc_clf = OneVsRestClassifier(estimator=svc)
for target in target_cols:
    y = train_df[target]
#     print(y)
    
    # Split from the loaded dataset
    X_train, X_valid, y_train, y_test = train_test_split(X_train_hashed, y, test_size=0.2, shuffle=True, random_state=0)
    
    orc_clf.fit(X_train, y_train)
    
    y_pred = orc_clf.predict(X_valid)
    
    print("Label: %s \n Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \tF1-Score: %1.3f\n" % (target, 
                                                                                    accuracy_score(y_test, y_pred),
                                                                                     precision_score(y_test, y_pred, average='micro'),
                                                                                     recall_score(y_test, y_pred, average='micro'),
                                                                                     f1_score(y_test, y_pred, average='micro')))
# Copy of submission dataframe
output_df = submission_df.copy()

# Iterate over the target variables
for target in target_cols:
    y = train_df[target]
    
    orc_clf.fit(X_train_hashed, y)
    
    # Predict the values for test data
    y_pred = orc_clf.predict(X_test_hashed)
    # Assign the predicted vector to each column
    output_df[target] = y_pred


# Submission dataframe
output_df
# Submission file.
output_df.to_csv("ovr_svc_hash_tfidf_07.csv", index=False)
# output_df.to_csv("ovr_lr_hash_tfidf_06.csv", index=False)
