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

from gensim.models import Word2Vec # Word2Vec module
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, remove_stopwords, strip_numeric, stem_text
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler

from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')
test_data = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')
submission_data = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv')
print(train_data.isnull().sum())
print(train_data.columns)
# Converting binary column to category
to_convert = ['Computer Science', 'Physics', 'Mathematics','Statistics', 'Quantitative Biology', 'Quantitative Finance']

# Make a copy of train data
topic_data = train_data.copy()
# Changing the binary fields to categorical fields
topic_data = topic_data[topic_data[to_convert]==1].stack().reset_index().drop(0,1)

topic_data['ID'] = topic_data['level_0'].apply(lambda x: x+1)
topic_data = topic_data.drop('level_0', axis=1)

# Merge the data based on ID
merge_data = train_data.merge(topic_data, how='left', on='ID' )
# Drop all the binary fields
merge_data = merge_data.drop(to_convert, axis=1)

# Rename the column to Category
merge_data = merge_data.rename({'level_1':'CATEGORY'}, axis=1)
merge_data
articles = merge_data

# list unique classes
print(np.unique(articles.CATEGORY))
# Plot category data
plt.figure(figsize=(10,6))
sns.countplot(articles.CATEGORY)
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
#             filtered_words = set(wnl.lemmatize(w) if w.endswith('e') or w.endswith('y') else porter.stem(w) for w in cleanse_words)
            return ' '.join(cleanse_words)
        except TypeError as te:
            raise(TypeError("Not a valid data {}".format(te)))
# Combine the Title and Abstract data
articles['TEXT'] = articles['TITLE'].map(str) + articles['ABSTRACT'].map(str)

articles['Processed'] = articles['TEXT'].apply(DataPreprocess())

articles['Processed'].values
X = np.reshape(articles['Processed'].values, (-1,1))
y = np.reshape(articles['CATEGORY'].values, (-1,1))

ros = RandomOverSampler(sampling_strategy='minority', random_state=27)

X_res, y_res = ros.fit_resample(X, y)

print(X_res.shape, y_res.shape)
def vectorize(vector, X_train, X_test):
    vector_fit = vector.fit(X_train)
    
    X_train_vec = vector_fit.transform(X_train)
    X_test_vec = vector_fit.transform(X_test)
    
    print("Vectorization is completed.")
    return X_train_vec, X_test_vec

def label_encoding(y_train):
    """
        Encode the given list of class labels
        :y_train_enc: returns list of encoded classes
        :labels: actual class labels
    """
    lbl_enc = LabelEncoder()
    
    y_train_enc = lbl_enc.fit_transform(y_train)
    labels = lbl_enc.classes_
    
    return y_train_enc, labels


# Encode the class labels
X = X_res
y = y_res

y_enc_train, labels = label_encoding(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y_enc_train, test_size=0.3, shuffle=True)
print(X_train.shape, X_valid.shape)
tfidf_vector = TfidfVectorizer(min_df=3, analyzer='word', 
                               strip_accents='unicode', 
                               token_pattern=r'\w{1}',
                               ngram_range=(1,3), 
                               max_features=3000,
                               use_idf=1, 
                               smooth_idf=1,
                                sublinear_tf=1,
                              stop_words='english')
# TFIDFVectorizer 
X_train_vec, X_valid_vec = vectorize(tfidf_vector, X_train.flatten(), X_valid.flatten())
lgbm_params = {
    'n_estimators': 500,
    'num_leaves': 60,
     'min_data_in_leaf': 60, 
     'objective':'multiclass',
     'max_depth': 6,
     'learning_rate': 0.2,
     "boosting": "gbdt",
     "feature_fraction": 0.8,
     "bagging_freq": 1,
     "bagging_fraction": 0.8 ,
     "bagging_seed": 11,
     "eval_metric": 'logloss',
     "lambda_l1": 0.5,
     "random_state": 42,
    'verbose':1
    
}

lgbm_clf = LGBMClassifier(**lgbm_params)
lgbm_clf.fit(X_train_vec, y_train)

# model = LinearSVC()
model = LogisticRegression(C=1.0, 
                           class_weight='balanced')

# Initialize OVR classifier with ML Algorithm
ovr = OneVsRestClassifier(estimator=model)

ovr.fit(X_train_vec, y_train)
# y_pred = lgbm_clf.predict(X_valid_vec)
y_pred = ovr.predict(X_valid_vec)

print("Accuracy: %1.3f \tPrecision: %1.3f \tRecall: %1.3f \tF1-Score: %1.3f\n" % (accuracy_score(y_valid, y_pred),
                                                                                     precision_score(y_valid, y_pred, average='micro'),
                                                                                     recall_score(y_valid, y_pred, average='micro'),
                                                                                     f1_score(y_valid, y_pred, average='micro')))
test_data['Text'] = test_data['TITLE'] + test_data['ABSTRACT']

test_data['Processed'] = test_data['Text'].apply(DataPreprocess())
# Feature extraction
X_train_vec, X_test_vec = vectorize(tfidf_vector, X_train.flatten(), test_data['Processed'])
# y_preds = lgbm_clf.predict(X_test_vec)
y_preds = ovr.predict(X_test_vec)
test_df = test_data.copy()

test_df['category'] = pd.Series(y_preds, index=test_data.index)
test_df['category'].unique()

test_df[labels] = pd.get_dummies(test_df['category'], columns=labels)

final_df = test_df.drop(['TITLE', 'ABSTRACT', 'Text', 'Processed', 'category'], axis=1)

submission_data = final_df[submission_data.columns]

submission_data

submission_data.to_csv('multiclass_lr_04.csv', index=False)
