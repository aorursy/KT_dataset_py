import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Sklearn package
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.svm import SVC

from sklearn import metrics
# class labels
class_labels = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

# Training set data
news_train = fetch_20newsgroups(subset='train', 
                               categories=class_labels,
                               shuffle=True,
                               random_state=42)
# Test set
news_test = fetch_20newsgroups(subset='test',
                              categories=class_labels,
                              shuffle=True,
                              random_state=42)

news_train.keys()
def display_report(pipe):
    y_test = news_test.target
    # Perform scoring on the test data
    y_pred = pipe.predict(news_test.data)
    
    report = metrics.classification_report(y_test, y_pred, target_names=news_test.target_names)
    
    print(report)
    
    print("Overall accuracy {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))

tfidf_vec = TfidfVectorizer(min_df=3, stop_words='english',
                      ngram_range=(1, 2))

svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)

normalizer = Normalizer()

lsa = make_pipeline(tfidf_vec, svd, normalizer)

# clf = SVC(kernel='linear', C=150, gamma=2e-2, probability=True)
clf = LogisticRegressionCV()

pipe = make_pipeline(lsa, clf)
pipe.fit(news_train.data, news_train.target)

# Display the score
display_report(pipe)

# !pip install eli5
import eli5 as mldebug
mldebug.show_weights(clf, top=10)
# Show weights with feature names
mldebug.show_weights(clf, vec=tfidf_vec, top=10,
                    target_names=news_test.target_names)
# Lets try with the 1st position test data
news_test.data[0]
mldebug.show_prediction(clf, 
                        news_test.data[0], 
                        top=10,
                        vec=lsa,
                        target_names=news_test.target_names)
