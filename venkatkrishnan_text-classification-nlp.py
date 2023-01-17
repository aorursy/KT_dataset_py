import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/spooky-author/train.csv')
test_data = pd.read_csv('/kaggle/input/spooky-author/test.csv')
sample_submit = pd.read_csv('/kaggle/input/spooky-author/sample_submission.csv')
train_data.info()
print("--"*30)
train_data.head()
test_data.info()
print("--"*30)
test_data.head()
train_data.isnull().sum()
# Define log_loss function for multi-class problem with eps 1 femto
def multiclass_logloss(actual, predicted, eps=1e-15):
    if len(actual.shape) == 1:
        binary_actual = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            binary_actual[i, val] = 1
        actual = binary_actual
        
    clip = np.clip(predicted, eps, 1-eps)
    
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    
    return -1.0 / rows * vsota

    
lbl_encoder = LabelEncoder()

y = lbl_encoder.fit_transform(train_data['author'].values)
X = train_data['text'].values

X_train, X_valid, y_train, y_valid = train_test_split(X,y, 
                                                      stratify=y,
                                                      shuffle=True,
                                                      random_state=42,
                                                      test_size=0.1)
print(X_train.shape)
print(X_valid.shape)
tfidf_vector = TfidfVectorizer(min_df=3, analyzer='word', 
                               strip_accents='unicode', 
                               token_pattern=r'\w{1}',
                               ngram_range=(1,3), 
                               use_idf=1, 
                               smooth_idf=1,
                                sublinear_tf=1,
                              stop_words='english')

tfidf_vector.fit(list(X_train) + list(X_valid))
xtrain_tfv = tfidf_vector.transform(X_train)
xvalid_tfv = tfidf_vector.transform(X_valid)
print(xtrain_tfv.shape)
print(xvalid_tfv.shape)
log_clf = LogisticRegression(C=1.0)
log_clf.fit(xtrain_tfv, y_train)

predictions = log_clf.predict_proba(xvalid_tfv)
print("logloss: %0.3f" % multiclass_logloss(y_valid, predictions))
svd_alg = TruncatedSVD(n_components=120)
svd_alg.fit(xtrain_tfv)

x_train_svd = svd_alg.transform(xtrain_tfv)
x_valid_svd = svd_alg.transform(xvalid_tfv)
# Scale the dim reduced matrix 
scaler = StandardScaler()

scaler.fit(x_train_svd)

x_train_scaled = scaler.transform(x_train_svd)
x_valid_scaled = scaler.transform(x_valid_svd)
# Random Forest classifier
rfc = RandomForestClassifier()

rfc.fit(x_train_scaled, y_train)
prediction = rfc.predict_proba(x_valid_scaled)
print("Log loss: %0.3f" % multiclass_logloss(y_valid, prediction))
