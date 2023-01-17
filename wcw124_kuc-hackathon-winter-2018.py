# ALL imports

import warnings  

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style; style.use('ggplot')



import re

import xgboost as xgb



from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from wordcloud import WordCloud, STOPWORDS



from keras.models import Sequential

from keras.layers import Dense, LSTM, Embedding

from keras.utils import to_categorical

# Create dataframes train and test

train = pd.read_csv('../input/drugsComTrain_raw.csv')

test = pd.read_csv('../input/drugsComTest_raw.csv')

print(len(train))

print(len(test))
# Sort train dataframe from most to least useful

useful_train = train.sort_values(by='usefulCount', ascending=False)

useful_train.iloc[:10]
# Print top 3 most useful reviews

print("3 most useful reviews: \n")

for i in useful_train.review.iloc[:3]:

    print(i, '\n')
# Print 3 of the least useful reviews

print("3 of the least useful reviews: \n")

for i in useful_train.review.iloc[-3:]:

    print(i, '\n')


wc = WordCloud(stopwords=STOPWORDS).generate(train.review[1])

plt.imshow(wc)
"""

# Create a list of all drugs and their average ratings, cast to dataframe

rate_ls = []



for i in train.drugName.unique():

    

    # Only consider drugs that have at least 10 ratings

    if np.sum(train.drugName == i) >= 10:

        rate_ls.append((i, np.sum(train[train.drugName == i].rating) / np.sum(train.drugName == i)))

    

avg_rate = pd.DataFrame(rate_ls)

"""
# Sort drugs by their ratings, look at top 10 best and worst rated drugs

#avg_rate = avg_rate.sort_values(by=[1], ascending=False).reset_index(drop=True)

#avg_rate[:10]
#avg_rate[-10:]
# 10 most common conditions

conditions = train.condition.value_counts().sort_values(ascending=False)

conditions[:10]
conditions[:10].plot(kind='bar')

plt.title('Histogram of Review Counts for the 10 Most Common Conditions')

plt.xlabel('Condition')

plt.ylabel('Number of Reviews');
# Empirical Distribution of Ratings

train.rating.hist(bins=10)

plt.title('Distribution of Ratings')

plt.xlabel('Rating')

plt.ylabel('Count')

plt.xticks([i for i in range(1, 11)]);
rating_avgs = (train['rating'].groupby(train['drugName']).mean())

rating_avgs.hist(color='skyblue')

plt.title('Distribution of average drug ratings')

plt.xlabel('Rating')

plt.ylabel('Count')
rating_avgs = (train['rating'].groupby(train['condition']).mean())

rating_avgs.hist(color='skyblue')

plt.title('Averages of medication reviews for each disease')

plt.xlabel('Rating')

plt.ylabel('Count')

plt.show()
# Is rating correlated with usefulness of the review?

plt.scatter(train.rating, train.usefulCount, c=train.rating.values, cmap='tab10')

plt.title('Useful Count vs Rating')

plt.xlabel('Rating')

plt.ylabel('Useful Count')

plt.xticks([i for i in range(1, 11)]);
# Create a list (cast into an array) containing the average usefulness for given ratings

use_ls = []



for i in range(1, 11):

    use_ls.append([i, np.sum(train[train.rating == i].usefulCount) / np.sum([train.rating == i])])

    

use_arr = np.asarray(use_ls)



plt.scatter(use_arr[:, 0], use_arr[:, 1], c=use_arr[:, 0], cmap='tab10', s=200)

plt.title('Average Useful Count vs Rating')

plt.xlabel('Rating')

plt.ylabel('Average Useful Count')

plt.xticks([i for i in range(1, 11)]);
train.review.head()
# Input: s - a string

# Output: a string with any enclosing quotation marks removed

def remove_enclosing_quotes(s):

    if s[0] == '"' and s[-1] == '"':

        return s[1:-1]

    else:

        return s

    

train.review = train.review.apply(remove_enclosing_quotes)

test.review = test.review.apply(remove_enclosing_quotes)

train.head()
import re

train.review = train.review.apply(lambda x: re.sub(r'&#\d+;',r'', x))

test.review = test.review.apply(lambda x: re.sub(r'&#\d+;',r'', x))
# Inputs:

#    data_frame - a pandas data frame containing text columns

#    text_cols - a list of column names in data_frame containing the text columns to be combined

# Outputs:

#    text_data - a dataframe containing a single column which joins all columns in the text_cols

#                list into a single column separated by spaces.

# Side-Effects: n/a

def combine_text_columns(data_frame, text_cols):

    """

    Converts all text in each row of data_frame to a single vector

    """

    #col_list = list(set(text_cols) & set(data_frame.columns.tolist()))

    text_data = data_frame[text_cols]

    

    text_data.fillna("", inplace=True)

    

    return text_data.apply(lambda x: " ".join(x), axis=1)
# Create the text_vector combining all  text columns

text_cols = ['drugName', 'condition', 'review']

train['text'] = combine_text_columns(train, text_cols)

test['text'] = combine_text_columns(test, text_cols)

train[['text', 'rating', 'usefulCount']].head()
# The token pattern grouping alpha-numerically

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'



# Instantiation of CountVectorizer object

vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1,2), lowercase=True, stop_words='english', min_df=2, max_df=0.99) # added min_df param



# Fit to training data

X = vec_alphanumeric.fit_transform(train.text)



# Create a column with binary rating indicating the polarity of a review

train['binary_rating'] = train['rating'] > 5



y = train.binary_rating



# Print the total number of tokesn and first 15 tokens

msg = 'There are {} tokens in the review column if we split on non-alpha numeric.'

print(msg.format(len(vec_alphanumeric.get_feature_names())))

print(vec_alphanumeric.get_feature_names()[:30])
# Train test split

# NB: We are using the data in the drugsComTest_raw.csv file as a holdout set for final performance evaluation.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.1) # Default is 75%/25% split



clf = MultinomialNB().fit(X_train, y_train)



pred = clf.predict(X_test)



print("Accuracy on training set: {}".format(clf.score(X_train, y_train)))

print("Accuracy on test set: {}".format(clf.score(X_test, y_test)))

print("Confusion Matrix")

print(confusion_matrix(pred, y_test))
"""

alphas = np.linspace(0, 2, 10)

train_accs = []

test_accs = []



for a in alphas:

    clfa = MultinomialNB(alpha=a).fit(X_train, y_train)

    pred = clfa.predict(X_test)

    train_accs.append(clfa.score(X_train, y_train))

    test_accs.append(clfa.score(X_test, y_test))

    

plt.plot(alphas, train_accs, 'ro')

plt.plot(alphas, test_accs, 'go')

plt.show()

"""
# Projects hold-out set's text into the space formed by the test corpus

X_validation = vec_alphanumeric.transform(test.text)



# Binarizes ratings for the validation set

test['binary_rating'] = test['rating'] > 5

y_validation = test.binary_rating



clf = MultinomialNB(alpha=0.0).fit(X, y)



#

print("Accuracy on test set: {}".format(clf.score(X_validation, y_validation)))

pred_validation = clf.predict(X_validation)

print("Confusion Matrix: ")

print(confusion_matrix(y_validation, pred_validation))

print(classification_report(y_validation, pred_validation))
# The token pattern grouping alpha-numerically

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'



# Instantiation of CountVectorizer object

vec_alphanumeric = TfidfVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1,2), lowercase=True, stop_words='english', min_df=2, max_df=0.99) # added min_df param



# Fit to training data

X = vec_alphanumeric.fit_transform(train.text)



# Create a column with binary rating indicating the polarity of a review

train['binary_rating'] = train['rating'] > 5



y = train.binary_rating



# Print the total number of tokesn and first 15 tokens

msg = 'There are {} tokens in the review column if we split on non-alpha numeric.'

print(msg.format(len(vec_alphanumeric.get_feature_names())))

print(vec_alphanumeric.get_feature_names()[:30])
# Train test split

# NB: We are using the data in the drugsComTest_raw.csv file as a holdout set for final performance evaluation.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.1) # Default is 75%/25% split



clf_lr = LogisticRegression(penalty='l2', C=100).fit(X_train, y_train)



pred = clf_lr.predict(X_test)



print("Accuracy on training set: {}".format(clf_lr.score(X_train, y_train)))

print("Accuracy on test set: {}".format(clf_lr.score(X_test, y_test)))

print("Confusion Matrix")

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))
# Projects hold-out set's text into the space formed by the test corpus

X_validation = vec_alphanumeric.transform(test.text)



# Binarizes ratings for the validation set

test['binary_rating'] = test['rating'] > 5

y_validation = test.binary_rating



clf_lr = LogisticRegression(penalty='l2', C=100).fit(X, y)



#

print("Accuracy on test set: {}".format(clf_lr.score(X_validation, y_validation)))

pred_validation = clf_lr.predict(X_validation)

print("Confusion Matrix: ")

print(confusion_matrix(y_validation, pred_validation)/len(y_validation))

print(classification_report(y_validation, pred_validation))
# roc curve

y_pred_prob = clf_lr.predict_proba(X_validation)[:,1]



fpr, tpr, thresholds = roc_curve(y_validation, y_pred_prob)



plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Logistic Regression Classifier")

plt.show()



print("AUC: {}".format(roc_auc_score(y_validation, y_pred_prob)))
# Train test split

# NB: We are using the data in the drugsComTest_raw.csv file as a holdout set for final performance evaluation.

"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y) # Default is 75%/25% split



clf_xg = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, max_depth=4).fit(X_train, y_train)



pred = clf_xg.predict(X_test)



print("Accuracy on training set: {}".format(clf_xg.score(X_train, y_train)))

print("Accuracy on test set: {}".format(clf_xg.score(X_test, y_test)))

print("Confusion Matrix")

print(confusion_matrix(pred, y_test))

"""
# Train test split

# NB: We are using the data in the drugsComTest_raw.csv file as a holdout set for final performance evaluation.

"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y) # Default is 75%/25% split



clf_rfc = RandomForestClassifier().fit(X_train, y_train)



pred = clf_rfc.predict(X_test)



print("Accuracy on training set: {}".format(clf_rfc.score(X_train, y_train)))

print("Accuracy on test set: {}".format(clf_rfc.score(X_test, y_test)))

print("Confusion Matrix")

print(confusion_matrix(pred, y_test))

"""
# Train test split

# NB: We are using the data in the drugsComTest_raw.csv file as a holdout set for final performance evaluation.

"""

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y) # Default is 75%/25% split



clf_svc = SVC().fit(X_train, y_train)



pred = clf_svc.predict(X_test)



print("Accuracy on training set: {}".format(clf_svc.score(X_train, y_train)))

print("Accuracy on test set: {}".format(clf_svc.score(X_test, y_test)))

print("Confusion Matrix")

print(confusion_matrix(pred, y_test))

"""
"""

# Instantiation of CountVectorizer object

vec_alphanumeric = TfidfVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1,2), lowercase=True, stop_words='english', max_df=0.99, min_df=2, max_features=200000) # added min_df param



# Fit to training data

X = vec_alphanumeric.fit_transform(train.text)



# Create a column with binary rating indicating the polarity of a review

train['binary_rating'] = train['rating'] > 5



y = train.binary_rating



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.1) 



model = Sequential()

model.add(Dense(units=4, activation='relu', input_dim=len(vec_alphanumeric.get_feature_names()))) # 4 nodes - 6 min/epoch

model.add(Dense(units=2, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

history = model.fit(X_train, y_train, epochs=4, batch_size=128, verbose=1, validation_data=(X_test, y_test)) # ~5 mins/epoch



# Projects hold-out set's text into the space formed by the test corpus

X_validation = vec_alphanumeric.transform(test.text)



# Binarizes ratings for the validation set

test['binary_rating'] = test['rating'] > 5

y_validation = test.binary_rating



y_validation = to_categorical(y_validation)

score = model.evaluate(X_validation, y_validation, batch_size=128)



print(score)



y_cat = to_categorical(y)

final = model.fit(X, y_cat, epochs=2, batch_size=128, verbose=1, validation_data=(X_validation, y_validation)) # ~6 mins/epoch

"""
#scaler = StandardScaler().fit(X)

#X_standardized = scaler.transform(X)

#standardized_X_test = scaler.transform()
print("The length of a tokenized vector is {}.".format(X.shape[1]))

print("The number of records is {}.".format(X.shape[0]))
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

num_features = 1000



pl = Pipeline([

    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1,2))),

    #('dim_red', SelectKBest(chi2, num_features)),

    ('clf', MultinomialNB())

])



#pl.fit(train.text, train.binary_rating)

#pl.score(train.text, train.binary_rating)
from itertools import combinations



import numpy as np

from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin





class SparseInteractions(BaseEstimator, TransformerMixin):

    def __init__(self, degree=2, feature_name_separator="_"):

        self.degree = degree

        self.feature_name_separator = feature_name_separator



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        if not sparse.isspmatrix_csc(X):

            X = sparse.csc_matrix(X)



        if hasattr(X, "columns"):

            self.orig_col_names = X.columns

        else:

            self.orig_col_names = np.array([str(i) for i in range(X.shape[1])])



        spi = self._create_sparse_interactions(X)

        return spi



    def get_feature_names(self):

        return self.feature_names



    def _create_sparse_interactions(self, X):

        out_mat = []

        self.feature_names = self.orig_col_names.tolist()



        for sub_degree in range(2, self.degree + 1):

            for col_ixs in combinations(range(X.shape[1]), sub_degree):

                # add name for new column

                name = self.feature_name_separator.join(self.orig_col_names[list(col_ixs)])

                self.feature_names.append(name)



                # get column multiplications value

                out = X[:, col_ixs[0]]

                for j in col_ixs[1:]:

                    out = out.multiply(X[:, j])



                out_mat.append(out)



        return sparse.hstack([X] + out_mat)
# HashingVectorizer could potentially speed this kernel up

# may also be useful if we want to try out more complex models





text_data = train.text

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

hashing_vect = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC, stop_words='english', n_features=200000, norm=None, binary=False, ngram_range=(1,2))

hashed_text = hashing_vect.fit_transform(text_data)

hashed_df = pd.DataFrame(hashed_text.data)

hashed_text.shape
# Train test split

# NB: We are using the data in the drugsComTest_raw.csv file as a holdout set for final performance evaluation.

X_train, X_test, y_train, y_test = train_test_split(hashed_text, y, random_state=42, stratify=y) # Default is 75%/25% split



clf_lr = LogisticRegression(penalty='l1', solver='liblinear', C=3).fit(X_train, y_train)



pred = clf_lr.predict(X_test)



print("Accuracy on training set: {}".format(clf_lr.score(X_train, y_train)))

print("Accuracy on test set: {}".format(clf_lr.score(X_test, y_test)))

print("Confusion Matrix")

print(confusion_matrix(pred, y_test))