# load packages and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import lightgbm as lgb
from sklearn.pipeline import Pipeline, FeatureUnion
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
%matplotlib inline
sns.set_style('darkgrid')
df = pd.read_csv('../input/animal-crossing/user_reviews.csv')
df.head()
# check null values
df.info()
# check the distribution of grade
plt.figure(figsize=(8,6))
plt.bar(df.grade.value_counts().index, df.grade.value_counts().values)
plt.xlabel('Review Grade')
plt.ylabel('Count')
plt.title('Distribution of Review Grade');
df['target'] = pd.cut(df.grade, 2, labels=[0, 1])
df.target.value_counts()
def tokenize(text):
    """Tokenize each review text
    Args: text
    Return: token lists after normalization and lemmatization
    """
    # remove punctuation and change to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
    # tokenize the word into words
    tokens = word_tokenize(text)

    # remove stopwords
    stop_words  = set(stopwords.words('english'))
    
    tokens = [word for word in tokens if word not in stop_words]

    # lemmatize the word
    lemmatizer = WordNetLemmatizer()
    clean_token = []
    for token in tokens:
        clean_token.append(lemmatizer.lemmatize(token, pos='v').lower().strip())
    return clean_token
# concatenate all comments in low grade group 
dict_low = df[df.target == 0].to_dict(orient='list')
low_grade = dict_low['text']

dict_high = df[df.target == 1].to_dict(orient='list')
high_grade = dict_high['text']
# tokenize text for Counter
low_grade = ' '.join(low_grade)
low_tokens = tokenize(low_grade)

high_grade = ' '.join(high_grade)
high_tokens = tokenize(high_grade)
# Count the most popular words 
low_counter = Counter(low_tokens)
low_top20 = low_counter.most_common(20)

high_counter = Counter(high_tokens)
high_top20 = high_counter.most_common(20)
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.barh(range(len(low_top20)), [val[1] for val in low_top20], align='center')
plt.yticks(range(len(low_top20)), [val[0] for val in low_top20])
plt.xlabel('Count')
plt.ylabel('Top 20 Common Words')
plt.title('Most 20 common words in low grading group')

plt.subplot(1, 2, 2)
plt.barh(range(len(high_top20)), [val[1] for val in high_top20], align='center')
plt.yticks(range(len(high_top20)), [val[0] for val in high_top20])
plt.xlabel('Count')
plt.ylabel('Top 20 Common Words')
plt.title('Most 20 common words in high grading group');
# use pos_tag in NLP to select adjectives
ad_tokens_low = []
for word, tag in pos_tag(low_tokens):
    if tag in ('JJ', 'JJR', 'JJS'):
        ad_tokens_low.append(word)

ad_tokens_high = []
for word, tag in pos_tag(high_tokens):
    if tag in ('JJ', 'JJR', 'JJS'):
        ad_tokens_high.append(word)
# Count the most popular adjective/adverb 
ad_low_counter = Counter(ad_tokens_low)
ad_low_top20 = ad_low_counter.most_common(20)

ad_high_counter = Counter(ad_tokens_high)
ad_high_top20 = ad_high_counter.most_common(20)
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.barh(range(len(ad_low_top20)), [val[1] for val in ad_low_top20], align='center')
plt.yticks(range(len(ad_low_top20)), [val[0] for val in ad_low_top20])
plt.xlabel('Count')
plt.ylabel('Top 20 Common Adjective')
plt.title('Most 20 common adjective in low grading group')

plt.subplot(1, 2, 2)
plt.barh(range(len(ad_high_top20)), [val[1] for val in ad_high_top20], align='center')
plt.yticks(range(len(ad_high_top20)), [val[0] for val in ad_high_top20])
plt.xlabel('Count')
plt.ylabel('Top 20 Common adjective')
plt.title('Most 20 common adjective in high grading group');
# setup stop words
stop_words  = set(stopwords.words('english'))
# update stop words
stop_words.update(['this', 'game', 'the', 'play'])
# generate word cloud for low score group
wordcloud = WordCloud(background_color='white', max_words=1000, contour_width=3,contour_color='firebrick', 
                      stopwords = stop_words)

wordcloud.generate(re.sub(r'[^a-zA-Z0-9]', ' ', low_grade).lower())
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# generate word cloud for high score group
wordcloud = WordCloud(background_color='white', max_words=1000, contour_width=3,contour_color='firebrick', 
                      stopwords = stop_words)

wordcloud.generate(re.sub(r'[^a-zA-Z0-9]', ' ', high_grade).lower())
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
class AdCount(BaseEstimator, TransformerMixin):
    """ Custom transformer to count the number of adj and adv in text
    Args: text
    Return: Adjective and Adverb counts in the text
    """
    def Ad_count(self, text):
        text = re.sub(r'[^a-zA-Z0-9]', ' ', text).lower()
        # tokenize the word into words
        tokens = word_tokenize(text)
        # remove stopwords
        stop_words  = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        count = 0
        for word, tag in pos_tag(tokens):
            if tag in ('RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS'):
                count+=1
        return count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        counts = pd.Series(X).apply(self.Ad_count)
        return pd.DataFrame(counts)
def build_model():
    """
    A Adaboost classifier machine learning pipeline for
    natural language processing with tdidf, adcount, and gridsearch for optimization.
    Args: X_train, y_train 
    Returns:
        Fitted model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
             ('ad_count', AdCount())
             ])),
        ('clf', lgb.LGBMClassifier(objective='binary', random_state=0))
    ])
    parameters = {
        #'clf__n_estimators': [100],
        'clf__learning_rate': [0.01, 1],
        'clf__num_leaves': [31, 62]
        #'clf__min_samples_split': [5]
        #'clf__estimator__C': [1, 10],
        #'clf__estimator__max_iter': [1000, 100000]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
def evaluate_model(cv, X_test, y_test):
    """Draw ROC curve for the model
    Args:
        Classification Model
        X_test, y_test, Array-like
    return: ROC curve and model pickles
    """
    y_pred = cv.predict_proba(X_test)[:,1]
    print('\nBest Parameters:', cv.best_params_)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr) # compute area under the curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
# split train and test 
X = df.text
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# build model (randomforests)
model = build_model()
model.fit(X_train, y_train)
# evaluate
pred = model.predict(X_test)
print('Accuracy_score is: {}'.format(accuracy_score(pred, y_test)))
evaluate_model(model, X_test, y_test)
f = open('rf'+'.pkl', 'wb')
pickle.dump(model, f)
# build model Adaboost
model_ad = build_model()
model_ad.fit(X_train, y_train)
# evaluate
pred_ad = model_ad.predict(X_test)
print('Accuracy_score is: {}'.format(accuracy_score(pred_ad, y_test)))
evaluate_model(model_ad, X_test, y_test)
f = open('ada'+'.pkl', 'wb')
pickle.dump(model, f)
# build model Adaboost
model_lgb = build_model()
model_lgb.fit(X_train, y_train)
# evaluate
pred_lgb = model_ad.predict(X_test)
print('Accuracy_score is: {}'.format(accuracy_score(pred_lgb, y_test)))
print('Best parameter is: {}'.format(model_lgb.best_params_))
evaluate_model(model_lgb, X_test, y_test)
f = open('lgb'+'.pkl', 'wb')
pickle.dump(model_lgb, f)
# fun test
test = ["it is amazing. I'm totally adicted"]
print(model.predict(test))
print(model_ad.predict(test))
print(model_lgb.predict(test))
# fun test
ntest = ["This game sucks, make me stressful"]
print(model.predict(ntest))
print(model_ad.predict(ntest))
print(model_lgb.predict(ntest))