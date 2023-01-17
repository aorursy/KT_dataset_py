import pandas as pd

import warnings



warnings.filterwarnings("ignore")





# Read CSV files to get questions and tags

df_questions = pd.read_csv("../input/Questions.csv", encoding="ISO-8859-1")

df_tags = pd.read_csv("../input/Tags.csv", encoding="ISO-8859-1", dtype={'Tag': str})





df_questions.head()
df_tags.head()
# Group tags by id and join them

df_tags['Tag'] = df_tags['Tag'].astype(str)

grouped_tags = df_tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))

grouped_tags.head(5)
# Reset index for making simpler dataframe

grouped_tags.reset_index()

grouped_tags_final = pd.DataFrame({'Id':grouped_tags.index, 'Tags':grouped_tags.values})

grouped_tags_final.head(5)
# Drop unnecessary columns

df_questions.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)



# Merge questions and tags into one dataframe

df = df_questions.merge(grouped_tags_final, on='Id')

df.head(5)
import nltk



# Filter out questions with a score lower than 5

new_df = df[df['Score']>5]



# Split tags in order to get a list of tags

new_df['Tags'] = new_df['Tags'].apply(lambda x: x.split())

all_tags = [item for sublist in new_df['Tags'].values for item in sublist]



flat_list = [item for sublist in new_df['Tags'].values for item in sublist]



keywords = nltk.FreqDist(flat_list)

keywords = nltk.FreqDist(keywords)



# Get most frequent tags

frequencies_words = keywords.most_common(25)

tags_features = [word[0] for word in frequencies_words]

# Drop unnecessary columns at this point

new_df.drop(columns=['Id', 'Score'], inplace=True)

print(tags_features)
def most_common(tags):

    """Function to check if tag is in most common tag list"""

    tags_filtered = []

    for i in range(0, len(tags)):

        if tags[i] in tags_features:

            tags_filtered.append(tags[i])

    return tags_filtered



# Change Tags column into None for questions that don't have a most common tag

new_df['Tags'] = new_df['Tags'].apply(lambda x: most_common(x))

new_df['Tags'] = new_df['Tags'].apply(lambda x: x if len(x)>0 else None)



# Drop rows that contain None in Tags column

new_df.dropna(subset=['Tags'], inplace=True)

new_df.shape
from bs4 import BeautifulSoup

import lxml

import re



from nltk.corpus import stopwords

from nltk.tokenize import ToktokTokenizer

from nltk.stem.wordnet import WordNetLemmatizer



# Filter out HTML

new_df['Body'] = new_df['Body'].apply(lambda x: BeautifulSoup(x, "lxml").get_text()) 



token = ToktokTokenizer()

lemma = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))



def strip_list_noempty(mylist):

    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)

    return [item for item in newlist if item != '']



def removeStopWords(text):

    words = token.tokenize(text)

    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered))



def removePunctuation(text):

    punct = '!"$%&\'()*,./:;<=>?@[\\]^_`{|}~'

    words=token.tokenize(text)

    punctuation_filtered = []

    regex = re.compile('[%s]' % re.escape(punct))

    remove_punctuation = str.maketrans(' ', ' ', punct)

    for w in words:

        if w in tags_features:

            punctuation_filtered.append(w)

        else:

            punctuation_filtered.append(regex.sub('', w))

  

    filtered_list = strip_list_noempty(punctuation_filtered)

        

    return ' '.join(map(str, filtered_list))



def lemmatizeWords(text):

    words=token.tokenize(text)

    listLemma=[]

    for w in words:

        x=lemma.lemmatize(w, pos="v")

        listLemma.append(x.lower())

    return ' '.join(map(str, listLemma))





# Remove stopwords, punctuation and lemmatize for text in body

new_df['Body'] = new_df['Body'].apply(lambda x: removeStopWords(x))

new_df['Body'] = new_df['Body'].apply(lambda x: removePunctuation(x))

new_df['Body'] = new_df['Body'].apply(lambda x: lemmatizeWords(x))



# Remove stopwords, punctuation and lemmatize for title. Also weight title 3 times

new_df['Title'] = new_df['Title'].apply(lambda x: str(x)) 

new_df['Title'] = new_df['Title'].apply(lambda x: removePunctuation(x)) 

new_df['Title'] = new_df['Title'].apply(lambda x: removeStopWords(x)) 

new_df['Title'] = new_df['Title'].apply(lambda x: lemmatizeWords(x)) 

new_df['Title'] = new_df['Title'].apply(lambda x: ' '.join(x.split()*3))

new_df['Title']

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from scipy.sparse import hstack



# Define X, y

X1 = new_df['Body']

X2 = new_df['Title']

y = new_df['Tags']

print(len(X1), len(X2), len(y))



# Define multilabel binarizer

multilabel_binarizer = MultiLabelBinarizer()

y_bin = multilabel_binarizer.fit_transform(y)





vectorizer_X1 = TfidfVectorizer(analyzer = 'word',

                                       min_df=0.0005,

                                       max_df = 1.0,

                                       strip_accents = None,

                                       encoding = 'utf-8', 

                                       ngram_range = (1, 3),

                                       preprocessor=None,

                                       token_pattern=r"(?u)\S\S+",

                                       max_features=35000)



vectorizer_X2 = TfidfVectorizer(analyzer = 'word',

                                       min_df=0.0,

                                       max_df = 1.0,

                                       strip_accents = None,

                                       encoding = 'utf-8', 

                                       ngram_range = (1, 3),

                                       preprocessor=None,

                                       token_pattern=r"(?u)\S\S+",

                                       max_features=35000)



X1_tfidf = vectorizer_X1.fit_transform(X1)

X2_tfidf = vectorizer_X2.fit_transform(X2)



# Stack X1 and X2 into X_tfidf

X_tfidf = hstack([X1_tfidf,X2_tfidf])



# Split training and test data    

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_bin, test_size = 0.2, random_state = 0)
# Using Label Powerset

from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

from sklearn.metrics import hamming_loss

from sklearn.metrics import f1_score

from skmultilearn.problem_transform import LabelPowerset

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import jaccard_similarity_score

from sklearn import model_selection

from sklearn.metrics import make_scorer

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score



svc = LinearSVC()

sgd = SGDClassifier(n_jobs=-1)



def print_score(y_pred, clf):

    print("Clf: ", clf.__class__.__name__)

    print("Accuracy score: {}".format(accuracy_score(y_test, y_pred)))

    print("Recall score: {}".format(recall_score(y_true=y_test, y_pred=y_pred, average='weighted')))

    print("Precision score: {}".format(precision_score(y_true=y_test, y_pred=y_pred, average='weighted')))

    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)*100))

    print("F1 score: {}".format(f1_score(y_pred, y_test, average='weighted')))

    print("---")    



clf = LabelPowerset(svc)



clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print_score(y_pred, clf)



kfold = KFold(n_splits=5)

X_sparse = X_tfidf.tocsr()



scores = []



for train_indices, test_indices in kfold.split(X_sparse, y_bin):

    clf.fit(X_sparse[train_indices], y_bin[train_indices])

    print(clf.score(X_sparse[test_indices], y_bin[test_indices]))

    scores.append(clf.score(X_sparse[test_indices], y_bin[test_indices]))



print(sum(scores)/len(scores))
# Using Classifier Chains

from sklearn.multioutput import ClassifierChain

import numpy as np





chains = [ClassifierChain(svc, order='random', random_state=i)

          for i in range(10)]



for chain in chains:

    chain.fit(X_train, y_train)



Y_pred_chains = np.array([chain.predict(X_test) for chain in

                          chains])



Y_pred_ensemble = Y_pred_chains.mean(axis=0)

ensemble_accuracy_score = accuracy_score(y_test, Y_pred_ensemble >= .5)

ensemble_recall_score = recall_score(y_test, Y_pred_ensemble >= .5, average='weighted')

ensemble_precision_score = precision_score(y_test, Y_pred_ensemble >= .5, average='weighted')

ensemble_f1_score = f1_score(y_pred, Y_pred_ensemble >= .5, average='weighted')

hamm = hamming_loss(Y_pred_ensemble >= .5, y_test)*100

print(ensemble_accuracy_score, ensemble_recall_score, ensemble_precision_score, ensemble_f1_score, hamm)

# Using Binary Relevance

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.naive_bayes import GaussianNB



# initialize binary relevance multi-label classifier

# with a gaussian naive bayes base classifier

classifier = BinaryRelevance(svc)



# train

classifier.fit(X_train, y_train)



# predict

predictions = classifier.predict(X_test)



print_score(predictions, classifier)