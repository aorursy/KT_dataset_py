%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

import re

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

import pickle

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from scipy.sparse import hstack

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
true_data = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

true_data.head()
true_data.shape
fake_data = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

fake_data.head()
fake_data.shape
random_true_news = random.randint(0,true_data.shape[0])

random_fake_news = random.randint(0,fake_data.shape[0])
true_data['title'][random_true_news]
true_data['text'][random_true_news]
fake_data['title'][random_fake_news]
fake_data['text'][random_fake_news]
fake_data['target'] = 'fake'

true_data['target'] = 'true'
news = pd.concat([fake_data, true_data]).reset_index(drop = True)

news.head()
# https://stackoverflow.com/a/47091490/4084039

'''Function to expand commonly occuring test'''

def decontracted(phrase):

    # specific

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
# https://gist.github.com/sebleier/554280

# we are removing the words from the stop words list: 'no', 'nor', 'not'

stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]
# Combining all the above stundents 

def preprocessTextData(dataToProcess):

    """This function do the preprocessing of the column text data in essay and title"""

    processedData = []

    # tqdm is for printing the status bar

    for sentance in tqdm(dataToProcess):

        lowersent = sentance.lower()

        sent = decontracted(lowersent)

        sent = sent.replace('\\r', ' ')

        sent = sent.replace('\\"', ' ')

        sent = sent.replace('\\n', ' ')

        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

        # https://gist.github.com/sebleier/554280

        sent = ' '.join(e for e in sent.split() if e not in stopwords)

        processedData.append(sent.strip())

    return processedData
news['processed title'] = preprocessTextData(news['title'].values)
news['processed text'] = preprocessTextData(news['text'].values)
news['title'][random_fake_news]
news['processed title'][random_fake_news]
news['text'][random_fake_news]
news['processed text'][random_fake_news]
x_train,x_test,y_train,y_test = train_test_split(news[['processed title', 'processed text', 'subject']], news.target, test_size=0.2, random_state=42)
#fitting categorical data

def fitCatogarizedData(dataToProcess, vocab = None):

    if vocab is None :

        vectorizer = CountVectorizer()

    else:

        vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, binary=True)

    vectorizer.fit(dataToProcess)

    return vectorizer



#transforming categorical data

def transformCatogarizedData(dataToProcess, vectorizer):

    categories_one_hot = vectorizer.fit_transform(dataToProcess)

    print(vectorizer.get_feature_names())

    print("Shape of matrix after one hot encodig ",categories_one_hot.shape)

    return categories_one_hot
train_vector = fitCatogarizedData(x_train['subject'].values)

x_train_cat = transformCatogarizedData(x_train['subject'].values, train_vector)

x_test_cat = transformCatogarizedData(x_test['subject'].values, train_vector)
# stronging variables into pickle files python: http://www.jessicayung.com/how-to-use-pickle-to-save-and-load-variables-in-python/

# make sure you have the glove_vectors file

glove_vectors = '/kaggle/input/donors-chose/glove_vectors'

with open(glove_vectors, 'rb') as f:

    model = pickle.load(f)

    glove_words =  set(model.keys())
# average Word2Vec

# compute average word2vec for each review.

def fitAvgW2V(dataToProcess):

    avg_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list

    for sentence in tqdm(dataToProcess): # for each review/sentence

        vector = np.zeros(300) # as word vectors are of zero length

        cnt_words =0; # num of words with a valid vector in the sentence/review

        for word in sentence.split(): # for each word in a review/sentence

            if word in glove_words:

                vector += model[word]

                cnt_words += 1

        if cnt_words != 0:

            vector /= cnt_words

        avg_w2v_vectors.append(vector)



    print(len(avg_w2v_vectors))

    print(len(avg_w2v_vectors[0]))

    return avg_w2v_vectors
#creating avgw2v essay and title vectors

avgw2v_title_train = fitAvgW2V(x_train['processed title'])

avgw2v_text_train = fitAvgW2V(x_train['processed text'])

avgw2v_title_test = fitAvgW2V(x_test['processed title'])

avgw2v_text_test = fitAvgW2V(x_test['processed text'])
x_train.drop(['subject'], axis = 1, inplace=True)

x_test.drop(['subject'], axis = 1, inplace=True)
#function to do batch prediction

def batch_predict(clf, data):

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs



    y_data_pred = []

    tr_loop = data.shape[0] - data.shape[0]%1000

    # consider you X_tr shape is 49041, then your tr_loop will be 49041 - 49041%1000 = 49000

    # in this for loop we will iterate unti the last 1000 multiplier

    for i in range(0, tr_loop, 1000):

        y_data_pred.extend(clf.predict_proba(data[i:i+1000])[:,1])

    # we will be predicting for the last data points

    if data.shape[0]%1000 !=0:

        y_data_pred.extend(clf.predict_proba(data[tr_loop:])[:,1])

    

    return y_data_pred
# we are writing our own function for predict, with defined thresould

# we will pick a threshold that will give the least fpr

def find_best_threshold(threshould, fpr, tpr):

    t = threshould[np.argmax(tpr*(1-fpr))]

    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high

    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))

    return t



def predict_with_best_t(proba, threshould):

    predictions = []

    for i in proba:

        if i>=threshould:

            predictions.append(1)

        else:

            predictions.append(0)

    return predictions
#function to do grid search cross validation

def doGridSearch(X_tr, y_train, dense=False):

    neigh = KNeighborsClassifier(n_jobs=-1)

    parameters = {'n_neighbors':[11, 21, 31, 41, 51]}

    clf = GridSearchCV(neigh, parameters, cv=3, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=10)

    clf.fit(X_tr, y_train)



    results = pd.DataFrame.from_dict(clf.cv_results_)

    results = results.sort_values(['param_n_neighbors'])



    train_auc= results['mean_train_score']

    train_auc_std= results['std_train_score']

    cv_auc = results['mean_test_score'] 

    cv_auc_std= results['std_test_score']

    K =  results['param_n_neighbors']



    plt.plot(K, train_auc, label='Train AUC')



    plt.plot(K, cv_auc, label='CV AUC')



    plt.scatter(K, train_auc, label='Train AUC points')

    plt.scatter(K, cv_auc, label='CV AUC points')





    plt.legend()

    plt.xlabel("K: hyperparameter")

    plt.ylabel("AUC")

    plt.title("Hyper parameter Vs AUC plot")

    plt.grid()

    plt.show()

    

    return clf
#function to plot auc and heat maps of confusion matrix

def plotAucAndHeatmap(neighbors, X_tr, X_te, y_train, y_test):

    neigh = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=-1)

    neigh.fit(X_tr, y_train)

    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class

    # not the predicted outputs



    y_train_pred = batch_predict(neigh, X_tr)    

    y_test_pred = batch_predict(neigh, X_te)



    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)

    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)



    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))

    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))

    plt.legend()

    plt.xlabel("FPR")

    plt.ylabel("TPR")

    plt.title("ERROR PLOTS")

    plt.grid()

    plt.show()

    

    print("="*100)

    best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

    print("Train confusion matrix")

    print(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)))

    print("Test confusion matrix")

    print(confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))

    print("="*100)



    

    plotheatMap(confusion_matrix(y_train, predict_with_best_t(y_train_pred, best_t)), confusion_matrix(y_test, predict_with_best_t(y_test_pred, best_t)))

    return str(auc(train_fpr, train_tpr)), str(auc(test_fpr, test_tpr))
# subplot seaborn : https://stackoverflow.com/a/41384984/8363466

#confusion matrix heat map : https://seaborn.pydata.org/generated/seaborn.heatmap.html

#plot confusion matrix of test and train

def plotheatMap(confusion_matrix_train, confusion_matrix_test):

    fig, (ax1, ax2) = plt.subplots(1,2)

    fig.set_figheight(5)

    fig.set_figwidth(15)

    

    confusion_train_bow = pd.DataFrame(confusion_matrix_train)

    sns.heatmap(confusion_train_bow, annot=True, fmt='d', ax=ax1)

    ax1.set_title("Train Confusion matrix")

    ax1.set(xlabel='Actual', ylabel='Predicted')

    

    confusion_test_bow = pd.DataFrame(confusion_matrix_test)

    sns.heatmap(confusion_test_bow, annot=True, fmt='d', ax=ax2)

    ax2.set_title("Test Confusion matrix")

    ax2.set(xlabel='Actual', ylabel='Predicted')
#creating train and test data for KNN brute force on AVG W2V

X_tr = hstack((avgw2v_title_train, avgw2v_text_train, x_train_cat)).tocsr()

X_te = hstack((avgw2v_title_test, avgw2v_text_test, x_test_cat)).tocsr()



print(X_tr.shape, y_train.shape)

print(X_te.shape, y_test.shape)

print("="*100)
%%time

#do grid search to find best K

avgw2v_clf = doGridSearch(X_tr, y_train)
#randomizedsearchcv sklearn: https://www.youtube.com/watch?v=Gol_qOgRqfA

print(avgw2v_clf.best_score_)

print(avgw2v_clf.best_params_)
print("Number of neighbours as per GridSearchCV : ",avgw2v_clf.best_params_['n_neighbors'])
#plot auc and confusion matrix

avgw2v_train_auc, avgw2v_test_auc = plotAucAndHeatmap(avgw2v_clf.best_params_['n_neighbors'], X_tr, X_te, y_train, y_test)