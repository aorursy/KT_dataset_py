import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import scikitplot as skplt

from keras import callbacks
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10)

plt.rc('figure', figsize=(10, 7))

num_epoch = 5
data = pd.read_csv('../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')
data.drop(columns="Unnamed: 0", axis=1, inplace=True)
data
department_list = data['Department Name'].dropna().unique()
department_list = [x.lower() for x in department_list]
department_list
class_list = data['Class Name'].dropna().unique()
class_list = [x.lower() for x in class_list]
class_list
department_and_class = np.concatenate((department_list, class_list, ['dress', 'petite', 'petit', 'skirt', 'shirt', 'jacket', 'intimate', 'blouse', 'coat', 'sweater']), axis=0)
department_and_class
review_data = data[['Review Text','Recommended IND']]
review_data
review_data.isnull().sum().sort_values()
review_data.dropna(axis=0,inplace=True)
review_data
#import for test train split and vect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(data):
    tfidf_vectorizer =TfidfVectorizer(min_df=3,  max_features=None, 
             analyzer='word', use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')


    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer
from sklearn.decomposition import  TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches


def plot_LSA(test_data, test_labels):
        #reduce into 2 dimensions using svd 
        lsa = TruncatedSVD(n_components=2)
        #fits to the train data
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]
        colors = ['orange','blue','blue']
        if plt:
            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
            red_patch = mpatches.Patch(color='orange', label='Recommended IND = 0')
            blue_patch = mpatches.Patch(color='blue', label='Recommended IND = 1')
            plt.legend(handles=[red_patch, blue_patch], prop={'size': 12})
X = review_data["Review Text"]
y = review_data["Recommended IND"]

# Create sequence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_data['Review Text'])
vocabulary_size = len(tokenizer.word_index) + 1
print(vocabulary_size)

# 限制最长长度为70，过长截断，过短就在后方（post）补齐
max_length = 70

sequences = tokenizer.texts_to_sequences(X)
features = pad_sequences(sequences, maxlen=max_length, padding='post')
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# if you don't have stopwords and have some error, please use the download code bollow!
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
### Text Normalizing function. Part of the following function was taken from this link. 
def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    text = [w for w in text if not w in stop_words]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
#     text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text
review_data['Review Text'] = review_data['Review Text'].map(lambda x: clean_text(x))
review_data
from keras.utils import to_categorical

X = review_data["Review Text"]
y = review_data["Recommended IND"]

# Create sequence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_data['Review Text'])
vocabulary_size = len(tokenizer.word_index) + 1
print(vocabulary_size)
sequences = tokenizer.texts_to_sequences(review_data['Review Text'])
np.max([len(x) for x in sequences])
# 限制最长长度为70，过长截断，过短就在后方（post）补齐
max_length = 70
padded_features = pad_sequences(sequences, maxlen=max_length, padding='post')
plot_LSA(padded_features, y)
plt.show()
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

def plot_roc(n_classes, y_test, y_score, title, class_name_list):
    # Plot linewidth.
    lw = 2

    y_test = sentiment_test[1]
    y_score = test_score
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_name_list[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
from sklearn import model_selection

X_train, X_val, y_train, y_val = model_selection.train_test_split(review_data['Review Text'], review_data['Recommended IND'], test_size=0.2, random_state=666)
X_test, X_val, y_test, y_val = model_selection.train_test_split(X_val, y_val, test_size=0.5, random_state=888)
print(len(X_train))
print(len(X_val))
print(len(X_test))
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler()
ros = RandomOverSampler()
X_over, X_cat, y_over, y_cat = model_selection.train_test_split(X_train, y_train, test_size=0.2, random_state=888)
X_over, X_under, y_over, y_under = model_selection.train_test_split(X_over, y_over, test_size=0.75, random_state=888)
print(X_under.shape, X_over.shape, y_under.shape, y_over.shape, X_cat.shape, y_cat.shape)
ros_sequences = tokenizer.texts_to_sequences(X_over)
ros_features = pad_sequences(ros_sequences, maxlen=max_length, padding='post')
train_X_ros, train_y_ros = ros.fit_sample(ros_features, y_over)
print(train_X_ros.shape, train_y_ros.shape)
train_y_ros.value_counts()
rus_sequences = tokenizer.texts_to_sequences(X_under)
rus_features = pad_sequences(rus_sequences, maxlen=max_length, padding='post')
train_X_rus, train_y_rus = rus.fit_sample(rus_features, y_under)
print(train_X_rus.shape, train_y_rus.shape)
train_y_rus.value_counts()
print(X_cat.shape, y_cat.shape)
y_cat.value_counts()
y_cat.value_counts()
cat_0_idx = y_cat[y_cat == 0]
cat_0_idx = list(cat_0_idx.keys())
cat_1_idx = y_cat[y_cat == 1]
cat_1_idx = list(cat_1_idx.keys())
X_cat_0 = X_cat[cat_0_idx]
X_cat_1 = X_cat[cat_1_idx]
count_0 = len(cat_0_idx)
import random
new_X_0 = []
for idx in cat_0_idx:
    cur = X_cat_0[idx]
    p = random.randint(0, 1)
    cur_idx = len(cur) // 2
    cur = cur[:cur_idx] if p == 0 else cur[cur_idx:]
    new_X_0.append(cur)
new_X_0.extend(list(X_cat_0.values))

new_X_1 = []
for idx in cat_1_idx:
    cur = X_cat_1[idx]
    p = random.randint(0, 1)
    cur_idx = len(cur) // 2
    cur = cur[:cur_idx] if p == 0 else cur[cur_idx:]
    new_X_1.append(cur)
new_X_1 = random.sample(new_X_1, count_0)
new_X_1.extend(random.sample(list(X_cat_1.values), count_0))
print(len(new_X_0))
print(len(new_X_1))
len(new_X_0 + new_X_1)
X_cat = pd.Series(new_X_0 + new_X_1)
y_cat = pd.Series([0] * count_0 * 2 + [1] * count_0 * 2)
cat_sequences = tokenizer.texts_to_sequences(X_cat)
cat_features = pad_sequences(cat_sequences, maxlen=max_length, padding='post')
cat_features.shape
cat_features
train_X_rus
train_X_ros
features_all = np.concatenate((cat_features, train_X_rus, train_X_ros))
features_all.shape
y_all = y_cat.append(train_y_rus).append(train_y_ros)
labels_all = to_categorical(y_all)
labels_all[0]
print(labels_all.shape)
plot_LSA(features_all, y_all)
plt.show()
val_sequences = tokenizer.texts_to_sequences(X_val)
val_features = pad_sequences(val_sequences, maxlen=max_length, padding='post')
val_labels = to_categorical(y_val)
print(val_features.shape, val_labels.shape)

test_sequences = tokenizer.texts_to_sequences(X_test)
test_features = pad_sequences(test_sequences, maxlen=max_length, padding='post')
test_labels = to_categorical(y_test)
print(test_features.shape, test_labels.shape)
model = Sequential()

e = Embedding(vocabulary_size, 100, input_length=max_length, trainable=True)
model.add(e)
model.add(Bidirectional(LSTM(128, dropout=0.5, return_sequences=True)))
model.add(Bidirectional(LSTM(256, dropout=0.5)))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(features_all, labels_all, epochs=num_epoch, batch_size=256, verbose=1,
          validation_data=(val_features, val_labels), shuffle=True)

score = model.evaluate(test_features, test_labels, verbose=1)

print('loss : {}, acc : {}'.format(score[0], score[1]))
test_score = model.predict(test_features)
test_predictions = np.argmax(test_score, axis=1)

class_names = ['(0) Not recommended class', '(1) Recommended class']
report = classification_report(np.argmax(test_labels, axis=1), test_predictions, target_names=class_names)
matrix = pd.DataFrame(confusion_matrix(y_true=np.argmax(test_labels, axis=1), y_pred=test_predictions), 
                                        index=class_names, columns=class_names)
print(matrix)
print(report)
f1_score(np.argmax(test_labels, axis=1), test_predictions, average='micro')   
skplt.metrics.plot_roc(np.argmax(test_labels, axis=1), model.predict_proba(test_features),
                      title='ROC Curves - hyper') 
model = Sequential()

class_weight = {0: 5, 1: 1}
e = Embedding(vocabulary_size, 100, input_length=max_length, trainable=True)
model.add(e)
model.add(Bidirectional(LSTM(128, dropout=0.5, return_sequences=True)))
model.add(Bidirectional(LSTM(256, dropout=0.5)))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(features_all, labels_all, epochs=num_epoch, batch_size=256, verbose=1,
          validation_data=(val_features, val_labels), shuffle=True, class_weight=class_weight)

score = model.evaluate(test_features, test_labels, verbose=1)

print('loss : {}, acc : {}'.format(score[0], score[1]))


test_score = model.predict(test_features)
test_predictions = np.argmax(test_score, axis=1)

class_names = ['(0) Not recommended class', '(1) Recommended class']
report = classification_report(np.argmax(test_labels, axis=1), test_predictions, target_names=class_names)
matrix = pd.DataFrame(confusion_matrix(y_true=np.argmax(test_labels, axis=1), y_pred=test_predictions), 
                                        index=class_names, columns=class_names)
print(matrix)
print(report)
f1_score(np.argmax(test_labels, axis=1), test_predictions, average='micro')   
skplt.metrics.plot_roc(np.argmax(test_labels, axis=1), model.predict_proba(test_features),
                      title='ROC Curves - hyper') 
model = Sequential()

class_weight = {0: 10, 1: 1}
e = Embedding(vocabulary_size, 100, input_length=max_length, trainable=True)
model.add(e)
model.add(Bidirectional(LSTM(128, dropout=0.5, return_sequences=True)))
model.add(Bidirectional(LSTM(256, dropout=0.5)))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(features_all, labels_all, epochs=num_epoch, batch_size=256, verbose=1,
          validation_data=(val_features, val_labels), shuffle=True, class_weight=class_weight)

score = model.evaluate(test_features, test_labels, verbose=1)

print('loss : {}, acc : {}'.format(score[0], score[1]))


test_score = model.predict(test_features)
test_predictions = np.argmax(test_score, axis=1)

class_names = ['(0) Not recommended class', '(1) Recommended class']
report = classification_report(np.argmax(test_labels, axis=1), test_predictions, target_names=class_names)
matrix = pd.DataFrame(confusion_matrix(y_true=np.argmax(test_labels, axis=1), y_pred=test_predictions), 
                                        index=class_names, columns=class_names)
print(matrix)
print(report)
f1_score(np.argmax(test_labels, axis=1), test_predictions, average='micro')   
skplt.metrics.plot_roc(np.argmax(test_labels, axis=1), model.predict_proba(test_features),
                      title='ROC Curves - hyper') 