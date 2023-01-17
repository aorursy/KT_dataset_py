import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import preprocessing, feature_extraction, model_selection, linear_model

import seaborn as sns

sns.set()
review_text_train = pd.read_csv('../input/review_text_train.csv', index_col = False, delimiter = ',', header=0)

review_text_test = pd.read_csv('../input/review_text_test.csv', index_col = False, delimiter = ',', header=0)

review_meta_train = pd.read_csv('../input/review_meta_train.csv', index_col = False, delimiter = ',', header=0)

review_meta_test = pd.read_csv('../input/review_meta_test.csv', index_col = False, delimiter = ',', header=0)
print('review_text_train :\t', str(review_text_train.shape))

print('review_text_test :\t', str(review_text_test.shape))

print('review_meta_train :\t', str(review_meta_train.shape))

print('review_meta_test :\t', str(review_meta_test.shape))
review_meta_train
review_text_train
print('review_text_train :\n', str(review_text_train.isna().sum()), "\n**********\n")

print('review_text_test :\n', str(review_text_test.isna().sum()), "\n**********\n")

print('review_meta_train :\n', str(review_meta_train.isna().sum()), "\n**********\n")

print('review_meta_test :\n', str(review_meta_test.isna().sum()), "\n**********\n")
# Train DataFrame

df_train = review_text_train.copy()
df_train['vote_funny'] = review_meta_train.vote_funny

df_train['vote_cool'] = review_meta_train.vote_cool

df_train['vote_useful'] = review_meta_train.vote_useful

df_train['rating'] = review_meta_train.rating

df_train
df_train.isna().sum()
df_test = review_text_test.copy()

df_test['vote_funny'] = review_meta_test.vote_funny

df_test['vote_cool'] = review_meta_test.vote_cool

df_test['vote_useful'] = review_meta_test.vote_useful

df_test
df_test.isna().sum()
# The re module raises the exception re. error if an error occurs while compiling or using a regular expression.

import re

# This module contains a number of functions to process standard Python strings.

import string
def clean_text_round_1(text):

    text = text.lower()

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Removes Punctuation

    text = re.sub('\w*\d\w*','',text) # Removes Alphanumeric Characters    

    return text

round_1 = lambda x: clean_text_round_1(x)
data_review_cleaned = pd.DataFrame(df_train.review.apply(round_1))
data_review_cleaned
rev_string = " "

for i in range(len(data_review_cleaned)):

    rev_string +=data_review_cleaned['review'][i]+" "

#rev_string
def clean_text_round_2(text):

    text = re.sub('\n', ' ', text)

    text = re.sub('[''""..._]', '', text)

    return text

round2 = lambda x: clean_text_round_2(x)
data_review_cleaned = pd.DataFrame(data_review_cleaned.review.apply(round2))
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(stop_words = 'english')

data_cv = cv.fit_transform(data_review_cleaned.review)

data_dtm = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())

data_dtm.index = data_review_cleaned.index

data_dtm
data_dtm_transposed = data_dtm.T
data_dtm_transposed
data_review_cleaned_test = pd.DataFrame(df_test.review.apply(round_1))

data_review_cleaned_test = pd.DataFrame(data_review_cleaned_test.review.apply(round2))
from textblob import TextBlob



# Polarity is how positive or Negative the expressioin is

# Subjectivity is how Factual or Opinionated the expression is

pol_test = lambda x: TextBlob(x).sentiment.polarity

sub_test = lambda x: TextBlob(x).sentiment.subjectivity



df_test['polarity'] = df_test['review'].apply(pol_test)

df_test['subjectivity'] = df_test['review'].apply(sub_test)
df_test
top_words = {}

for o in data_dtm_transposed.columns:

    top = data_dtm_transposed[o].sort_values(ascending = False).head(50)

    top_words[o] = list(zip(top.index, top.values))

#top_words
data = data_dtm_transposed
from collections import Counter



words = []

for user_id in data.columns:

    top = [word for (word,count) in top_words[user_id]]

    for t in top:

        words.append(t)
common_words_count  = Counter(words).most_common()
df_common_words = pd.DataFrame(common_words_count, columns = ['word', 'count'])

df_common_words
# plot

fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(15, 12)

ax = sns.barplot(x='count', y='word', data=df_common_words[:30])

ax.set_title('Top 30 Words in the Corpus', size = 24)

ax.set_xlabel('Count', size = 20)

ax.set_ylabel("Words", size = 20)



fig.savefig('top_30_words.png')
from wordcloud import WordCloud
from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer



#Adding the Stop Words

stop_words = text.ENGLISH_STOP_WORDS
wc = WordCloud(width = 800, height = 400, stopwords = stop_words, background_color = 'white', colormap = 'Dark2', 

               max_font_size = 170, random_state = 45)
data_for_wc = pd.DataFrame()

data_for_wc['review'] = data_review_cleaned['review']

data_for_wc = data_for_wc.reset_index(drop = True)
#data_for_wc
#Extracting text of all reviews into one string for WordCloud

text_wc = ' '

for i in range(len(data_for_wc)):

    text_wc += data_for_wc['review'][i]
cloud = wc.generate(text_wc)

plt.figure(figsize=(30,15))

plt.title('Reviews Text WordCloud', fontsize = 30)

plt.imshow(wc, interpolation = 'bilinear')

plt.axis("off")
cloud.to_file('word_cloud.png')
df_train['review'] = data_review_cleaned['review']

df_train
data = df_train
from textblob import TextBlob
pol = lambda x: TextBlob(x).sentiment.polarity

sub = lambda x: TextBlob(x).sentiment.subjectivity



data['polarity'] = data['review'].apply(pol)

data['subjectivity'] = data['review'].apply(sub)
plt.figure(figsize = (10,10))

scatter = plt.scatter(data['polarity'], data['subjectivity'], c= data['rating'], cmap = 'winter', alpha = 0.4)

plt.xlabel('<-- Negative ------------------------Polarity-------------------------- Positive -->')

plt.ylabel('<-- Factual --------------------------Subjectivity------------------------- Opinionated -->')

plt.title('Polarity v/s Subjectivity', size  = 20)

plt.colorbar()

plt.show



plt.savefig('polarity_vs_subjectivity_sentiment', dpi = 150)
positive_values = 0

negative_values = 0



for i in range(len(data['polarity'])):

    if data['polarity'][i]<0.5:

        negative_values +=1

    else:

        positive_values +=1
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

polarity_prop = ['Positive Reviews', 'Negative Reviews']

polarity_vals = [positive_values, negative_values]

ax.bar(polarity_prop, polarity_vals)

plt.show()



plt.savefig('pos_vs_neg_reviews.png', dpi = 150)
factual_values = 0

opinionated_values = 0



for i in range(len(data['subjectivity'])):

    if data['subjectivity'][i]<0.5:

        factual_values +=1

    else:

        opinionated_values +=1
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

subjectivity_prop = ['Facts', 'Opinions']

subjectivity_vals = [factual_values, opinionated_values]

ax.bar(subjectivity_prop, subjectivity_vals)

plt.show()



plt.savefig('facts_vs_opinions.png', dpi = 150)
cor = df_train.loc[:,["vote_funny","vote_cool",'vote_useful',"polarity", "subjectivity","rating"]]

correlation_map = sns.clustermap(cor.corr(), center=0, cmap="vlag",

               linewidths=.75, figsize=(10, 5))



correlation_map.savefig('feature_correlation.png', dpi = 150)
df_train.columns.values
cols = ['review', 'vote_funny', 'vote_cool', 'vote_useful','polarity', 'subjectivity','rating']

df_train = df_train[cols]

df_train
df_train['rating'].unique()
targets = df_train['rating']

inputs = df_train.drop(['review','rating'], axis = 1)
plt.scatter(inputs['vote_funny'], targets)

plt.xlabel('vote_funny', size = 20)

plt.ylabel('rating', size = 20)

plt.title('vote_funny v/s rating', size = 24)

plt.show()
fig, axes = plt.subplots(nrows=3, ncols=2)

fig.set_figheight(8)

fig.set_figwidth(15)



axes[0,0].scatter(inputs['vote_funny'], targets)

axes[0,0].set_title('vote_funny v/s rating', size = 15)

axes[0,0].set_xlabel('vote_funny', size = 10)

axes[0,0].set_ylabel('rating', size = 10)



axes[0,1].scatter(inputs['vote_cool'], targets)

axes[0,1].set_title('vote_cool v/s rating', size = 15)

axes[0,1].set_xlabel('vote_cool', size = 10)

axes[0,1].set_ylabel('rating', size = 10)



axes[1,0].scatter(inputs['vote_useful'], targets)

axes[1,0].set_title('vote_useful v/s rating', size = 15)

axes[1,0].set_xlabel('vote_useful', size = 10)

axes[1,0].set_ylabel('rating', size = 10)



axes[1,1].scatter(inputs['polarity'], targets)

axes[1,1].set_title('polarity v/s rating', size = 15)

axes[1,1].set_xlabel('polarity', size = 10)

axes[1,1].set_ylabel('rating', size = 10)



axes[2,0].scatter(inputs['subjectivity'], targets)

axes[2,0].set_title('subjectivity v/s rating', size = 15)

axes[2,0].set_xlabel('subjectivity', size = 10)

axes[2,0].set_ylabel('rating', size = 10)



fig.tight_layout()



plt.savefig('logistic_regression_data_distribution_scatter_plots.png', dpi = 150)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(inputs, targets, test_size = 0.1, random_state = 365)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
log_reg = LogisticRegression()

log_results  = log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
class_names = [1,3,5]

fig, ax = plt.subplots()

tick_marks = np.arange(2)

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion Matrix', y=1.1, size = 24)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



fig.savefig('logistic_regression_confusion_matrix.png', dpi = 150)
print("Accuracy: "+str((metrics.accuracy_score(y_test, y_pred)*100).round(3))+"%")
y_pred_logistic_regression = log_reg.predict(df_test.drop(['review'], axis = 1))
pd.DataFrame(y_pred_logistic_regression).to_csv("logistic_regression_predictions.csv")
#Import knearest neighbors Classifier model

from sklearn.neighbors import KNeighborsClassifier



#Create KNN Classifier

knn = KNeighborsClassifier(n_neighbors=125)



#Train the model using the training sets

knn.fit(x_train, y_train)



#Predict the response for test dataset

y_pred = knn.predict(x_test)


#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# Calculating the Model Accuracy

print("Accuracy:" + str((metrics.accuracy_score(y_test, y_pred)*100).round(3))+"%")
y_pred_KNN_predictions = knn.predict(df_test.drop(['review'], axis = 1))
pd.DataFrame(y_pred_KNN_predictions).to_csv("KNN_predictions.csv")
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(x_train, y_train);
pred = model.predict(x_test)
acc = model.score(x_test,y_test)

print("Accuracy(Gaussian Naive Bayes) = " + str((acc*100).round(3))+"%")
from sklearn.naive_bayes import BernoulliNB

from sklearn import metrics

from sklearn.metrics import accuracy_score



BernNB = BernoulliNB(binarize = False)

BernNB.fit(x_train, y_train)



y_expected = y_test

y_pred_ber = BernNB.predict(x_test)



print("Accuracy (Bernoulli Naive Bayes): "+ str(((accuracy_score(y_expected, y_pred))*100).round(3)) + "%")
from sklearn.naive_bayes import MultinomialNB



model_mn = MultinomialNB()

model_mn.fit(x_train.drop(['polarity'], axis = 1), y_train)





accuracy_multinomial_nb = model_mn.score(x_test.drop(['polarity'], axis = 1), y_test)

print("Accuracy (Multinomial Naive Bayes): "+ str((accuracy_multinomial_nb*100).round(3)) + "%")
y_pred = BernNB.predict(df_test.drop(['review'], axis = 1))
pd.DataFrame(y_pred).to_csv("Bernoulli_Naive_Bayes_Predictions.csv")
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 1515, criterion = 'gini', random_state = 45)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

cm
class_names = [1,3,5]

fig, ax = plt.subplots()

tick_marks = np.arange(1)

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="RdBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion Matrix', y=1.1, size = 24)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy = "+ str(((model.score(x_test,y_test))*100).round(3))+"%")
y_pred = model.predict(df_test.drop(['review'], axis = 1))
pd.DataFrame(y_pred).to_csv("Random_Forest_Classifier_predictions.csv")
from sklearn import svm
#Create a svm Classifier

clf = svm.SVC(kernel='linear') # Linear Kernel



#Train the model using the training sets

clf.fit(x_train, y_train)



#Predict the response for test dataset

y_pred = clf.predict(x_test)
print("Accuracy:",str(((metrics.accuracy_score(y_test, y_pred))*100).round(3)) + "%")
y_pred_svm = clf.predict(df_test.drop(['review'], axis = 1))

pd.DataFrame(y_pred).to_csv("SVM_predictions.csv")
#import numpy as np

import tensorflow as tf

from sklearn import preprocessing
unscaled_inputs_all = df_train[['vote_funny', 'vote_cool', 'vote_useful','polarity', 'subjectivity']]

targets_all = df_train['rating']
scaled_inputs = preprocessing.scale(unscaled_inputs_all)
shuffled_indices = np.arange(scaled_inputs.shape[0])

np.random.shuffle(shuffled_indices)
shuffled_inputs = scaled_inputs[shuffled_indices]

shuffled_targets = targets_all[shuffled_indices]
samples_count = shuffled_inputs.shape[0]

samples_count
train_samples_count = int(0.8 * samples_count)

validation_samples_count = int(0.2 * samples_count)
train_inputs = shuffled_inputs[:train_samples_count]

train_targets = shuffled_targets[:train_samples_count]



validation_inputs = shuffled_inputs[train_samples_count:train_samples_count + validation_samples_count]

validation_targets = shuffled_targets[train_samples_count:train_samples_count + validation_samples_count]
train_inputs = shuffled_inputs[:train_samples_count]

train_targets = shuffled_targets[:train_samples_count]



validation_inputs = shuffled_inputs[train_samples_count:train_samples_count + validation_samples_count]

validation_targets = shuffled_targets[train_samples_count:train_samples_count + validation_samples_count]
np.savez('rating_train_data', inputs = train_inputs, targets = train_targets)

np.savez('rating_validation_data', inputs = validation_inputs, targets = validation_targets)
npz = np.load('/kaggle/working/rating_train_data.npz')

train_inputs, train_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)



npz = np.load('/kaggle/working/rating_validation_data.npz')

validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
input_size = 5

output_size = 3



hidden_layer_size = 100
model = tf.keras.Sequential([

    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),



    tf.keras.layers.Dense(hidden_layer_size, activation='softmax')

])



model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



batch_size = 55

max_epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)
history = model.fit(train_inputs, train_targets,

         batch_size = batch_size,

         epochs= max_epochs,

         callbacks = [early_stopping],

         validation_data = (validation_inputs, validation_targets),

         verbose = 1)
model.save_weights("restaurant_rating_model.h5")
plt.figure(figsize = (20,5))

plt.plot(history.history['loss'], color = 'red', label = 'Training Loss')

plt.plot(history.history['val_loss'], color = 'blue', label = 'Validation Loss')

plt.title('Training Loss v/s Validation Loss', size = 20)

plt.legend()

plt.show()
plt.figure(figsize = (20,5))

plt.plot(history.history['accuracy'], color = 'red', label = 'Training Accuracy')

plt.plot(history.history['val_accuracy'], color = 'blue', label = 'Validation Accuracy')

plt.title('Training Accuracy v/s Validation Accuracy', size = 20)

plt.legend()

plt.show()
df_test_dnn = df_test.drop(['review'], axis = 1)

test_inputs_unscaled = df_test_dnn

np.savez('rating_test_data', inputs = test_inputs_unscaled)
npz = np.load('/kaggle/working/rating_test_data.npz')

test_inputs = npz['inputs'].astype(np.float)
predictions = model.predict(df_test_dnn)

pd.DataFrame(predictions).to_csv("DNN_predictions.csv")