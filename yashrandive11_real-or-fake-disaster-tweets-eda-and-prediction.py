import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing, feature_extraction, model_selection, linear_model

import seaborn as sns

sns.set()
unprocessed_csv_train = pd.read_csv('../input/nlp-getting-started/train.csv')

unprocessed_csv_test = pd.read_csv('../input/nlp-getting-started/test.csv')
unprocessed_csv_train.head(20)
df_1 = pd.DataFrame()



df_2 = pd.DataFrame()
df_1['user_id'],df_1['text'], df_1['target'] = unprocessed_csv_train['id'], unprocessed_csv_train['text'], unprocessed_csv_train['target']

df_1.set_index('user_id', inplace = True)

df_1
import re

import string
def clean_text_round_1(text):

    text = text.lower()

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\w*\d\w*','',text)

    text = re.sub('love,','',text)

    

    return text

round_1 = lambda x: clean_text_round_1(x)

    
data_cleaned = pd.DataFrame(df_1.text.apply(round_1))
data_cleaned
def clean_text_round_2(text):

    text = re.sub('\n', ' ', text)

    text = re.sub('[''""..._]', '', text)

    text = re.sub('\\x89ûª', '', text)

    text = re.sub('\\x89ûò', '', text)

    text = re.sub('\\x89û', '', text)

    text = re.sub('\\x89', '', text)

    text = re.sub('\\x89ã¢', '', text)

    text = re.sub('ï', '', text)

    text = re.sub('÷', '', text)

    text = re.sub('\\x9d', '', text)

    

    return text

round2 = lambda x: clean_text_round_2(x)
data_cleaned = pd.DataFrame(data_cleaned.text.apply(round2))
data_cleaned
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(stop_words = 'english')

data_cv = cv.fit_transform(data_cleaned.text)

data_dtm = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())

data_dtm.index = data_cleaned.index

data_dtm
data_dtm_transposed = data_dtm.T
data_dtm_transposed
top_words = {}

for o in data_dtm_transposed.columns:

    top = data_dtm_transposed[o].sort_values(ascending = False).head(50)

    top_words[o] = list(zip(top.index, top.values))

top_words
data = data_dtm_transposed
from collections import Counter



words = []

for user_id in data.columns:

    top = [word for (word,count) in top_words[user_id]]

    for t in top:

        words.append(t)

words
Counter(words).most_common()
add_stop_words = [word for word, count in Counter(words).most_common() if count >=191]

add_stop_words
from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer



#Adding the Stop Words

stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)



#Recreating the Document Term Matrix

cv = CountVectorizer(stop_words = stop_words)

data_cv = cv.fit_transform(data_cleaned.text)

data_stop = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())

data_stop.index = data_cleaned.index



data_stop
from wordcloud import WordCloud
wc = WordCloud(width = 800, height = 400, stopwords = stop_words, background_color = 'white', colormap = 'Dark2', 

               max_font_size = 170, random_state = 45)
data_for_wc = pd.DataFrame()

data_for_wc['text'] = data_cleaned['text']

data_for_wc = data_for_wc.reset_index(drop = True)
data_for_wc
#Extracting text of all tweets into one string for WordCloud

text_wc = ' '

for i in range(len(data_for_wc)):

    text_wc += data_for_wc['text'][i]

text_wc
wc.generate(text_wc)

plt.figure(figsize=(15,15))

plt.title('Tweets Text WordCloud', fontsize = 30)

plt.imshow(wc, interpolation = 'bilinear')

plt.axis("off")
data = data_cleaned

data
from textblob import TextBlob
# Polarity is how positive or Negative the expressioin is

# Subjectivity is how Factual or Opinionated the expression is

pol = lambda x: TextBlob(x).sentiment.polarity

sub = lambda x: TextBlob(x).sentiment.subjectivity



data['polarity'] = data['text'].apply(pol)

data['subjectivity'] = data['text'].apply(sub)
data['target'] = df_1['target']

data
scatter = plt.scatter(data['polarity'], data['subjectivity'], c= data['target'], cmap = 'winter', alpha = 0.4)

plt.xlabel('<----- Negative --------------Polarity---------------- Positive ----->')

plt.ylabel('<----- Factual ------Subjectivity----- Opinionated ----->')

plt.colorbar()

plt.show
subjectivity_df = pd.DataFrame()

subjectivity_df['subjectivity'] = data['subjectivity']

subjectivity_df = subjectivity_df.reset_index(drop = True)



polarity_df = pd.DataFrame()

polarity_df['polarity'] = data['polarity']

polarity_df = polarity_df.reset_index(drop = True)
factual_values = 0

opinionated_values = 0



for i in range(len(subjectivity_df['subjectivity'])):

    if subjectivity_df['subjectivity'][i]<0.5:

        factual_values +=1

    else:

        opinionated_values +=1
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

subjectivity_prop = ['Facts', 'Opinions']

subjectivity_vals = [factual_values, opinionated_values]

ax.bar(subjectivity_prop, subjectivity_vals)

plt.show()
positive_values = 0

negative_values = 0



for i in range(len(polarity_df['polarity'])):

    if polarity_df['polarity'][i]<0.5:

        negative_values +=1

    else:

        positive_values +=1
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

polarity_prop = ['Positive Tweets', 'Negative Tweets']

polarity_vals = [positive_values, negative_values]

ax.bar(polarity_prop, polarity_vals)

plt.show()
data
unscaled_x_train = data[['polarity', 'subjectivity']]

unscaled_y_train = data['target']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test = train_test_split(unscaled_x_train, unscaled_y_train, test_size=0.1, random_state = 365)
import statsmodels.api as sm

x = sm.add_constant(x_train)

reg_log = sm.Logit(y_train,x_train)

results_log = reg_log.fit()

results_log.summary()
def confusion_matrix(data,actual_values,model):

        

        

        pred_values = model.predict(data)

        bins=np.array([0,0.5,1])

        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]

        accuracy = (cm[0,0]+cm[1,1])/cm.sum()

        string = 'Accuracy is ' + repr(accuracy*100)+' %'

        return cm, string, pred_values
confusion_matrix(x_train,y_train,results_log)
data = np.loadtxt('../input/pretraincsvfinal/pre_train_csv_final.csv', delimiter = ',')
unscaled_inputs_all = data[:,0:-1]

targets_all = data[:, -1]
scaled_inputs = preprocessing.scale(unscaled_inputs_all)
shuffled_indices = np.arange(scaled_inputs.shape[0])

np.random.shuffle(shuffled_indices)



shuffled_inputs = scaled_inputs[shuffled_indices]

shuffled_targets = targets_all[shuffled_indices]
samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.9 * samples_count)

validation_samples_count = int(0.1 * samples_count)
train_inputs = shuffled_inputs[:train_samples_count]

train_targets = shuffled_targets[:train_samples_count]





validation_inputs = shuffled_inputs[train_samples_count : train_samples_count + validation_samples_count]

validation_targets = shuffled_targets[train_samples_count : train_samples_count + validation_samples_count]
np.savez('disaster_tweets_train', inputs = train_inputs, targets = train_targets)

np.savez('disaster_tweets_validation', inputs = validation_inputs, targets = validation_targets)
npz = np.load('/kaggle/working/disaster_tweets_train.npz')



train_inputs = npz['inputs'].astype(np.float)

train_targets = npz['targets'].astype(np.int)



npz = np.load('/kaggle/working/disaster_tweets_validation.npz')



validation_inputs = npz['inputs'].astype(np.float)

validation_targets = npz['targets'].astype(np.int)
import tensorflow as tf
input_size = 2

output_size = 2

hidden_layer_size = 300



model = tf.keras.Sequential([

    #tf.keras.layers.Dense(input_size, activation = 'relu'),

    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu' ),

    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu' ),

    tf.keras.layers.Dense(output_size, activation = 'softmax')

])



model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])



batch_size = 100

max_epochs = 100



early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)



model.fit(train_inputs, train_targets, batch_size = batch_size, epochs = max_epochs,

          callbacks = [early_stopping],

         validation_data = (validation_inputs, validation_targets),

         verbose = 2)
dnn_pred =model.predict_classes(train_inputs, verbose=1)
from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB



from sklearn import metrics

from sklearn.metrics import accuracy_score
BernNB = BernoulliNB(binarize = True) # Function will binarize the data

BernNB.fit(train_inputs, train_targets)

print(BernNB)



y_expected = validation_targets

y_pred = BernNB.predict(validation_inputs)



print(accuracy_score(y_expected, y_pred))