import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import os
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet') 
lemmatizer = WordNetLemmatizer() 
from spacy.lang.en.stop_words import STOP_WORDS
# plt.xkcd()
# read the train and test tsv files (seperator is tab that's why tsv)
train = pd.read_csv("../input/mercari-dataset/train.tsv",sep = '\t') 
print("Number of rows {} and columns {} ".format(train.shape[0],train.shape[1]))
train.sample(5)
test = pd.read_csv("../input/mercari-dataset/test.tsv",sep = "\t")
print("Number of rows {} and columns {} ".format(test.shape[0],test.shape[1]))
test.sample(5)
train.info()
test.info()
# let's first explore our target variable 
# let's see target variable (price) distribution
plt.xkcd()
fig,(ax1,ax2) = plt.subplots(nrows = 1,ncols=2,figsize = (18,8))
ax1.hist(train.price,bins = 30,range = [0,200])
ax2.hist(np.log(train.price + 1),bins = 30,range = [0,8])
ax1.set_title("Actual Price Distribution")
ax1.axvline(x = np.percentile(train.price,q = 80),color = 'red',label = "80 percent of population lies on left")
ax1.set_xlabel("Price")
ax2.set_title("Log(Price) Distribution")
ax2.set_xlabel("Price")
#plt.legend()
plt.suptitle("Price Feature",fontsize = 20)
plt.show()
# from above analysis it will be better if we add log(Price) as different features, so that forthcoming analysis will be comfortable.
train['log(price)'] = np.log(train['price']+1)
# added new features 
# this is called feature engineering where from the given features you come up with intuitive and useful features.
# shipping feature in data seems to be reasonable to explore.
# let's see distribution of shipping data with log(price)

fig, ax = plt.subplots(figsize=(14,8))
ax.hist(train[train.shipping==1]['log(price)'],bins=30,range=[0,8],label="Buyer_charged",color='b',alpha=0.5)
ax.hist(train[train.shipping==0]['log(price)'],bins=30,range=[0,8],label="seller_charged",color='r',alpha=0.5)
plt.title('Price distribution', fontsize=15)
ax.set_xlabel('log(Price+1)',fontsize=15)
ax.set_ylabel('No of items',fontsize=15)
plt.legend(loc='upper right')

plt.show()
# let's fill in missing data for further analysis with brand name and category name
def fill_missing_data(data):
    data.category_name.fillna(value = "others", inplace = True)
    data.brand_name.fillna(value = "not known", inplace = True)
    data.item_description.fillna(value = "no description", inplace = True)
    return(data)
train = fill_missing_data(train)
#test  = fill_missing_data(test)
# let's also split category_name into 3 parts main_category, sub_category_1 and sub_category_2


# Let's split the categories into three different columns. We will see later that this information is actually quite important from the seller's point of view and how we handle 
# the missing information in the brand_name column will impact the model's prediction.

def split_cat(text):
    try:
        text1, text2, text3 = text.split('/')
        return text1, text2, text3
    except: 
        return ("No Label", "No Label", "No Label")
train['main_cat'], train['subcat_1'], train['subcat_2'] = zip(*train['category_name'].apply(lambda x: split_cat(x)))
#test['main_cat'], test['subcat_1'], test['subcat_2'] = zip(*test['category_name'].apply(lambda x: split_cat(x)))

train['summary'] = train['name'] + train['item_description']
#test['summary'] = test['name'] + test['item_description']
train.head()
plt.figure(figsize = (16,8))
plt.suptitle("Distribution of Price per Main Category")
sns.boxplot(y = 'log(price)', x = 'main_cat',data = train)
plt.xticks(rotation = 45)
plt.show()
import plotly
x = train['subcat_1'].value_counts().index.values.astype('str')[:15]
y = train['subcat_1'].value_counts().values[:15]
pct = [("%.2f"%(v*100))+"%"for v in (y/len(train))][:15]

trace1 = go.Bar(x=x, y=y, text=pct)
layout = dict(title= 'Number of Items by Sub Category 1',
              yaxis = dict(title='Count'),
              xaxis = dict(title='Category'))
fig=go.Figure(data = trace1,layout=layout)
fig.show()
import plotly
x = train['subcat_2'].value_counts().index.values.astype('str')[:15]
y = train['subcat_2'].value_counts().values[:15]
pct = [("%.2f"%(v*100))+"%"for v in (y/len(train))][:15]

trace1 = go.Bar(x=x, y=y, text=pct)
layout = dict(title= 'Number of Items by Sub Category 2',
              yaxis = dict(title='Count'),
              xaxis = dict(title='Category'))
fig=go.Figure(data = trace1,layout=layout)
fig.show()
import re
# here STOP_WORDS is spacy stopwords list
def wordCount(text):
    # convert to lower case and strip regex
    try:
         # convert to lower case and strip regex
        text = text.lower()
        #regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        #txt = regex.sub(" ", text)
        txt = re.sub('[^a-zA-Z]+',' ',text)
        words = [lemmatizer.lemmatize(w) for w in txt.split(" ") \
                 if not w in STOP_WORDS and len(w)>3]
        return(len(words))
    except: 
        return 0
# add a column of word counts to both the training and test set
train['desc_len'] = train['summary'].apply(lambda x: wordCount(x))
plt.figure(figsize = (16,8))
sns.violinplot(x = 'main_cat',y = 'desc_len',data = train)
plt.xticks(rotation = 45)
plt.show()
# let's see for top 10 description length the prize distribution.

temp = pd.DataFrame()
for i in tqdm(train['desc_len'].value_counts().index.values[:10]):
    temp = pd.concat([temp,train[train['desc_len']==i]],axis=0)

plt.figure(figsize = (16,8))
sns.boxplot(x = 'desc_len',y = 'log(price)',data = temp)
plt.show()
%%time
embeddings_index = {}
f = open('../input/glovetwitter27b100dtxt/glove.twitter.27B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
import re
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


def text_preprocess(df):
    pre_list = []
    for sentance in tqdm(df['summary'].values):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(lemmatizer.lemmatize(e) for e in sent.split() if e.lower() not in stopwords)
        pre_list.append(sent.lower().strip())
    return(pre_list)
train_preprocess = text_preprocess(train)
#test_preprocess = text_preprocess(test)
train['summary'] = train_preprocess
#test['summary'] = test_preprocess
vocab = embeddings_index.keys()
# this function will add the vectors of each word and returns the avg vector of given sentance
def build_avg_vec(sentence, num_features, doc_id, m_name):
    # sentace: its title of the apparel
    # num_features: the lenght of word2vec vector, its values = 300
    # m_name: model information it will take two values
        # if  m_name == 'avg', we will append the model[i], w2v representation of word i
        # if m_name == 'weighted', we will multiply each w2v[word] with the idf(word)

    featureVec = np.zeros((num_features,), dtype="float32")
    # we will intialize a vector of size 300 with all zeros
    # we add each word2vec(wordi) to this fetureVec
    nwords = 0
    
    for word in sentence.split():
        nwords += 1
        if word in vocab:
            if m_name == 'weighted' and word in  idf_title_vectorizer.vocabulary_:
                featureVec = np.add(featureVec, idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[word]] * embeddings_index[word])
            elif m_name == 'avg':
                featureVec = np.add(featureVec, embeddings_index[word])
    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    # returns the avg vector of given sentance, its of shape (1, 300)
    return featureVec

def calc_avgw2v(train_preprocess,test_preprocess):
    doc_id = 0
    avg_w2v = []
    # for every description we build a avg vector representation
    for i in tqdm(train_preprocess):
        avg_w2v.append(build_avg_vec(i, 100, doc_id,'avg'))
        doc_id += 1
    # w2v_desc = np.array(# number of doc in courpus * 100), each row corresponds to a doc 
    avg_w2v_train = np.array(avg_w2v)

    avg_w2v = []
    # for every title we build a avg vector representation
    for i in tqdm(test_preprocess):
        avg_w2v.append(build_avg_vec(i, 100, doc_id,'avg'))
        doc_id += 1
    # w2v_title = np.array(# number of doc in courpus * 100), each row corresponds to a doc 
    avg_w2v_test = np.array(avg_w2v)
    return(avg_w2v_train,avg_w2v_test)
avg_w2v_train,_ = calc_avgw2v(train['summary'].values,[])

# splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(avg_w2v_train,train['log(price)'].values, test_size = 0.2,random_state = 42)
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation,BatchNormalization,LeakyReLU
from tensorflow.keras.layers import Embedding
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
model = Sequential()
model.add(Dense(300,activation=LeakyReLU(),kernel_initializer='glorot_normal',input_shape = (100,)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(300,activation=LeakyReLU(),kernel_initializer='glorot_normal'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(200,activation=LeakyReLU(),kernel_initializer='glorot_normal'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(200,activation=LeakyReLU(),kernel_initializer='glorot_normal'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(100,activation=LeakyReLU(),kernel_initializer='glorot_normal'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(50,activation=LeakyReLU(),kernel_initializer='glorot_normal'))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(1,activation='linear'))
model.summary()
from tensorflow.keras.optimizers import Adam,Nadam

# tf.keras.losses.MeanSquaredLogarithmicError(
#     reduction="auto", name="mean_squared_logarithmic_error"
# )
# mae = tf.keras.losses.MeanAbsoluteError(name = "mean_absolute_error")

reduce_lr = ReduceLROnPlateau(monitor= "mse", factor=0.2,
                              patience=2, min_lr=0.00001)

tnsr_brd = TensorBoard(log_dir='logs')


ckpt = ModelCheckpoint('cnn_model.h5',
                            verbose=1, save_weights_only=True,save_best_only = True)

callbacks = [reduce_lr, tnsr_brd, ckpt ]

adam = Adam(learning_rate = 2e-3)
nadam = Nadam(learning_rate = 1e-4)
model.compile(loss=["mse"], optimizer=nadam)
history = model.fit(X_train,y_train,
                    batch_size=256,
                    epochs=20,
                    verbose=1,
                    validation_data = (X_test,y_test),callbacks = callbacks)
from sklearn.metrics import mean_squared_error

print("root mean squared error is : ",np.sqrt(mean_squared_error(y_test,model.predict(X_test))))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train.drop(['price','log(price)'],axis=1),train[['log(price)','price']],test_size = 0.05,random_state = 42)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_brnd = CountVectorizer()
vectorizer_brnd.fit(X_train['brand_name'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_brandname = vectorizer_brnd.transform(X_train['brand_name'].values)
test_brandname = vectorizer_brnd.transform(X_test['brand_name'].values)


print("After vectorizations")
print(train_brandname.shape)
print(test_brandname.shape)

print(len(vectorizer_brnd.get_feature_names()))
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_name = CountVectorizer()
vectorizer_name.fit(X_train['name'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_name = vectorizer_name.transform(X_train['name'].values)
test_name = vectorizer_name.transform(X_test['name'].values)


print("After vectorizations")
print(train_brandname.shape)
print(test_brandname.shape)

print(len(vectorizer_name.get_feature_names()))
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_main = CountVectorizer()
vectorizer_main.fit(X_train['main_cat'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_cat = vectorizer_main.transform(X_train['main_cat'].values)
test_cat= vectorizer_main.transform(X_test['main_cat'].values)


print("After vectorizations")
print(train_cat.shape)
print(test_cat.shape)

print(len(vectorizer_main.get_feature_names()))
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_sub_1 = CountVectorizer()
vectorizer_sub_1.fit(X_train['subcat_1'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_subcat_1 = vectorizer_sub_1.transform(X_train['subcat_1'].values)
test_subcat_1 = vectorizer_sub_1.transform(X_test['subcat_1'].values)


print("After vectorizations")
print(train_subcat_1.shape)
print(test_subcat_1.shape)

print(len(vectorizer_sub_1.get_feature_names()))
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_sub_2 = CountVectorizer()
vectorizer_sub_2.fit(X_train['subcat_2'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
train_subcat_2 = vectorizer_sub_2.transform(X_train['subcat_2'].values)
test_subcat_2 = vectorizer_sub_2.transform(X_test['subcat_2'].values)


print("After vectorizations")
print(train_subcat_2.shape)
print(test_subcat_2.shape)

print(len(vectorizer_sub_2.get_feature_names()))
## number of words in description

def tokens(text):
    text = re.sub("[^A-Za-z]+"," ",text)
    return(len(text.split(" ")))

X_train['description_len'] = X_train['item_description'].apply(tokens)
X_test['description_len'] = X_test['item_description'].apply(tokens)
from sklearn.preprocessing import Normalizer,StandardScaler,normalize
normalizer = StandardScaler()
normalizer.fit(X_train['description_len'].values.reshape(-1,1))

#X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
train_des_norm = normalizer.transform(X_train['description_len'].values.reshape(-1,1))

test_des_norm = normalizer.transform(X_test['description_len'].values.reshape(-1,1))

print("After normalizations")
print(train_des_norm.shape, y_train.shape)

print(test_des_norm.shape, y_test.shape)
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()
def sentiment_analyzer(df,preprocess_text):
    temp = [] 
    for sentence in tqdm(preprocess_text):
        for_sentiment = sentence
        ss = sid.polarity_scores(for_sentiment)
        temp.append(ss)
    negative=[]
    neutral=[]
    positive=[]
    compounding=[]
    for i in temp:
        for polarity,score in i.items():
            if(polarity=='neg'):
                negative.append(score)
            if(polarity=='neu'):
                neutral.append(score)
            if(polarity=='pos'):
                positive.append(score)
            if(polarity=='compound'):
                compounding.append(score)
    df['negative']=negative
    df['neutral']=neutral
    df['positive']=positive
    df['compound']=compounding
    return(df)

X_train = sentiment_analyzer(X_train,X_train['item_description'].values)
X_test = sentiment_analyzer(X_test,X_test['item_description'].values)
def encoding(feature,tr,ts):
    vectorizer = CountVectorizer()
    vectorizer.fit(tr[feature].values) # fit has to happen only on train data

    # we use the fitted CountVectorizer to convert the text to vector
    train_temp = vectorizer.transform(tr[feature].values)
    test_temp= vectorizer.transform(ts[feature].values)


    print("After vectorizations")
    print(train_temp.shape)
    print(test_temp.shape)

    print(len(vectorizer.get_feature_names()))
    
    
def norm(feature,tr,ts):
    normalizer = StandardScaler()
    normalizer.fit(tr[feature].values.reshape(-1,1))

    #X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
    train_temp = normalizer.transform(tr[feature].values.reshape(-1,1))

    test_temp = normalizer.transform(ts[feature].values.reshape(-1,1))

    print("After normalizations")
    print(train_temp.shape, y_train.shape)

    print(test_temp.shape, y_test.shape)
    return(train_temp,test_temp)
train_neu,test_neu = norm('neutral',X_train,X_test)
train_pos,test_pos = norm('positive',X_train,X_test)
train_neg,test_neg = norm('negative',X_train,X_test)
train_comp,test_comp = norm('compound',X_train,X_test)
avg_w2v_train,avg_w2v_test = calc_avgw2v(X_train['summary'].values,X_test['summary'].values)

from scipy.sparse import csr_matrix
train_dummies = csr_matrix(pd.get_dummies(X_train[['item_condition_id', 'shipping']],
                                          sparse=True).values)

test_dummies = csr_matrix(pd.get_dummies(X_test[['item_condition_id', 'shipping']],
                                          sparse=True).values)

%%time
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=10,max_features=5000)
vectorizer.fit(X_train['summary'].values)
X_train_itemdes = vectorizer.transform(X_train['summary'].values)
X_test_itemdes = vectorizer.transform(X_test['summary'].values)
from scipy.sparse import hstack
train_data = hstack((X_train_itemdes,avg_w2v_train,train_brandname, train_name,train_dummies,train_cat,train_subcat_1,train_subcat_2,train_des_norm,train_neu,train_neg,train_pos,train_comp))
test_data = hstack((X_test_itemdes,avg_w2v_test,test_brandname,test_name,test_dummies ,test_cat,test_subcat_1,test_subcat_2,test_des_norm,test_neu,test_neg,test_pos,test_comp))
print("Final Data matrix")
print(train_data.shape, y_train['log(price)'].shape)

print(test_data.shape, y_test['log(price)'].shape)
print("="*100)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
lgbm_params ={'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 50, 'objective': 'regression','boosting_type': 'gbdt','learning_rate': 0.5,
              'max_depth': 8,'n_estimators': 500,'num_leaves': 80,
      }
model = LGBMRegressor(**lgbm_params)
model.fit(train_data, y_train['log(price)'].values,
         verbose=True)

preds2 = model.predict(test_data)
np.sqrt(mean_squared_error(y_test['log(price)'].values, preds2))
sample_sub = pd.read_csv("../input/mercari-dataset-asd/sample_submission.csv")
def predict(dataframe):
    
    dataframe  = fill_missing_data(dataframe)
    dataframe['main_cat'], dataframe['subcat_1'], dataframe['subcat_2'] = zip(*dataframe['category_name'].apply(lambda x: split_cat(x)))
    dataframe['summary'] = dataframe['name'] + dataframe['item_description']
    dataframe['summary'] = text_preprocess(dataframe)
    test_brand = vectorizer_brnd.transform(dataframe['brand_name'].values)
    test_name = vectorizer_name.transform(dataframe['name'].values)
    test_cat= vectorizer_main.transform(dataframe['main_cat'].values)
    test_subcat_1 = vectorizer_sub_1.transform(dataframe['subcat_1'].values)
    test_subcat_2 = vectorizer_sub_2.transform(dataframe['subcat_2'].values)
    dataframe['description_len'] = dataframe['item_description'].apply(tokens)
    test_dummies = csr_matrix(pd.get_dummies(dataframe[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    test_des_norm = normalizer.transform(dataframe['description_len'].values.reshape(-1,1))
    dataframe = sentiment_analyzer(dataframe,dataframe['item_description'].values)
    train_neu,test_neu = norm('neutral',X_train,dataframe)
    train_pos,test_pos = norm('positive',X_train,dataframe)
    train_neg,test_neg = norm('negative',X_train,dataframe)
    train_comp,test_comp = norm('compound',X_train,dataframe)
    avg_w2v = []
    doc_id=0
    for i in tqdm(dataframe['summary'].values):
        avg_w2v.append(build_avg_vec(i, 100, doc_id,'avg'))
        doc_id += 1
    avg_w2v = np.array(avg_w2v)
    X_test_itemdes = vectorizer.transform(dataframe['summary'].values)
    test_data = hstack((X_test_itemdes,avg_w2v,test_brand,test_name,test_dummies,test_cat,test_subcat_1,test_subcat_2,test_des_norm,test_neu,test_neg,test_pos,test_comp))
    return(test_data)
%%time
test_data = predict(test)
sample_sub['price'] = np.expm1(model.predict(test_data))
sample_sub.to_csv("submission_lgbm.csv")