!pip install tweet-preprocessor 2>/dev/null 1>/dev/null
import preprocessor as pcr
import numpy as np 
import pandas as pd 
import emoji
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from sklearn import preprocessing,  model_selection
from keras.preprocessing import sequence, text
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import plotly.express as px
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tokenizers import Tokenizer, models 
from tensorflow.keras.layers import SpatialDropout1D
df_data_1 = pd.read_csv("../input/tweetscsv/Tweets.csv")
df_data_1.head()
df_data = df_data_1[["tweet_id","airline_sentiment","text"]]
df_data.head()
data_spell = pd.read_csv("../input/spelling/aspell.txt",sep=":",names=["correction","misspell"])
data_spell.misspell = data_spell.misspell.str.strip()
data_spell.misspell = data_spell.misspell.str.split(" ")
data_spell = data_spell.explode("misspell").reset_index(drop=True)
data_spell.drop_duplicates("misspell",inplace=True)
miss_corr = dict(zip(data_spell.misspell, data_spell.correction))

#Sample of the dict
{v:miss_corr[v] for v in [list(miss_corr.keys())[k] for k in range(20)]}
def correct_spell(v):
    for a in v.split(): 
        if a in miss_corr.keys(): 
            v = v.replace(a, miss_corr[a]) 
    return v

df_data["clean_content"] = df_data.text.apply(lambda a : correct_spell(a))
contract = pd.read_csv("../input/contractions/contractions.csv")
cont_dict = dict(zip(contract.Contraction, contract.Meaning))
def contract_to_meaning(v): 
  
    for a in v.split(): 
        if a in cont_dict.keys(): 
            v = v.replace(a, cont_dict[a]) 
    return v

df_data.clean_content = df_data.clean_content.apply(lambda a : contract_to_meaning(a))
pcr.set_options(pcr.OPT.MENTION, pcr.OPT.URL)
pcr.clean("hello guys @alx #sport🔥 1245 https://github.com/s/preprocessor")
df_data["clean_content"]=df_data.text.apply(lambda a : pcr.clean(a))
def punct(v): 
  
    punct = '''()-[]{};:'"\,<>./@#$%^&_~'''
  
    for a in v.lower(): 
        if a in punct: 
            v = v.replace(a, " ") 
    return v

punct("test @ #ldfldlf??? !! ")
df_data.clean_content = df_data.clean_content.apply(lambda a : ' '.join(punct(emoji.demojize(a)).split()))
def text_cleaning(v):
    v = correct_spell(v)
    v = contract_to_meaning(v)
    v = pcr.clean(v)
    v = ' '.join(punct(emoji.demojize(v)).split())
    
    return v
text_cleaning("isn't 💡 adultry @ttt good bad ... ! ? ")
df_data = df_data[df_data.clean_content != ""]
df_data.airline_sentiment.value_counts()
id_for_sentiment = {"neutral":0, "negative":1,"positive":2}
df_data["sentiment_id"] = df_data['airline_sentiment'].map(id_for_sentiment)
df_data.head()
encoding_label = LabelEncoder()
encoding_integer = encoding_label.fit_transform(df_data.sentiment_id)

encoding_onehot = OneHotEncoder(sparse=False)
encoding_integer = encoding_integer.reshape(len(encoding_integer), 1)
Y = encoding_onehot.fit_transform(encoding_integer)
X_train, X_test, y_train, y_test = train_test_split(df_data.clean_content,Y, random_state=1995, test_size=0.2, shuffle=True)
# using keras tokenizer here
tkn = text.Tokenizer(num_words=None)
maximum_length = 160
Epoch = 15
tkn.fit_on_texts(list(X_train) + list(X_test))
X_train_pad = sequence.pad_sequences(tkn.texts_to_sequences(X_train), maxlen=maximum_length)
X_test_pad = sequence.pad_sequences(tkn.texts_to_sequences(X_test), maxlen=maximum_length)
t_idx = tkn.word_index
embedding_dimension = 160
lstm_out = 250

model_sql = Sequential()
model_sql.add(Embedding(len(t_idx) +1 , embedding_dimension,input_length = X_test_pad.shape[1]))
model_sql.add(SpatialDropout1D(0.2))
model_sql.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model_sql.add(keras.layers.core.Dense(3, activation='softmax'))
#adam rmsprop 
model_sql.compile(loss = "categorical_crossentropy", optimizer='adam',metrics = ['accuracy'])
print(model_sql.summary())
size_of_batch = 32
model_sql.fit(X_train_pad, y_train, epochs = Epoch, batch_size=size_of_batch,validation_data=(X_test_pad, y_test))
def get_emotion(model_sql,text_1):
    text_1 = text_cleaning(text_1)
    #tokenize
    tweet = tkn.texts_to_sequences([text_1])
    tweet = sequence.pad_sequences(tweet, maxlen=maximum_length, dtype='int32')
    emotion = model_sql.predict(tweet,batch_size=1,verbose = 2)
    emo = np.round(np.dot(emotion,100).tolist(),0)[0]
    rslt = pd.DataFrame([id_for_sentiment.keys(),emo]).T
    rslt.columns = ["sentiment","percentage"]
    rslt=rslt[rslt.percentage !=0]
    return rslt
def result_plotting(df):
    #colors=['#D50000','#000000','#008EF8','#F5B27B','#EDECEC','#D84A09','#019BBD','#FFD000','#7800A0','#098F45','#807C7C','#85DDE9','#F55E10']
    #fig = go.Figure(data=[go.Pie(labels=df.sentiment,values=df.percentage, hole=.3,textinfo='percent',hoverinfo='percent+label',marker=dict(colors=colors, line=dict(color='#000000', width=2)))])
    #fig.show()
    clrs={'neutral':'rgb(213,0,0)','negative':'rgb(0,0,0)',
                    'positive':'rgb(0,142,248)'}
    col={}
    for i in rslt.sentiment.to_list():
        col[i]=clrs[i]
    figure = px.pie(df, values='percentage', names='sentiment',color='sentiment',color_discrete_map=col,hole=0.3)
    figure.show()
rslt =get_emotion(model_sql,"Had an absolutely brilliant day ðŸ˜ loved seeing an old friend and reminiscing")
result_plotting(rslt)
rslt =get_emotion(model_sql,"The pain my heart feels is just too much for it to bear. Nothing eases this pain. I can’t hold myself back. I really miss you")
result_plotting(rslt)
rslt =get_emotion(model_sql,"I hate this game so much,It make me angry all the time ")
result_plotting(rslt)
def data_reading(file):
    with open(file,'r') as z:
        word_vocabulary = set() 
        word_vector = {}
        for line in z:
            line_1 = line.strip() 
            words_Vector = line_1.split()
            word_vocabulary.add(words_Vector[0])
            word_vector[words_Vector[0]] = np.array(words_Vector[1:],dtype=float)
    print("Total Words in DataSet:",len(word_vocabulary))
    return word_vocabulary,word_vector
vocabulary, word_to_index =data_reading("../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt")
matrix_embedding = np.zeros((len(t_idx) + 1, 200))
for word, i in t_idx.items():
    vector_embedding = word_to_index.get(word)
    if vector_embedding is not None:
        matrix_embedding[i] = vector_embedding
embedding_dimension = 200
lstm_out = 250

model_lstm = Sequential()
model_lstm.add(Embedding(len(t_idx) +1 , embedding_dimension,input_length = X_test_pad.shape[1],weights=[matrix_embedding],trainable=False))
model_lstm.add(SpatialDropout1D(0.2))
model_lstm.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(keras.layers.core.Dense(3, activation='softmax'))
#adam rmsprop 
model_lstm.compile(loss = "categorical_crossentropy", optimizer='adam',metrics = ['accuracy'])
print(model_lstm.summary())
size_of_batch = 32
model_lstm.fit(X_train_pad, y_train, epochs = Epoch, batch_size=size_of_batch,validation_data=(X_test_pad, y_test))
rslt =get_emotion(model_lstm,"Had an absolutely brilliant day ðŸ˜ loved seeing an old friend and reminiscing")
result_plotting(rslt)
rslt =get_emotion(model_lstm,"The pain my heart feels is just too much for it to bear. Nothing eases this pain. I can’t hold myself back. I really miss you")
result_plotting(rslt)
rslt =get_emotion(model_lstm,"I hate this game so much,It make me angry all the time ")
result_plotting(rslt)