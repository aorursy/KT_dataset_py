# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from tensorflow.keras.layers import Embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.text import one_hot

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input,GlobalMaxPool1D,Dropout

from keras.utils import plot_model

from tensorflow.keras.layers import Input,GlobalMaxPool1D,Dropout,concatenate

from tensorflow.keras.models import Model

from textblob import TextBlob



import nltk

import re

from nltk.corpus import stopwords #corpus is collection of text

#from nltk.stem.porter import PorterStemmer

from nltk.stem import RSLPStemmer



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score



#excel_data_df = pandas.read_excel('records.xlsx', sheet_name='Employees')
# Carregamento dos dados de treinamento - base de dados fake-news traduzida

train = pd.read_excel('../input/train-testxlsx/train.xlsx')

train.head()
#train = train.iloc[:200,]

train.shape
# Reemoção dos valores nulos - uso de texto como titulo e titulo como texto.

train['text' ] = train['text' ].replace(np.nan, train['title'])

train['title'] = train['title'].replace(np.nan, train['text' ])

train.isnull().sum()
# Separando a coluna com o label

X_train=train.drop('label',axis=1)

y_train=train['label']
# Define o tamanho do vocabulario. Sera usado no embeding

vo_size=500

messages=X_train.copy()

messages.reset_index(inplace=True)
#Convertendo tudo para str

for i in range(0, len(messages)):

    if not (type(messages['text'][i]) is str):

        print(str(messages['text'][i]))
stop_words = stopwords.words('portuguese') #stop words in Português

newStopWords = ['The New York Times', 'de', 'do', 'para', 'no', 'na', 'nos' , 'nas', 'uma', 'um', 

                'umas','uns', 'a', 'Breitbart', 'New York','York', 'Times', 'The', 'New','the', 'The New',

               ' York times', ' Times', ' times', 'York Times', 'breitbart', 'times', 'Times', ' Breitbart',

               '- The New York Times','new','york' ]

stop_words.extend(newStopWords)
#Pre processamento da base - normalização das palavras com base em seus radicais

#https://pythonspot.com/nltk-stemming/

#ps_title =PorterStemmer()

#ps_text =PorterStemmer()

ps_title =RSLPStemmer() # Removedor de sufixos da lingua portuguesa

ps_text =RSLPStemmer()

corpus_title = []

corpus_text = []

for i in range(0, len(messages)):

    if not (type(messages['text'][i]) is str):

        messages['text'][i] = str(messages['text'][i])

    print("Status: %s / %s" %(i, len(messages)), end="\r")

    

    #preproc title

    review = re.sub('[^a-zA-Z]', ' ',messages['title'][i]) #Filtro de caracteres - Deixndo apenas texto

    review = review.lower()

    review = review.split()

    

    review = [ps_title.stem(word) for word in review if not word in stop_words]

    review = ' '.join(review)

    corpus_title.append(review)

    

    #preproc text

    review = re.sub('[^a-zA-Z]', ' ',messages['text'][i])

    review = review.lower()

    review = review.split()

    review = [ps_text.stem(word) for word in review if not word in stop_words]

    review = ' '.join(review)

    corpus_text.append(review)
# corpus_text - vetor com strings ['etc alt pal ...','process pul presid ...',...]
# Representação One hot - Codificação em números

# Vetor de vetores com numeros - strings tranformadas em vetores - As redes neurais não trabalham com dados categoricos diretamente

onehot_rep_title = [one_hot(words, vo_size) for words in corpus_title] #vo_size definido anteriormente como o tamanho do vocabulario

onehot_rep_text = [one_hot(words, vo_size) for words in corpus_text] #one_hot codifica as palavras em vetores numericos preenchidos com zeros com exceção de um único 1
 # A camada de  Embedding do Keras  requer que todos os documentos tenham o mesmo tamanho.

# Explicação do embeding: https://www.kaggle.com/rajmehra03/a-detailed-explanation-of-keras-embedding-layer

# Uniformização do tamanho - preenchimento de lacunas com o 'padding'- os vetores passam a ter mesmo tamanho

# Tamanhos definidos abaixo serão usadas no Input da rede neural

sent_length_title = 20 # Tamanho da linha para o titulo

sent_length_text = 1000 # Tamanho da linha para o corpo do texto

embedded_doc_title=pad_sequences(onehot_rep_title, padding='pre', maxlen=sent_length_title)

embedded_doc_text=pad_sequences(onehot_rep_text, padding='pre', maxlen=sent_length_text)
# Check do formato - y-train é uma série do Pandas

print(len(embedded_doc_title),y_train.shape)

print(len(embedded_doc_text),y_train.shape)
# Dados concluidos para o processamento pela rede neural - NN

# Confirmação do tipo como array Numpy

#Verificação das dimensões X e y

X_final_title=np.array(embedded_doc_title)

X_final_text=np.array(embedded_doc_text)

y_final=np.array(y_train)

print(X_final_title.shape,y_final.shape)

print(X_final_text.shape,y_final.shape)
# Modelagem

# Embedding é uma técnica usada para representar palavras em documentos como vetores. A representação usa numeros reais tal que palavras similares semanticamente são mapeadas uma proxima à outra.

#Thus the embedding layer in Keras can be used when we want to create the embeddings to embed higher dimensional data into lower dimensional vector space.

# the integer encoding for the word remains same in different docs. eg 'butter' is denoted by 31 in each and every document.

# the vocab_size is specified large enough so as to ensure unique integer encoding for each and every word.

# The Keras Embedding layer requires all individual documents to be of same length. Hence we wil pad the shorter documents with 0 for now. 

# Therefore now in Keras Embedding layer the 'input_length' will be equal to the length (ie no of words) of the document with maximum length or maximum number of words.

# To pad the shorter documents I am using pad_sequences functon from the Keras library.

""""" Parametros da camada de embedding ---



'input_dim' = tamanho do vocabulario. Ou seja, o numero de palavras unicas no vocabulario.



'output_dim' = Número dimensões resultante. Cada palavra será representada por um vetor com estas dimensões.



"""



embedding_vector_feature_title = 10

embedding_vector_feature_text = 100



input_title = Input(shape=(sent_length_title,)) # define formato da entrada para o titulo. O tamanho (sent_length_title) foi definido anteriormente.

input_text = Input(shape=(sent_length_text,)) # define formato da entrada para o texto



emb_title = Embedding(vo_size,embedding_vector_feature_title)(input_title) # primeiro argumento é o tamanho do vocabulario. O segundo é o tamanho do vetor do embeding

lstm_title = LSTM(128, return_sequences=False)(emb_title)# https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47

#LSTM supera outros modelos quando queremos aprendizado baseado em termos distantes no tempo.  A habilidade das LSTM  esquecer, lembrar e atualizar informações a coloca à frente das RNN.

# a informação de entrada é sequencial

#A camada LSTM transforma a sequencia de vetores em um vetor unico, condesando a informação de toda a sequencia.



emb_text = Embedding(vo_size,embedding_vector_feature_text)(input_text)

lstm_text = LSTM(128, return_sequences=True)(emb_text) 





max_pool_text = GlobalMaxPool1D()(lstm_text) # GlobalMaxpooling difere de maxpooling porque pool_length = tamanho da entrada.Promove a redução da dimensionalidade para prevenir overfitting e diminuir o custo computacional.

# In the layer of polling, we use the max-pooling mechanism to reduce the impact of noise that from the output of convolutional layer, at the same time, use it can reduce the feature dimension and prevent model overfitting.

dropout_1_text = Dropout(0.1)(max_pool_text) # desliga 10# das celulas. Também minimiza o overfitting.

dense_1_text = Dense(50, activation='relu')(dropout_1_text) # celula comum que combina as entradas

dropout_2_text = Dropout(0.1)(dense_1_text)



out = concatenate([lstm_title,dropout_2_text],axis=-1) # concatena titulo e saida de texto

output=Dense(1, activation='sigmoid')(out)



model = Model(inputs=[input_title, input_text], outputs=output) # modelo com duas entradas

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # saida binaria.  entropia cruzada – indica a distância entre o que a rede acredita que essa distribuição deve ser e o que realmente deveria ser. 

# Pode ser mais útil em problemas nos quais os alvos são 0 e 1.

print(model.summary())
# Get model plot

plot_model(model, to_file='model_plot4.png', show_shapes=True, show_layer_names=True)
# treinamento - O tamanho do batch define o tamanho do bloco a ser usado no treinamento antes de atualziar os parametros da rede. Blocos menores geram gradientes com comportamento mais erratico.

# as epocas definem a quantidade de vezes que se passa por toda a base.

history = model.fit(x=[X_final_title,X_final_text], y=y_final, batch_size=128, epochs=10, verbose=1, validation_split=0.2) 
history.history
# Carregamento dos dados de teste

test = pd.read_excel('../input/train-testxlsx/test.xlsx')

test.head()
#test = test.iloc[:100,]
# Replace NA

test['text'] = test['text'].replace(np.nan, test['title'])

test['title'] = test['title'].replace(np.nan, test['text'])

test.isnull().sum()
# prepare test data for NN

X_test=test

messages=X_test.copy()

messages.reset_index(inplace=True)



ps_title =RSLPStemmer()

ps_text =RSLPStemmer()

corpus_title = []

corpus_text = []

for i in range(0, len(messages)):

    print("Status: %s / %s" %(i, len(messages)), end="\r")

    

    #preproc title

    review = re.sub('[^a-zA-Z]', ' ',messages['title'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps_title.stem(word) for word in review if not word in stop_words]

    review = ' '.join(review)

    corpus_title.append(review)

    

    #preproc text

    review = re.sub('[^a-zA-Z]', ' ',messages['text'][i])

    review = review.lower()

    review = review.split()

    

    review = [ps_text.stem(word) for word in review if not word in stop_words]

    review = ' '.join(review)

    corpus_text.append(review)



onehot_rep_title = [one_hot(words, vo_size) for words in corpus_title]

onehot_rep_text = [one_hot(words, vo_size) for words in corpus_text]



sent_length_title = 20

sent_length_text = 1000

embedded_doc_title=pad_sequences(onehot_rep_title, padding='pre', maxlen=sent_length_title)

embedded_doc_text=pad_sequences(onehot_rep_text, padding='pre', maxlen=sent_length_text)



X_final_title=np.array(embedded_doc_title)

X_final_text=np.array(embedded_doc_text)

print(X_final_title.shape)

print(X_final_text.shape)
# predição final

y_pred_final = model.predict ([X_final_title,X_final_text])

y_prob = pd.DataFrame(y_pred_final)

y_prob['0'] = 1 - y_prob[0]

y_class = pd.DataFrame(y_prob.values.argmax(axis=-1))

y_class[0] = np.where(y_class[0]==1, 0, 1)

y_class.head()

submit = pd.concat([test['id'].reset_index(drop=True), y_class], axis=1)

submit.rename(columns={ submit.columns[1]: "label" }, inplace = True)

submit.isnull().sum()
# Salvar o modelo

model.save_weights("model_multi.h5")
# Gráfico da acurácia



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# Gráfico da diferença (loss) 

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()