# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import gensim
all_stopwords = stopwords.words('english') #we won't remove them for this particular dataset



#Function to preprocess text

def cleaning_text(Text):
    documents = []      #to use for the embedding
    documents_Tok = []  #to use for indexing words
    len_sentences = []  #to register the lenght of sentences
    for line in Text:
        
#Remove HTML tags
        document = re.sub(r'<[^>]+>', ' ', line)

#Remove all the special characters
        document = re.sub(r'\W', ' ', document)
    
#remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
#Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
#Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
#Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
    
#Converting to Lowercase
        document = document.lower()
    
#Filling the array that we will use to create the tokenized vectors 
        documents_Tok.append(document)
    
#Tokenizing
        document = word_tokenize(document)
        #document = [word for word in document if not word in all_stopwords] #to remove stopwords, we will not
    
#Recording the lenghts of all sentences
        len_sentences.append(len(document)) 
    
#Filling the array that we will use to create the embedded vector for words
        documents.append(document)
    
    return documents, documents_Tok, len_sentences



#INPUT
rev_input = open(r"../input/inputsfile/input.txt")

lines_X = rev_input.readlines()

X, X_for_Tok, len_sen = cleaning_text(lines_X)

#Mesuring the lenght of the longest sentence to know the size of the input
max_len = max(len_sen)

print(max_len)




#DOC for TRAINING
LongText = open(r"../input/inputsfile/ForLearning.txt", errors='ignore')

ForLearning, NotUsed, NotUsedLen = cleaning_text(LongText)

training_docs = X + ForLearning



#TARGET
rev_label = open(r"../input/inputsfile/label.txt")

lines_y = rev_label.readlines()

y = []

#Classifing the label, substitiution of negative/positive target with 0/1
rank0 = r'''negative'''
rank1 = r'''positive'''
rank_n = re.compile(rank0)
rank_p = re.compile(rank1)


for line in lines_y:
    if rank_n.search(line) != None:
        y.append(0)
    elif rank_p.search(line) != None:
        y.append(1)
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


X_for_Tok = np.array(X_for_Tok)


# prepare tokenizer and indexed vocabulary
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(X_for_Tok)
vocab_size = len(t.word_index) + 1


# integer encode the documents
encoded_docs = t.texts_to_sequences(X_for_Tok)


# pad documents to a max length of 50 words
max_length = max_len
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
from gensim.models import KeyedVectors
from gensim.downloader import base_dir
from gensim.models import Word2Vec



#Loading and training

#Defining function for the word2vec pre-trained by google
word2vec_path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin"


Google_Word2Vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


google_emb_size = 300 #size of vector representing the embedded word:300 fixed by Google pre-trained model


#Word2Vec trained on our dataset to embedd the absent words in the Google pre-trained model
word_to_vec_model = Word2Vec(training_docs, min_count = 1, size = google_emb_size , window = 5, sg = 1)
#Embedding of words of the indexed vocabulary we created before

word_index = t.word_index

#We fill a matrix, were every row represents, in order, the indexed words and will be the embedded vector
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    if (word in Google_Word2Vec.vocab)==True:
        embedding_vector = Google_Word2Vec[word]
    else:
        embedded_vector = word_to_vec_model[word]
    embedding_matrix[i] = embedding_vector
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#Train of the model Doc2Vec

tag_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(training_docs)]


model = doc2vec.Doc2Vec(vector_size = 300, window = 1, min_count = 3, workers = 6, epochs=20)

model.build_vocab(tag_documents)
model.train(tag_documents, total_examples=model.corpus_count, epochs=20)
model.save("d2v.model")
#Creating the vectors representing the embedded documents

D2V_model = Doc2Vec.load("d2v.model")

v = []


for i in range(len(X)):
    v.append(D2V_model.docvecs[i])
    
    
from sklearn.preprocessing import normalize
wnorm=v
wnorm=np.array(wnorm)
wnorm=normalize(wnorm)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



#Divide data in training and test set
test_size = 0.2

index_for_splitting = list(range(0, len(X), 1)) #to mantain the same order for the two different inputs

SEED = 2000

index_train, index_test, y_train, y_test = train_test_split(index_for_splitting, y, test_size=test_size, random_state = SEED)


#for training set
Embedded_train = []
Tokenized_train = []

for i in range(len(index_train)):
    Embedded_train.append(wnorm[index_train[i]])
    Tokenized_train.append(padded_docs[index_train[i]])


#for test set
Embedded_test = []
Tokenized_test = []

for i in range(len(index_test)):
    Embedded_test.append(wnorm[index_test[i]])
    Tokenized_test.append(padded_docs[index_test[i]])

    
    
    

#Creating a validation set from the training set
val_size = .2

X_index_train, X_index_val, y_train, y_val = train_test_split(index_train, y_train, test_size=val_size, random_state = SEED)


#for training set
X_embedded_train = []
X_tokenized_train = []

for i in range(len(X_index_train)):
    X_embedded_train.append(wnorm[X_index_train[i]])
    X_tokenized_train.append(padded_docs[X_index_train[i]])


#for validation set
X_embedded_val = []
X_tokenized_val = []

for i in range(len(X_index_val)):
    X_embedded_val.append(wnorm[X_index_val[i]])
    X_tokenized_val.append(padded_docs[X_index_val[i]])

    

    
#tranforming lists into arrays

X_embedded_train = np.array(X_embedded_train)
X_embedded_val = np.array(X_embedded_val)
Embedded_test = np.array(Embedded_test)

X_tokenized_train = np.array(X_tokenized_train)
X_tokenized_val = np.array(X_tokenized_val)
Tokenized_test = np.array(Tokenized_test)

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

print(X_tokenized_train.shape)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D, Reshape, Dropout, Add
### keras.backend.clear_session()



embedding_layer = Embedding(vocab_size,
                            google_emb_size,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=True)


#The First Model
inputs1 = Input(shape=(50,), dtype='int32')#sentences lenght

x = embedding_layer(inputs1)

x = Conv1D(filters=200, kernel_size=(3), padding='same', strides=(1), activation='relu')(x)

x = Dropout(0.3)(x)

x = Conv1D(filters=200, kernel_size=(3), padding='same', strides=(1), activation='relu')(x)

#x = MaxPooling1D(pool_size=(2), strides=(1),  padding='valid')(x)

x = Dropout(0.3)(x)

x = Conv1D(filters=150, kernel_size=(3), padding='same', strides=(1), activation='relu')(x)

x = MaxPooling1D(pool_size=(2), strides=(1),  padding='valid')(x)

x = Conv1D(filters=150, kernel_size=(2), padding='same', strides=(1), activation='relu')(x)

x = MaxPooling1D(pool_size=(2), strides=(1),  padding='valid')(x)

x = Flatten()(x)

outputs1 = Dense(100, activation='tanh')(x)

model1 = Model(inputs=inputs1, outputs=outputs1, name='convolution_for_text')



#The Second Model
inputs2 = Input(shape=(300))

x = Dense(200, activation='relu')(inputs2) #100 hidden units with RELU activation

x = Dropout(0.3)(x)

x = Dense(100, activation='relu')(x)

outputs2 = Dense(100, activation='tanh')(x)

model2 = Model(inputs=inputs2, outputs=outputs2, name='NN_for_doc_vecs')



#The unifing layer
mergedOut =Add()([model1.output,model2.output])

#The last layers
mergedOut = Dense(300, activation='elu')(mergedOut)

memergedOut = Dropout(0.3)

mergedOut = Dense(300, activation='tanh')(mergedOut)

mergedOut = Dense(1, activation='sigmoid')(mergedOut)


#we reconstruct the net
CompleteModel = Model([model1.input,model2.input], mergedOut)


eta = 0.07
decay_rate = 0.001

#For the first epochs we make the Word2Vec wights not trainable
embedding_layer.trainable = False
CompleteModel.summary()

#Compiling the model
CompleteModel.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=eta, momentum=0.01, decay=decay_rate),
    metrics=['accuracy']
)

print(eta)
epochs1 = 4
batch_size = 50


checkpointer = keras.callbacks.ModelCheckpoint(filepath="WordDoc2VecReviews.hdf5", verbose=1, save_best_only=True)

#Fitting and showing the results at each epoch
history = CompleteModel.fit(
    [X_tokenized_train, X_embedded_train],
    y_train,
    epochs=epochs1,
    batch_size=batch_size,
    validation_data=([X_tokenized_val, X_embedded_val],y_val),
    verbose=1
)    

#After the net is stabilized, we will also train the weights of the word vectors
embedding_layer.trainable = True

CompleteModel.summary()


#Compiling the model
CompleteModel.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=eta, momentum=0.0, decay=decay_rate),
    metrics=['accuracy']
)


epochs2 = 15


#Fitting and showing the results at each epoch
history = CompleteModel.fit(
    [X_tokenized_train, X_embedded_train],
    y_train,
    callbacks=[checkpointer], #in this way we save the model with the best loss in validation set
    epochs=epochs2,
    batch_size=batch_size,
    validation_data=([X_tokenized_val, X_embedded_val],y_val),
    verbose=1
)  
CompleteModel.load_weights('WordDoc2VecReviews.hdf5')

score, acc = CompleteModel.evaluate([X_tokenized_val,X_embedded_val], y_val,
                            batch_size=batch_size)
print(acc)


#Evaluating model on test set
y_test = np.array(y_test)
test_loss, test_acc = CompleteModel.evaluate([Tokenized_test, Embedded_test], y_test)

print()
print('Test Loss:\t', test_loss)
print('Test Accuracy:\t', test_acc) #we obtain 0.825+- 0.005 on test set
predModel = CompleteModel.predict([Tokenized_test, Embedded_test])

flat_list = [item for sublist in predModel for item in sublist]
flat_list = np.array(flat_list)
#print(flat_list)

#roc curves
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, flat_list)
roc_auc = auc(fpr, tpr) 

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)'% roc_auc )

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curves for NN model with fine tuned embedded vectors: test set')
plt.legend(loc="lower right")
plt.show()

#we obtain area = 0.905+-0.003 on test set