import numpy as np # linear algebra
import os
from nltk.corpus import stopwords
from string import punctuation
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
#print(os.listdir("../input"))
from gensim.models.word2vec import Word2Vec
import os, json
import pandas as pd
from sklearn import preprocessing, model_selection


path_to_json = '../input/jsondata/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

df2 = pd.DataFrame(columns=['description', 'Document ID'])
#json file data to dataframe conversion
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)        
        desc = json_text['jd_information']['description']
        id= json_text['_id']
        df2.loc[index] = [desc ,id]
#print(df2)

df=pd.read_csv('../input/departmentdata/document_departments.csv')
#df.head(5)
#df.shape[0]
df["Document ID"]=df["Document ID"].astype(str)
#print(df.info())
#print(df2.info())

dfinal = df2.merge(df, on="Document ID", how = 'inner') #final dataframe 
#dfinal

y = dfinal['Department']
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
#print(dummy_y)


dfinal["description"] = dfinal['description'].str.replace('[^\w\s]','') #removing 
dfinal["token"] = dfinal["description"].str.lower().str.split()

stopwords = set(stopwords.words('english'))
stopwords_= stopwords.union(set(punctuation))
dfinal['token']=dfinal['token'].apply(lambda x: [item for item in x if item not in stopwords_])
#dfinal.head(4)
length = list(map(len,dfinal['token']))
print(length)
sentences = []
for sentence_group in dfinal['token']:
    sentences.extend(sentence_group)
    
#print(sentences)    
#print((sentences)) 

# Set values for various parameters
num_features = 90   # Word vector dimensionality
min_word_count = 20    # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 6           # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
w2vmodel = Word2Vec(sentences=sentences,sg=1, hs=0, workers=num_workers, size=num_features, min_count=min_word_count, window=context,sample=downsampling,negative=5,iter=6)

w2vmodel.init_sims(replace=True)
w2vmodel.save('train_model')

#del sentences

def get_w2v_features(w2v_model, sentence_group,num_features):
    words = "".join(sentence_group)  # words in text
    #words =np.concatenate(sentence_group)
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
    
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1
    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

dfinal['w2v'] = list(map(lambda sen_group:get_w2v_features(w2vmodel, sen_group,num_features),dfinal['token']))

x_new = dfinal['w2v'] #list of list
#print(x_new)
MAX_SEQUENCE_LENGTH = 100
x_final = pad_sequences(x_new, maxlen=MAX_SEQUENCE_LENGTH) #padding the input sequence
#print(len(x_new))
#print(train_data.shape)
#x = sequence.pad_sequences(x, maxlen=100)
#train test split
train_x, test_x, train_y, test_y = model_selection.train_test_split(x_final,dummy_y,test_size = 0.2, random_state = 0)

#print(len(dfinal['token'][0]))
#len(dfinal['w2v'][4])

#creation kera NN model >> input neuron 100, output neuron 
model = Sequential()
model.add(Dense(100, input_dim = 100 , activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
#model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 10, batch_size = 2)

scores = model.evaluate(test_x, test_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

