import pandas as pd

import numpy as np
train_data = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
train_data.head()
train_data["words_text"] = [ str(x).split() for x in train_data.text ]

train_data["words_selected_text"] = [ str(x).split() for x in train_data.selected_text ]
train_data.head()
import re



def clean(row):

    row = row.replace('.', ' ')

    row = row.replace(',', '')

    row = row.replace("'", "")

    row = re.sub("\d+", "<NUM>", row)

    row = re.sub("\*+", "<CURSE>", row)

    row = re.sub("^@.*", "<USER>", row)

    row = re.sub("^#.*", "<HASH>", row)

    row = re.sub("^((https|http|ftp|file)?:\/\/).*", "<LINK>", row)

    row = re.sub("[0-9]+:[0-9]+(am|AM|pm|PM)?", "<DATE>", row)

    row = row.lower().strip()

    return row.split()

train_data["words_text"] = train_data.text.apply(lambda row: clean(str(row)))

train_data["words_selected_text"] = train_data.selected_text.apply(lambda row: clean(str(row)))
train_data.head()
## Spelling correction
# from spellchecker import SpellChecker

# spell = SpellChecker()

# from spellchecker import SpellChecker

# spell = SpellChecker()





# def spelling_correction(row) : 

    

#     constant = ["<curse>", "<num>", "<user>", "<hash>", '<link>', '<date>']

#     temp = [ spell.correction(word) if word not in constant else word for word in row ]

    

#     return temp

    
#train_data["words_text"] = [ spelling_correction(row) for row in train_data.words_text ]
#train_data["words_selected_text"] = [ spelling_correction(row) for row in train_data.words_selected_text ]
train_data.head()
#train_data.to_csv("/kaggle/input/tweet-sentiment-extraction/spell_correct_train.csv")
train_data2 = pd.read_csv("/kaggle/input/spell-correct/spell_correct_train.csv")
train_data2.head()
train_data2.words_text[0]
train_data2.words_text[0][0]
import ast



train_data2.words_text = [ ast.literal_eval(str(x)) for x in train_data2.words_text ]

train_data2.words_selected_text = [ ast.literal_eval(str(x)) for x in train_data2.words_selected_text]
train_data2.head()
del train_data2['Unnamed: 0']
del train_data2['Unnamed: 0.1']
train_data2.head()
train_data2.words_text[0]
train_data2.words_text[0][0]
import difflib as diff





# def first_matching_index(text,selected_text) :

#     try :

#         return  text.index(diff.get_close_matches(selected_text[0],text)[0])

#     except :

#         return  None

        



# def last_matching_index(text,selected_text) :

#     length = len(selected_text)

#     try : 

#         return text.index(diff.get_close_matches(selected_text[length-1],text)[0])

#     except :

#         return None

    

import difflib as diff



def matching_index_search(text,selected,index):

    text = list(text)

    selected = list(selected)

    return text.index(diff.get_close_matches(selected[index],text,cutoff=0)[0])

train_data2["start_indices"] = train_data2.apply(lambda x: matching_index_search(x.words_text,x.words_selected_text,0),axis=1)

train_data2["end_indices"] = train_data2.apply(lambda x: matching_index_search(x.words_text,x.words_selected_text,-1),axis=1)

train_data2.head()

data = pd.read_csv("/kaggle/input/temp-file/try_submission.csv")
#temp1 = [ first_matching_index(x.text_split,x.selected_text_split) for x in  train_data2 ]



# train_dataCp = train_data2.copy()



# train_dataCp["start_indices"] = train_data2.apply(lambda x : first_matching_index(x.text_split,x.selected_text_split), axis = 1 )



# train_dataCp["end_indices"] = train_data2.apply(lambda x : last_matching_index(x.text_split,x.selected_text_split), axis = 1 )
train_data2.iloc[49]
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
fig = plt.figure(figsize = (30,10))

sns.heatmap(train_data2.isnull())
# train_dataCp.drop(["initial_indice"],axis = 1, inplace = True)
# fig = plt.figure(figsize = (30,10))

# sns.heatmap(train_dataCp.isnull())
# null_start_indices = train_dataCp[train_dataCp['start_indices'].isnull()].index.tolist()

# null_end_indices = train_dataCp[train_dataCp['end_indices'].isnull()].index.tolist()
# (len(null_start_indices),len(null_start_indices))
train_data2.iloc[49]
# len([ x  for x in  null_end_indices if x not in null_start_indices])
train_data2.head()
train_dataCp = train_data2[ train_data2.start_indices <= train_data2.end_indices ]
train_dataCp.head()
train_dataCp.to_csv("range_data.csv")
import nltk

import pandas as pd 

import ast

import tensorflow

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
train_data3 =   train_dataCp.copy() 
train_data3.head()
#train_data3["words_text"] = train_data3.words_text.apply(lambda x: ast.literal_eval(x))
train_data3["words_text"][0][0]
dictionary = []

for words in train_data3.words_text :

    dictionary.extend(words)

    

dictionary = [ word for word in dictionary if word.isalnum() ]
whole_text = " ".join(dictionary)
tokens = nltk.word_tokenize(whole_text)
(len(tokens),len(dictionary))
tokenizer = Tokenizer(num_words=20000,oov_token="<OOV>")



tokenizer.fit_on_texts(train_data3.words_text)

tokenized_text = tokenizer.texts_to_sequences(train_data3.words_text)

tokenized_selected_text = tokenizer.texts_to_sequences(train_data3.words_selected_text)
len(tokenizer.word_index)
tokenizer.word_index["this"]
pad_token_text = pad_sequences(tokenized_text,padding = "post")
pad_token_text[0]
len(pad_token_text[0])
pad_token_selected_text = pad_sequences(tokenized_selected_text,padding="post")
pd.DataFrame(pad_token_text).to_csv("pad_token_data.csv",header=None,index=None)
train_data3.to_csv("tokenized_form.csv",index=None)
with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout, BatchNormalization, Flatten

from tensorflow.keras.regularizers import l2, l1, l1_l2

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_similarity_score
df = pd.read_csv("tokenized_form.csv")

targets = df[["start_indices","end_indices"]]

targets.head()
training = pd.read_csv("pad_token_data.csv",header= None)

training.head()
x_train, x_test, y_train, y_test = train_test_split(training.values, targets.values, test_size=0.2, random_state=42)
def Baseline(vocab_size):

    model = Sequential([

        Embedding(vocab_size, 128, input_length=33),

        Bidirectional(GRU(128, return_sequences=True, dropout=0.8, recurrent_dropout=0.8)),

        Bidirectional(GRU(128,return_sequences=True, dropout=0.8, recurrent_dropout=0.8)),

        BatchNormalization(),

        Dense(64, activation='elu',kernel_regularizer=l1_l2()),

        Dropout(0.8),

        Dense(2, activation='elu'),

        Flatten(),

        Dense(2, activation='elu')



    ])

    return model
vocab = 20000

model = Baseline(vocab)

es = EarlyStopping(patience=5)

mcp_save = ModelCheckpoint('tweet_sentiment_model.hdf5', save_best_only=True, monitor='val_mse')

model.compile(loss="mse",optimizer="adam",metrics=['mse',"mae"])

model.summary()

## finally submission
import pandas as pd

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout, BatchNormalization, Flatten

from tensorflow.keras.regularizers import l2, l1, l1_l2

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_similarity_score

from tensorflow.keras.preprocessing.sequence import pad_sequences

import re

import numpy as np

import pickle
data_test4 = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

with open('tokenizer.pickle', 'rb') as handle:

    tokenizer = pickle.load(handle)

def clean(row):

    row = row.replace('.', ' ')

    row = row.replace(',', '')

    row = row.replace("'", "")

    row = re.sub("\d+", "<NUM>", row)

    row = re.sub("\*+", "<CURSE>", row)

    row = re.sub("^@.*", "<USER>", row)

    row = re.sub("^#.*", "<HASH>", row)

    row = re.sub("^((https|http|ftp|file)?:\/\/).*", "<LINK>", row)

    row = re.sub("[0-9]+:[0-9]+(am|AM|pm|PM)?", "<DATE>", row)

    row = row.lower().strip()

    return row.split()




data_test4["test_text_split"] = data_test4.text.apply(lambda row: clean(str(row)))

test_tokenized_text = tokenizer.texts_to_sequences(data_test4.test_text_split)
test_pad_token_text = pad_sequences(test_tokenized_text,maxlen=33, padding = "post")
test_pad_token_text[0]
def Baseline(vocab_size):

    model = Sequential([

        Embedding(vocab_size, 128, input_length=33),

        Bidirectional(GRU(128, return_sequences=True, dropout=0.8, recurrent_dropout=0.8)),

        Bidirectional(GRU(128,return_sequences=True, dropout=0.8, recurrent_dropout=0.8)),

        BatchNormalization(),

        Dense(64, activation='elu',kernel_regularizer=l1_l2()),

        Dropout(0.8),

        Dense(2, activation='elu'),

        Flatten(),

        Dense(2, activation='elu')



    ])

    return model
model = Baseline(20000)

model.load_weights("/kaggle/input/tweeter-model/tweet_sentiment_model.hdf5")
results = model.predict(test_pad_token_text)

results
results = np.round(results)

results
sum(results[0])
data_test4["final_split"] = data_test4.text.apply(lambda x: x.split())
def add_selected_text(split_text,indices):

    try:

        return " ".join(split_text[int(indices[0][0]):int(indices[0][1])])

    except:

        return " ".join(split_text)
data_test4["selected_text"] = data_test4.apply(lambda x: add_selected_text(x.test_text_split,results), axis=1)

fig = plt.figure(figsize = (30,10))

sns.heatmap(data_test4.isnull())

data_fn = data_test4.copy()

data_test4 = data.copy()
#data_test4["selected_text"] = data_test4.apply(lambda x: add_selected_text(x.test_text_split,results), axis=1)
data_test4.to_csv("submission.csv",index=None,columns=["textID","selected_text"])
# data_test4.head()
# fig = plt.figure(figsize = (30,10))

# sns.heatmap(data_test4.isnull())