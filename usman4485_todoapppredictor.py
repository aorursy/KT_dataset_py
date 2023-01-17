FILENAME = '/kaggle/input/smart-todo-appdataset/2 year dataset.xlsx'
import pandas as pd

import numpy as np
data = pd.read_excel(FILENAME)
data
def A(kk):

    if pd.isna(kk):

        return 0

    else:

        return kk

data['rice']=data['rice'].map(A) #eliminating the NaN value
data.iloc[9,:] 
col = list(data.columns)
col
for index,val in enumerate(col):

    col[index] = val.replace(" ","")
col
data.columns = col
data.columns
dataToConvert = data.iloc[:,1:-1] #eliminating the date and prac column

dataToConvert
columns = np.asarray(dataToConvert.columns[:]) #getting all columns

columns
sentence = []

for i in range(len(dataToConvert)):

    x = dataToConvert.iloc[i,:].values

    s_s = ''

    s_s +=" "+str(i+1) #addig the date as from day 0 to day 730

    for index,value in enumerate(x):

        if value == 1.0:

            s_s+=" "+columns[index]

    sentence.append(s_s)
sentence[0:5]
sentence[300:305]
sentence[-5:-1]
import keras

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense,LSTM,Dropout,Embedding

from keras.optimizers import Adam,rmsprop

from keras import Input,Model

from keras.utils import to_categorical
sentence
lengthh = len(sentence)

for index in range(lengthh):

    sentence[index] +=' stop'

    if 'wings' in sentence[index]:

        print(True)
sentence
token = Tokenizer(num_words=10000)

token.fit_on_texts(sentence)
tokenized_text = token.texts_to_sequences(sentence)
tokenized_text[0:3]
token.index_word

token_index = {}

token_index = {value:key for (key,value) in token.index_word.items()}

token_index
MAX = 0

MIN = 200

for i in tokenized_text:

    if len(i) > MAX:

        MAX = len(i)

    if len(i) < MIN:

        MIN = len(i)

print("MAX and MIN length of the sentence is ==> {maxs} & {mins}".format(maxs=MAX,mins=MIN))
single_sentence = ''

for i in sentence:

    single_sentence += i
single_sentence
all_single_tokens = []

for _token in tokenized_text:

    for j in _token:

        all_single_tokens.append(j) #adding all key words in a single array
print("The expected samples of X are ===> {samples}".format(samples=int(len(all_single_tokens)-MIN)))
#samples = int(len(all_single_tokens)/MIN)

xdata = []

ydata = []

for c in range(len(all_single_tokens)-9):

    xdata.append(all_single_tokens[c:c+MIN])

    ydata.append(all_single_tokens[c+MIN])
MAX = 0

for i in xdata:

    if len(i) > MAX:

        MAX = len(i)

MIN = MAX

for i in xdata:

    if len(i) < MIN:

        MIN = len(i)

print("MAX and MIN length of the sentence is ==> {maxs} & {mins}".format(maxs=MAX,mins=MIN))
numpy_xdata = np.asarray(xdata)

reshape_xdata = np.reshape(xdata,(numpy_xdata.shape[0],numpy_xdata.shape[1],1))
numpy_xdata.shape,reshape_xdata.shape
from keras.utils import to_categorical

enc_ydata = to_categorical(ydata)
enc_ydata.shape
from keras.models import Sequential

from keras import layers

from keras.layers import Dense, Dropout,LSTM

from keras import Input

from keras.models import  Model
def BaseLine():

    inpt = Input(shape=(reshape_xdata.shape[1],1))

    x = LSTM(512,return_sequences=True)(inpt)

    x = LSTM(100)(x)

    x = Dense(60)(x)

    outpt = Dense(enc_ydata.shape[-1], activation='softmax')(x)

    model = Model(inpt,outpt)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model
xmodel = BaseLine()
xmodel.fit(reshape_xdata,enc_ydata,epochs=10,batch_size=500)
sample = 'apple buger donuts garlic icecream jelly mango pasta quicee'

inpt_sample = sample.split()

tokened = [token_index[i] for i in inpt_sample]

i = np.asarray(tokened).reshape(1,9,1)
count = 0

output_samples = []

output_sample = sample

while True:

    output = xmodel.predict(i) 

    word = token.index_word[output.argmax()]

    output_sample = output_sample+" "+word

    sample = sample+" "+word

    print("{} ===> {}".format(sample,word))

    if word == 'stop':count+=1

    if count == 5:break #generating todo for 5 days

    inpt_sample = sample.split()[-9:]

    tokened = [token_index[i] for i in inpt_sample]

    i = np.asarray(tokened).reshape(1,9,1)

    
output_sample
output_tokened = [token_index[i] for i in output_sample.split()]
x = []

day_samples = []

for i in output_tokened:

    x.append(i)

    if i == 1:

        day_samples.append(x[1:-1]) #removing date

        x = []


day_samples
final_output = []

tokened_column = [token_index[a] for a in columns]

year = '-2020'

month = '-jan'

for _ in range(5):

    day = x.append(str(_)+month+year)

    for col_tok in tokened_column:

        x.append(1) if col_tok in day_samples[_] else x.append(0)

    final_output.append(x)

    x = []

final_output
len(data.columns),data.columns,len(columns),columns
for i in range(5):

    print(len(final_output))

pd.DataFrame(final_output,columns=data.columns[:-1])