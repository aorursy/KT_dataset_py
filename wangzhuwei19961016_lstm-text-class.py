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
from matplotlib import pyplot as plt
import jieba
data = pd.read_csv('/kaggle/input/comment/data.csv')
data.head(3)
#构建label值
def zhuanhuan(score):
    if score > 3:
        return 1
    elif score < 3:
        return 0
    else:
        return None
    
#特征值转换
data['target'] = data['stars'].map(lambda x:zhuanhuan(x))
data_model = data.dropna()
import seaborn as sns
sns.countplot(data['target'])
#切分测试集、训练集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_model['cus_comment'], data_model['target'], random_state=3, test_size=0.25)

#引入停用词
infile = open("/kaggle/input/stopword/stopwords.txt",encoding='utf-8')
stopwords_lst = infile.readlines()
stopwords = [x.strip() for x in stopwords_lst]

#中文分词
def fenci(train_data):
    words_df = train_data.apply(lambda x:' '.join(jieba.cut(x)))
    return words_df
 
x_train[:5]
#使用TF-IDF进行文本转向量处理
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(stop_words=stopwords, max_features=3000, ngram_range=(1,2))
tv.fit(x_train)
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
clf = DecisionTreeClassifier(max_depth = 2,random_state = 0)
clf.fit(tv.transform(x_train), y_train)
print(metrics.classification_report(clf.predict(tv.transform(x_test)),y_test))
# Import the AdaBoost classifier
from sklearn.ensemble import AdaBoostClassifier
#计算分类效果的准确率
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, f1_score

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)

# Train Adaboost Classifer
model1 = abc.fit(tv.transform(x_train), y_train)


#Predict the response for test dataset
y_pred = model1.predict(tv.transform(x_test))
from sklearn import metrics
print(metrics.classification_report(y_test,y_pred))

# 创建tensor
from keras.layers import Input,Dense,Embedding,Conv2D,MaxPool2D
from keras.layers import Reshape,Flatten,Dropout,Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
sequence_length=150
max_words = 1000
max_len = 150
vocabulary_size=1000
embedding_dim=150
num_filters=512
filter_sizes=[3,4,5]
drop=0.5
epochs=100
batch_size=30
print("正在创建模型...")
inputs=Input(shape=(sequence_length,),dtype='int32')
embedding=Embedding(input_dim=vocabulary_size,output_dim=embedding_dim,input_length=sequence_length)(inputs)
reshape=Reshape((sequence_length,embedding_dim,1))(embedding)

# cnn
conv_0=Conv2D(num_filters,kernel_size=(filter_sizes[0],embedding_dim),padding='valid',kernel_initializer='normal',activation='relu')(reshape)
conv_1=Conv2D(num_filters,kernel_size=(filter_sizes[1],embedding_dim),padding='valid',kernel_initializer='normal',activation='relu')(reshape)
conv_2=Conv2D(num_filters,kernel_size=(filter_sizes[2],embedding_dim),padding='valid',kernel_initializer='normal',activation='relu')(reshape)

maxpool_0=MaxPool2D(pool_size=(sequence_length-filter_sizes[0]+1,1),strides=(1,1),padding='valid')(conv_0)
maxpool_1=MaxPool2D(pool_size=(sequence_length-filter_sizes[1]+1,1),strides=(1,1),padding='valid')(conv_1)
maxpool_2=MaxPool2D(pool_size=(sequence_length-filter_sizes[2]+1,1),strides=(1,1),padding='valid')(conv_2)


concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(units=1, activation='softmax')(dropout)
model=Model(inputs=inputs,outputs=output)
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)
model.compile(optimizer='rmsprop',
             loss = 'binary_crossentropy',
             metrics=['acc'])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
%matplotlib inline
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
history=model.fit(sequences_matrix,y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#test 集
test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
#进行预测
y_pred=model.predict(test_sequences_matrix)
confu=metrics.classification_report(y_test,y_pred)
print(confu)
from keras.models import Sequential
from keras.layers import Embedding,SimpleRNN
model = Sequential()
model.add(Embedding(10000,32,input_length=150))
model.add(SimpleRNN(64))
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer='rmsprop',
             loss = 'binary_crossentropy',
             metrics=['acc'])
history=model.fit(sequences_matrix,y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
accr = model.evaluate(test_sequences_matrix,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
max_words = 1000
max_len = 150
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
%matplotlib inline
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)
sequences = tok.texts_to_sequences(x_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)
history=model.fit(sequences_matrix,y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
import matplotlib.pyplot as plt



plt.plot(history.history['loss'],label='loss')

plt.plot(history.history['val_loss'],label='val_loss')

plt.title("train & val_loss")

plt.ylabel("loss")

plt.xlabel("epoch")
plt.legend(["train","val"],loc="upper right")

plt.show()

#test 数据集
test_sequences = tok.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
#模型预测评价
accr = model.evaluate(test_sequences_matrix,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
import os
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model
from keras.layers import *
from matplotlib import pyplot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.layers import Dense,Dropout,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.layers import LSTM
def design_model():
    # design network
    inp=Input(shape=(150,))
    reshape=Reshape((150,1,1))(inp)
    conv1=Convolution2D(32,3,3,border_mode='same',init='glorot_uniform')(reshape)
    l1=Activation('relu')(conv1)
    conv2=Convolution2D(64,3,3, border_mode='same',)(l1)
    l2=Activation('relu')(conv2)
    m2=MaxPooling2D(pool_size=(4, 1), border_mode='valid')(l2)
    reshape1=Reshape((4,-1))(m2)
    lstm1=LSTM(input_shape=(10,64),output_dim=30,activation='tanh',return_sequences=False)(reshape1)
    dl1=Dropout(0.3)(lstm1)
    # den1=Dense(100,activation="relu")(dl1)
    den2=Dense(1,activation="relu")(dl1)
    model=Model(input=inp,outputs=den2)
    model.summary() #打印出模型概况
    adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
    model.compile(loss=["mae"], optimizer=adam,metrics=['accuracy'])
    return model
model=design_model()

from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)
history=model.fit(sequences_matrix,y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
#模型预测评价
accr = model.evaluate(test_sequences_matrix,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('/kaggle/input/picture/decision.png') 

plt.imshow(lena) 
plt.axis('off') 
plt.show()

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('/kaggle/input/picture/rnn.png') 

plt.imshow(lena) 
plt.axis('off') 
plt.show()