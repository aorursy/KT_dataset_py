'''เริ่มต้นจากการ import libraries ทั้งหลายที่จำเป็นต้องใช้งานเข้ามานะครับ'''
'''บาง libraries พวกเราอาจจะยังไม่เคยใช้ แต่ถ้าเห็นวิธีการใช้บ่อยๆ ก็จะคุ้นชินไปเองครับ'''
import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)

'''Keras libraries เหล่านี้เป็นมาตรฐานที่เราแทบจะ Copy ไปใช้กับทุกปัญหาได้เลยครับ'''
# Tokenizer ใช้จัดการตัดสตริงให้เป็นคำ และ map แต่ละคำเป็นตัวเลข ท้ายที่สุดสตริงจะเปลี่ยนเป็น Sequence of numbers
from tensorflow.keras.preprocessing.text import Tokenizer
# Pad_sequences ใช้สำหรับเติม 0 ลงไปให้ทุกสตริงสุดท้ายมีความยาวเท่ากัน
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
# โหลดข้อมูล train/test โดยใช้ panda.read_csv ฟังก์ชันที่ใช้บ่อยที่สุดใน Data Science :)
train_df = pd.read_csv("../input/quora-insincere-questions-classification/train.csv")
test_df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)
train_df.head(10) # ลองดูซิ ว่าข้อมูลสอนมีหน้าตาเป็นอย่างไร
test_df.head(5) # ข้อมูลทดสอบหรือ test data จะเหมือน train ยกเว้นเราจะไม่รู้ target ที่เราต้องการทำนาย
%%time
## 1) ใช้ฟังก์ชัน train_test_split เพื่อแบ่งข้อมูลออกเป็น train/valid data
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## กำหนดค่าคงที่ต่างๆ ซึ่งสามารถปรับจูนได้เพื่อประสิทธิภาพที่ดีขึ้น ในภายหลัง
embed_size = 300 # ชนาด dimension ของ word vectors
max_features = 100000 # จำนวนคำศัพท์ที่เราจะให้ model รู้จัก
maxlen = 70 # กำหนดให้ความยาวของทุกประโยคเท่ากันที่ 70

## 2) ดึงเฉพาะข้อความกระทู้ขึ้นมา รวมทั้งจัดการ missing values อย่างง่าย
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values

## 3) ใช้ Tokenizer
#3.1) สร้าง object tokenizer ขึ้นมา ระบุคำศัพท์สูงสุดที่ต้องการจำ
tokenizer = Tokenizer(num_words=max_features)
#3.2) ให้ tokenizer เรียนรู้ศัพท์ใน training data
tokenizer.fit_on_texts(list(train_X))
#3.3) ใช้ tokenizer เปลี่ยนจาก string เป็น sequence of numbers
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## 4) Pad the sentences 
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## 5) เก็บค่า targets หรือ labels ไว้ในตัวแปร y
## เรามองเกือบทุกปัญหาใน machine learning เป็นการหาฟังก์ชันที่ดีในการ map จาก X --> y
train_y = train_df['target'].values
val_y = val_df['target'].values
'''ตอนนี้ข้อมูลเราจะอยู่ในรูป Sequence of numbers ที่ีมีความยาว maxlen = 70 แล้วครับ
สังเกตว่า เราแบ่งข้อมูล 10% ไปเป็น Validation data, ทำให้ training data เหลืออยู่ 1.175 ล้านกระทู้ครับ
'''
print('ขนาดข้อมูลหลังแปลง : ',train_X.shape, val_X.shape, train_y.shape, val_y.shape, '\n')
'''เราลอง print train_X[0] เพื่อดูตัวอย่าง sequence of numbers, สังเกตว่าเราแปะ 0 ไว้ที่ข้างหน้าเพื่อให้ความยาวครบ 70 โดยแท้จริงแล้วเราเลือกแปะ 0 ไว้ข้างหลังก็ได้เช่นกัน '''
print('ตัวอย่างข้อมูลสอนตัวแรกจะอยู่ในรูป sequence of numbers ความยาว maxlen = %d ดังนี้ : ' % (maxlen))
print(train_X[0])
'''ปัจจัยหนึ่งที่ทำให้ปัญหาคัดกรองกระทู้ ทำได้ยากก็คือ สัดส่วนของกระทู้ที่เจตนาไม่ดีนั้น มีน้อยกว่ากระทู้ปกติมาก 
(ซึ่งทำให้โมเดลของเราทำความเข้าใจรูปแบบกระทู้ที่ไม่ดีได้ลำบาก เพราะตัวอย่างส่วนใหญ่จะเป็นกระทู้ดี)

สัดส่วนของกระทู้ที่ไม่ดี มีเพียงราว 6.2% เท่านั้น
'''
print('สัดส่วนกระทู้ไม่ดีเท่ากับ %.4f ' % (sum(train_y)/len(train_y)))
'''โค้ดต่อไปนี้ เป็นโค้ดของ Keras ที่สามารถใช้สร้าง Deep Sequential Neural Network Model ขึ้นมาได้อย่างง่ายดาย 
โดย Deep Neural Networks นั้นจะประกอบไปด้วย Neural Layers ต่างๆ มาประมวลผลต่อเนื่องกัน โดยเพื่อนๆ สามารถศึกษา
รายละเอียดของ Layers ต่างๆ ได้จาก Course online เช่น DeepLearning.ai บน coursera.org หรือ course ของ ThAIKeras ในอนาคต

ลักษณะการสร้าง Layers ของ Keras จะมีรูปแบบตายตัวดังนี้

output_tensor = NewLayer(layer_parameters)(input_tensor)

โดยเราสามารถนำ Layers ต่างๆ อาทิเช่น Embedding, GRU, MaxPool, หรือ Dense มาต่อกันได้อย่างอิสระ คล้ายกับการต่อ Lego ดังโค้ดข้างล่าง
'''

## นิยามโมเดลด้วยการนำ Layers มาเชื่อมกัน
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)

## สั่งสร้าง model ด้วยการกำหนด input และ output tensors
model = Model(inputs=inp, outputs=x)

# compile คือการระบุวิธีการฝึกโมเดล ว่าต้องการ loss/metric ประเภทไหน และใช้ optimizer แบบไหน
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# print โมเดลสรุปออกมา
print(model.summary())
## Train the model 
model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)[:,0]
print(pred_noemb_val_y.shape)
print('ตัวอย่างคำทำนาย 10 กระทู้แรกใน validation data: ' ) # print ดูที่เราทำนาย 10 cases แรก
print(pred_noemb_val_y[:10])

print('ค่า target จริงของ 10 กระทู้แรก: ', val_y[:10]) # print ดู label จริง
from sklearn import metrics

thresh = 0.50
pred_noemb_val_y_int = pred_noemb_val_y > thresh
print('ค่า accuracy เท่ากับ %.4f' % (sum(pred_noemb_val_y_int == val_y)/len(val_y)))
print('ค่า F1 เท่ากับ %.4f' % (metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int)) ) )
count = np.logical_and(pred_noemb_val_y_int == val_y, val_y == 1) # นับจำนวนที่เราทำนาย class1 ถูกต้อง

print('ค่า precision : %4f '% (sum(count)/sum(pred_noemb_val_y_int == 1)))
print('ค่า recall : %4f '% (sum(count)/sum(val_y == 1)))
idx_1 = np.where(val_y == 1)[0] # หา id เฉพาะกระทู้ไม่ดี (class1) ใน validation data
print(pred_noemb_val_y[idx_1[:10]]) # print คำนาย 10 cases แรกของกระทู้ไม่ดี
print(val_y[idx_1[:10]]) # 1 ทั้งหมด (กระทู้ไม่ดีทั้งหมด)
best_thresh = 0.2
max_f1 = 0
for thresh in np.arange(0.2, 0.601, 0.01):
    thresh = np.round(thresh, 2)
    current_f1 = metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))
    print("ค่า F1 ที่ threshold %.2f is %.4f" % (thresh, current_f1) )
    
    if current_f1 > max_f1:
        best_thresh = thresh
        max_f1 = current_f1

print('ค่า threshold ที่ดีที่สุดคือ %.2f ให้ค่า F1 เท่ากับ %.4f' % (best_thresh,max_f1))
pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)
del model, inp, x
import gc; gc.collect()
time.sleep(10)
!ls ../input/quoratextemb/
'''ฟังก์ชันในการโหลด pre-trained word embedding vectors

INPUTS: 
(1) word_index : dictionary ที่ map จาก "ศัพท์ภาษาอังกฤษ" เป็น "ตัวเลข" ได้จาก keras tokenizer
(2) max_words : จำนวนคำศัพท์ที่เราต้องการ ในที่นี่้มีค่าเท่ากับ max_features
(3) embed_size ขนาด dimension ของ vectors ซึ่งเรากำหนดค่ามาตรฐานสากล = 300

OUTPUT: embedding matrix ขนาด (max_words x embed_size)

'''

def load_glove_fast(word_index, max_words=max_features, embed_size=300):
    EMBEDDING_FILE = '../input/quoratextemb/glove.840B.300d/glove.840B.300d.txt'
    emb_mean, emb_std = -0.005838499, 0.48782197

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding="utf8") as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= max_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def load_fasttext_fast(word_index, max_words=max_features, embed_size=300):
    EMBEDDING_FILE = '../input/quoratextemb/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    emb_mean, emb_std = -0.0033470048, 0.109855264

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding="utf8") as f:       
        for line in f:
            if len(line) <= 100:
                continue
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= max_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def load_para_fast(word_index, max_words=max_features, embed_size=300):
    EMBEDDING_FILE = '../input/quoratextemb/paragram_300_sl999/paragram_300_sl999.txt'
    emb_mean, emb_std = -0.0053247833, 0.49346462

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_words, embed_size))
    with open(EMBEDDING_FILE, 'r', encoding="utf8", errors="ignore") as f:        
        for line in f:
            if len(line) <= 100:
                continue
            word, vec = line.split(' ', 1)
            if word not in word_index:
                continue
            i = word_index[word]
            if i >= max_words:
                continue
            embedding_vector = np.asarray(vec.split(' '), dtype='float32')[:300]
            if len(embedding_vector) == 300:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix
%%time
'''โหลด weights ของ pre-trained vectors มาเก็บไว้ในตัวแปร'''

embedding_matrix =  load_glove_fast(tokenizer.word_index)

'''สร้าง Deep Learning Model ด้วย Keras'''
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp) # ต่างจากเดิมที่บรรทัดนี้บรรทัดเดียวครับ
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
pred_glove_val_y = model.predict([val_X], batch_size=1024, verbose=1)

best_thresh = 0.2
max_f1 = 0
for thresh in np.arange(0.2, 0.601, 0.01):
    thresh = np.round(thresh, 2)
    current_f1 = metrics.f1_score(val_y, (pred_glove_val_y>thresh).astype(int))
    print("ค่า F1 ที่ threshold %.2f is %.4f" % (thresh, current_f1) )
    
    if current_f1 > max_f1:
        best_thresh = thresh
        max_f1 = current_f1

print('ค่า threshold ที่ดีที่สุดคือ %.2f ให้ค่า F1 เท่ากับ %.4f' % (best_thresh,max_f1))
'''ทำนาย test'''
pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)
'''เคลียร์หน่วยความจำ'''
del embedding_matrix, model, x
import gc; gc.collect()
time.sleep(10)
embedding_matrix =  load_fasttext_fast(tokenizer.word_index)
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
pred_fasttext_val_y = model.predict([val_X], batch_size=1024, verbose=1)

max_f1 = 0
for thresh in np.arange(0.2, 0.601, 0.01):
    thresh = np.round(thresh, 2)
    current_f1 = metrics.f1_score(val_y, (pred_fasttext_val_y>thresh).astype(int))
    print("ค่า F1 ที่ threshold %.2f is %.4f" % (thresh, current_f1) )
    
    if current_f1 > max_f1:
        best_thresh = thresh
        max_f1 = current_f1

print('ค่า threshold ที่ดีที่สุดคือ %.2f ให้ค่า F1 เท่ากับ %.4f' % (best_thresh,max_f1))

pred_fasttext_test_y = model.predict([test_X], batch_size=1024, verbose=1)
'''เคลียร์หน่วยความจำ'''
del embedding_matrix, model, x
import gc; gc.collect()
time.sleep(10)
embedding_matrix =  load_para_fast(tokenizer.word_index)
        
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train_X, train_y, batch_size=512, epochs=3, validation_data=(val_X, val_y))
pred_paragram_val_y = model.predict([val_X], batch_size=1024, verbose=1)
max_f1 = 0
for thresh in np.arange(0.2, 0.601, 0.01):
    thresh = np.round(thresh, 2)
    current_f1 = metrics.f1_score(val_y, (pred_paragram_val_y>thresh).astype(int))
    print("ค่า F1 ที่ threshold %.2f is %.4f" % (thresh, current_f1) )
    
    if current_f1 > max_f1:
        best_thresh = thresh
        max_f1 = current_f1

print('ค่า threshold ที่ดีที่สุดคือ %.2f ให้ค่า F1 เท่ากับ %.4f' % (best_thresh,max_f1))
pred_paragram_test_y = model.predict([test_X], batch_size=1024, verbose=1)
'''เคลียร์หน่วยความจำ'''
del embedding_matrix, model, x
import gc; gc.collect()
time.sleep(10)
'''สร้าง ensemble อย่างง่ายที่นำค่าความน่าจะเป็นของทั้งสามโมเดลมาเฉลี่ยกัน เพื่อให้ได้ค่าความน่าจะเป็นสุดท้าย'''
pred_val_y = 0.33*pred_glove_val_y + 0.33*pred_fasttext_val_y + 0.34*pred_paragram_val_y 

max_f1 = 0
for thresh in np.arange(0.2, 0.601, 0.01):
    thresh = np.round(thresh, 2)
    current_f1 = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
    print("ค่า F1 ที่ threshold %.2f is %.4f" % (thresh, current_f1) )
    
    if current_f1 > max_f1:
        best_thresh = thresh
        max_f1 = current_f1

print('ค่า threshold ที่ดีที่สุดคือ %.2f ให้ค่า F1 เท่ากับ %.4f' % (best_thresh,max_f1))
pred_test_y = 0.33*pred_glove_test_y + 0.33*pred_fasttext_test_y + 0.34*pred_paragram_test_y
pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)
