'''install packages'''
!pip install cnocr
!pip install cnstd
import os,jieba,pickle
import numpy as np
import pandas as pd
from os.path import join as joinp

'''set path'''

mod_path = '../input/w2v-cn/sgns.sogounews.bigram-char'
suicide_path = '../input/tempstrenthen/.xlsx'
s_path = '../input/sucide-text/.xlsx'
stopword_path = '../input/stopwordch/stopwordCh.txt'
path_dict = ['anger.txt','neg.txt','pos.txt']
path_dict=['../input/emotion-word/'+p for p in path_dict]
other_txt_path='../input/coversationzhihu/zhihu.txt'
pic_path='../input/pic-ocr-test/pic.jpeg'
'''load stopword'''
with open(stopword_path,'r',encoding='utf8') as f:
    stopword = [w.strip() for w in f.readlines()]
    stopword.append(' ')
'''recognize chinese character'''
from cnstd import CnStd
from cnocr import CnOcr

std = CnStd()
cn_ocr = CnOcr(cand_alphabet=None)

box_info_list = std.detect(pic_path)
TEXT=[]
for box_info in box_info_list:
    cropped_img = box_info['cropped_img']  # 检测出的文本框
    ocr_res = cn_ocr.ocr_for_single_line(cropped_img)
    print('ocr result: %s' % ''.join(ocr_res))
    TEXT.append(''.join(ocr_res))
TEXT=''.join(TEXT)
'''add vocabularies'''
def load_dict(path):
    jieba.load_userdict(path)
    with open(path,'r',encoding='utf8') as f:
        word = f.readlines()
        return [w.strip() for w in word]
    
word_lst_add = []
for path in path_dict:
    word_lst_add.extend(load_dict(path))
word_lst_add = list(set(word_lst_add))
'''enhancement'''
'''load txt suicide'''
import xlrd
def load_text_cut(path):
    import xlrd
    txt_add=xlrd.open_workbook(path)
    sheet=txt_add.sheet_by_index(0)
    s=sheet.col_values(0)
    txt,w = [],[]
    for s_ in s:
        txt.append([w for w in s_.split(' ') if w not in stopword])
    for k in txt:
        w.extend(k)
    return txt,list(set(w))
   
txt_c,word_lst_scd = load_text_cut(suicide_path)
'''get zhihu context with string len more than 5 ans escape tag'''
import re
def get_other_txt(num):
    others = []
    f = open(other_txt_path,'r')
    lines = f.readlines()
    for s in lines:
        s = re.sub('<(.*)>','',s).strip()
        pres = re.search(r"(?:_content\" \: \")(.*)(?:\")",s)
        if pres != None:
            if len(pres.group(1))>20:
                lst = jieba.cut(pres.group(1))
                others.append([w for w in list(lst) if w not in stopword])
            if len(others)>num*0.7:
                break
    word_lst_other = []
    for s in others:
        word_lst_other.extend([w for w in s if w not in stopword])
    word_lst_other = list(set(word_lst_other))
    return others,word_lst_other

txt_c_other,word_lst_other = get_other_txt(len(txt_c))
'''get word dictionary'''
word_lst = list(set(word_lst_scd+word_lst_other+word_lst_add))
d_w2i = {w:i for i,w in enumerate(word_lst)}
'''get digit data and organize'''
from keras.preprocessing.sequence import pad_sequences

LEN = 2000
def tokenizer(d_w2i,txt_c,len=LEN):
    '''given dictionary word to digt , txt cut and pad LEN. return digt tok '''
    tok = []
    for i,s in enumerate(txt_c):
        tok.append([])
        for w in s:
            try:
                if d_w2i[w] != None:
                    tok[i].append(d_w2i[w])
                else:
                    tok[i].append(0)
            except:
                tok[i].append(0)
    tok = pad_sequences(tok,LEN)
    return tok

tok_pos = tokenizer(d_w2i,txt_c)
tok_neg = tokenizer(d_w2i,txt_c_other)

tok = np.vstack((tok_pos,tok_neg))
tag = np.vstack(( np.ones((tok_pos.shape[0],1),dtype = 'int'),
                 np.zeros((tok_neg.shape[0],1),dtype = 'int')
               )) 
ind = np.array(range(len(tok)))
sample = list(zip(tok,tag,ind))
'''shuffle the samples'''
from keras.utils import to_categorical
from random import shuffle as sf
sf(sample)
tok,tag,ind = tuple(zip(*sample))
# tok,tag = np.array(tok),to_categorical(np.array(tag))
tok,tag = np.array(tok),to_categorical(np.array(tag))
'''load word embeding model'''
from keras.preprocessing.sequence import pad_sequences
import codecs
def Load_model_emb(data_dict,embMod_path):
    '''输入词典,模型路径。返回需要的可供keras.layers.Embedding使用的词向量的矩阵'''
    with codecs.open(embMod_path,'r','utf-8') as f:
        emb_mat = np.zeros((len(data_dict) + 1, 300))
        size =None
        num = 0 #记录命中词数量
        for line in f:
            line = line.strip().split(' ') ## 有毒
            if not size:
                size=line
                continue
            if line[0] in data_dict.keys():
                num+=1
                emb_mat[data_dict[line[0]]] = np.asarray(line[1:], dtype='float32')
    return emb_mat,len(data_dict)-num
Emb_mod,oov = Load_model_emb(d_w2i,mod_path)
oov
'''save data'''
import pickle
np.save('tok',np.array(tok))
np.save('tag',np.array(tag))
np.save('emb',Emb_mod)

'''build model'''
from keras.models import Sequential,save_model,load_model
from keras.layers import *
def build_mod(tok,tag,emb_mat,drop_rate=0.3):
    model = Sequential()
    # embedding with unknown method

    #Embedding 如果不会可以直接在这个模型中输入词向量试试
    model.add(Embedding(emb_mat.shape[0],emb_mat.shape[1],weights=[emb_mat],input_length=tok.shape[1],trainable = False,))
    # 随机删掉一部分
    model.add(Dropout(drop_rate))

    # GRU 模型
    model.add(GRU(2**5))
    model.add(Dropout(drop_rate))
    # 全连接
    model.add(Dense(2**6,activation='relu'))
    model.add(Dropout(drop_rate))
    
    #最后一层softmax分类
    model.add(Dense(tag.shape[1],activation = 'softmax'))

    # 添加损失函数为分类的交叉熵，使用随机梯度下降
    model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    return model
mod = build_mod(np.array(tok),np.array(tag),Emb_mod)
mod.summary()
'''split data and train'''
def sep(tok,tag,k):
    tok_train = tok[:-k]
    tag_train = tag[:-k]
    tok_test = tok[-k:]
    tag_test = tag[-k:] 
    return (tok_train,tag_train),(tok_test,tag_test)
(tok_train,tag_train),(tok_test,tag_test) = sep(tok,tag,100)
mod.fit(x=tok_train,y=tag_train, validation_split=0.2,batch_size=2**4,epochs=10,workers=4)
'''test set'''
# mod.predict(np.array([tok_train[1]]))
mod.evaluate(tok_test,tag_test)
'''try OCR text'''
txt = TEXT
if type(txt)==str:
    txt = [txt]
elif type(txt)!=list:
    print('input txt must str or list')
tok=[]
for i,t in enumerate(txt):
    tok.append([])
    txt_cut = [w for w in list(jieba.cut(t)) if w not in stopword]
    for w in txt_cut:
        if w in d_w2i.keys():
            tok[i].append(d_w2i[w])
        else:
            tok[i].append(0)
sequences = pad_sequences(np.array(tok),LEN)
mod.predict(np.array(sequences))
'''save'''
save_model(mod,'category_scd-mod.h5')
pickle.dump(d_w2i,open('category_scd-d_w2i','wb'))
import jieba
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

mod = load_model('./category_scd-mod.h5')
d_w2i = pickle.load(open('./category_scd-d_w2i','rb'))

# txt = input('please enter a sentence:\n\n')
txt = ['通过对词频进行时间序列分析，可以更详细地区分短期、长期与周期性热点；对一些更有价值的热词做热度预警；对热词的增长趋势进行分析等等。',
      '大海那么近，之前也有几人做了，为什么自己不去做，还在活着啊，能不能有个意外让我离世，算是我这一生收到的最大的礼物']
if type(txt)==str:
    txt = [txt]
elif type(txt)!=list:
    print('input txt must str or list:\n\n')
tok=[]
for i,t in enumerate(txt):
    tok.append([])
    txt_cut = [w for w in list(jieba.cut(t)) if w not in stopword]
    for w in txt_cut:
        if w in d_w2i.keys():
            tok[i].append(d_w2i[w])
        else:
            tok[i].append(0)
sequences = pad_sequences(np.array(tok),LEN)
res=mod.predict(np.array(sequences))
for k in range(len(res)):
    print("+----------------------+"+
          "\n|正常/自杀：{:.4f}/{:.4f}|\n".format(*np.round(res,decimals=3)[k])+
          "+----------------------+")
