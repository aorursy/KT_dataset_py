

!pip install tensorflow-gpu==2.0.0-beta1

!pip install tensorflow-datasets

!pip install bert-for-tf2



import tensorflow as tf

import tensorflow_hub as hub

import tensorflow_datasets as tfds

import unidecode

import time

import numpy as np

import matplotlib.pyplot as plt

import collections

import unicodedata



import os

from bert import BertModelLayer

from bert.loader import StockBertConfig, load_stock_weights
import re

from bert.loader import map_to_stock_variable_name

!wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

!unzip chinese_L-12_H-768_A-12
config = tfds.translate.wmt.WmtConfig(

    description="WMT 2019 translation task dataset.",

    version="0.0.3",

    language_pair=("zh", "en"),

    subsets={

        tfds.Split.TRAIN: ["newscommentary_v13"],

        tfds.Split.VALIDATION: ["newsdev2017"],

    }

)



builder = tfds.builder("wmt_translate", config=config)

print(builder.info.splits)

builder.download_and_prepare()

datasets = builder.as_dataset(as_supervised=True)

print('datasets is {}'.format(datasets))
train_set=datasets["train"]

val_set=datasets["validation"]
vocab_file = 'vocab_en'

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(

      (en.numpy() for zh, en in train_set), target_vocab_size=2 ** 13)

tokenizer_en.save_to_file('vocab_en')
def convert_by_vocab(vocab,items):

    output=[]

    for item in items:

        output.append(vocab[item])

    return outputjust
def load(vocab_file):

    vocab = collections.OrderedDict()

    index = 0

    with tf.io.gfile.GFile(vocab_file, "r") as reader:

        while True:

            token=convert_to_unicode(reader.readline())

                

            if not token:

                break

            token = token.strip()

            vocab[token] = index

            index += 1

    return vocab

def convert_to_unicode(text):

    if isinstance(text,str):

        return text

    if isinstance(text,bytes):

        return text.decode('utf-8','ignore')

    else:

        pass

                           
file='chinese_L-12_H-768_A-12/vocab.txt'
v=load(file)
inv_v={v:k for k,v in v.items()}
class Tokenizer():

    def __init__(self,vocab_file):

       

        self.vocab=load(vocab_file)

        self.inv_vocab={v:k for k,v in self.vocab.items()}



    def whitespace_tokenize(self,text):

        text=text.strip()

        output=[]

        token=text.split()

    

        return token 

    

    def remove_diacritics(self,text):

        return "".join(token for token in unicodedata.normalize("NFD",text)

                       if unicodedata.category(token) != "Mn")

    



        

    def remove_punc(self ,text):

        output=[]

        chars=list(text)

        i=0

        x=True

        while i < len(chars):

            char = chars[i]

            if self._is_punc(char):

                output.append([char])

                x=True

            else:

                if x:

                    output.append([])

                x=False

                output [-1].append(char)

            i+=1

        return ["".join(x) for x in output]

    

    def _is_punc(self,char):

        cp = ord(char)

        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or



(cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):



            return True



        cat = unicodedata.category(char)



        if cat.startswith("P"):

            return True

        return False

    

    def tokenize_chinese_char(self,text):

        output=[]



        for char in text:

            cp=ord(char)

            if ((cp >= 0x4E00 and cp <= 0x9FFF) or  (cp >= 0x3400 and cp <= 0x4DBF) or  (cp >= 0x20000 and cp <= 0x2A6DF) or  

            (cp >= 0x2A700 and cp <= 0x2B73F) or (cp >= 0x2B740 and cp <= 0x2B81F) or   (cp >= 0x2B820 and cp <= 0x2CEAF) or  

            (cp >= 0xF900 and cp <= 0xFAFF) or   (cp >= 0x2F800 and cp <= 0x2FA1F)):  

                output.append(" ")

                output.append(char)

                output.append(" ")

            else:

                output.append(char)

        return "".join(output)

    

    

    def clean_text(self ,text):

        output=[]

        for char in text:

            cp=ord(char)

            if cp == 0 or cp == 0xfffd or unicodedata.category(char).startswith("C"): #oxffd  for unknown 

                continue

            if char == " " or char == "\n" or char == "\t" or char == "\r" or unicodedata.category(char)=="Zs":

                output.append(" ")

            else:

                output.append(char)

        

        return "".join(output)

    

    



        

    

    def basic_tokenize(self, text):

        

        if isinstance(text,bytes):

            text=text.decode('utf-8','ignore')

        else:

            text=text

            

        text = self.clean_text(text)

        text = self.tokenize_chinese_char(text)

        text = self.whitespace_tokenize(text)

        split_tokens=[]

        for token in text:

            token=token.lower()

            token = self.remove_diacritics(token)

            split_tokens.extend(self.remove_punc(token))

        return self.whitespace_tokenize(" ".join(split_tokens))

        

    

    def wordpiece_tokenize(self,text):

        output=[]

        unk='[UNK]'

        if isinstance(text,bytes):

            text=text.decode('utf-8','ignore')

        else:

            text=text

        for token in self.whitespace_tokenize(text):#split() should also work

            chars=list(token)

            if len(chars) > 200:

                output.append(unk)

                continue

            is_bad=False

            start=0

            sub_tokens=[]

            while start < len(chars):

                end=len(chars)

                curr_substr = None

                while start < end:

                    substr= "".join(chars[start: end])

                    if start >0:

                        substr="##"+substr

                    if substr in self.vocab:

                        curr_substr=substr

                        break

                    end -= 1

                if curr_substr is None:

                    is_bad=True

                    break

                sub_tokens.append(curr_substr)

                start=end

            if is_bad:

                output.append(unk)

            

            else:

                output.extend(sub_tokens)

                

        return output

    

    def convert_by_ids(self,token):

        x=[]

        [x.append(v[i]) for i in token]

        return x

    

    def convert_by_token(self,ids):

        x=[]

        [x.append(inv_v[i]) for i in ids]

        return x

        

    def TOKENIZER(self,text):

        output=[]

        for token in self.basic_tokenize(text):

            for sub_token in self.wordpiece_tokenize(token):

                output.append(sub_token)

        return output

                    

                

    

                

    

    

    

            
tokenizer_zh=Tokenizer(file)
test_tokens = tokenizer_zh.TOKENIZER('埃隆·马斯克创建primo特斯拉')
print(test_tokens)##og
test_tokens
test_ids = tokenizer_zh.convert_by_ids(test_tokens)

print(test_ids)

print(tokenizer_zh.convert_by_token(test_ids))
def encode(zh,en,seq_len=128):

    tokens_zh=tokenizer_zh.TOKENIZER(tf.compat.as_text(zh.numpy()))

    lang1=tokenizer_zh.convert_by_ids(['[CLS]']+tokens_zh+['[SEP]'])

    if len(lang1) < seq_len:

        lang1=lang1+list(np.zeros(seq_len-len(lang1),'int32'))

    lang2=[tokenizer_en.vocab_size]+tokenizer_en.encode(tf.compat.as_text(en.numpy()))+[tokenizer_en.vocab_size+1]

    if len(lang2) < seq_len:

        lang2=lang2+list(np.zeros(seq_len-len(lang2),'int32'))

    return lang1,lang2

def filter_by_max_len(x,y,seq_len=128):

    return tf.logical_and(tf.size(x)<=seq_len,tf.size(y)<=seq_len)

    
buffer_size=50000

batch_size=64

train_dataset=train_set.map(lambda zh,en : tf.py_function(encode,[zh,en],[tf.int32,tf.int32]))

train=train_dataset.filter(filter_by_max_len)

train_dataset = train_dataset.shuffle(buffer_size).padded_batch(

    batch_size, padded_shapes=([-1], [-1]), drop_remainder=True)

train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)



val_dataset = val_set.map(

    lambda zh, en: tf.py_function(encode, [zh, en], [tf.int32, tf.int32]))

val_dataset = val_dataset.filter(filter_by_max_len)

val_dataset = val_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]))

 


for zh, en in train_dataset.take(2):

    print(zh.numpy())

    print(en.numpy())
zh.numpy()[1].shape
def get_angles(pos,i,d_model):

    return pos*(1/np.power(10000,(2*(i//2))/np.float32(d_model)))
def positional_encoding(position ,d_model):

    get_rads=get_angles(np.arange(position)[: ,np.newaxis],#position is multiplied by 

                           np.arange(d_model) [np.newaxis, :],#1arange(d_model) divided by the value of d_model

                           d_model)

    sines = np.sin(get_rads[:,0::2])# sin is applied to all values with  even no.

    cosines = np.cos(get_rads[:,1::2])# cos is applied to all values with odd  no.

    pos_encoding = np.concatenate([sines,cosines],axis=-1)

    pos_encoding = pos_encoding[np.newaxis,...]

    return tf.cast(pos_encoding,dtype=tf.float32)   
pos,d_m=5,12

p,i=np.arange(pos),np.arange(d_m)

P=p[:,np.newaxis]

I=i[np.newaxis,:]

ar=get_angles(P,I,d_m)
np.sin(ar[:,0::2])

a=np.random.random([2,3])

b=np.random.random([2,3])
c=tf.random.normal([2,3])

tf.math.equal(c,0)[:,np.newaxis,np.newaxis,:]

c.shape[1]
(1-tf.linalg.band_part(tf.ones((3,3)),-1,1))*-1e9
def create_padding_mask(seq):

    out=tf.cast(tf.math.equal(seq,0),tf.float32)

    # 0 padded elements will be shown as False

    return out[:,np.newaxis,np.newaxis,:]
def look_ahead_mask(size):

    mask=1-tf.linalg.band_part(tf.ones((size,size)),-1,0)

    return mask
def self_attention(q,k,v,mask):

    matmul_qk=tf.matmul(q,k, transpose_b=True)

    dk=tf.cast(tf.shape(k)[-1],tf.float32)

    scaled_attention=matmul_qk/tf.math.sqrt(dk)

    if mask is not None:

        scaled_attention += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention,axis=-1)

    output = tf.matmul(attention_weights,v)

    return output, attention_weights
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self,d_model,num_heads):

        super(MultiHeadAttention,self).__init__()

        self.d_model=d_model

        self.num_heads=num_heads

        assert d_model % self.num_heads == 0

        self.depth=d_model//self.num_heads

        self.wk=tf.keras.layers.Dense(d_model)

        self.wq=tf.keras.layers.Dense(d_model)

        self.wv=tf.keras.layers.Dense(d_model)

        self.dense=tf.keras.layers.Dense(d_model)

        

    def split_heads(self,x,batch_size):

        

        x=tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))

        return tf.transpose(x,perm=[0,2,1,3]) #batch size , number of heads, unspecified, depth

    

    def call(self,v,k,q,mask):

        #print ("shape of q : {}".format(tf.shape(q)))

        batch_size=tf.shape(v)[0]

        q=self.wq(q) # batch size , seq_len, d_model

        #print ("shape of layer q : {}".format(tf.shape(q)))

        k=self.wk(k)

        v=self.wv(v)

        

        q=self.split_heads(q ,batch_size) #batch size , num heads ,seq len ,depth

        #print ("shape of splited q : {}".format(tf.shape(q)))

        k=self.split_heads(k, batch_size)

        v=self.split_heads(v, batch_size)

        

        scaled_attention, attention_weights=self_attention(q,k,v,mask)

        scaled_attention=tf.transpose(scaled_attention,perm=[0,2,1,3])

        concat_attention=tf.reshape(scaled_attention,(batch_size,-1,self.d_model))

        output=self.dense(concat_attention)

        return output, attention_weights

        

    

    
def point_wise_feed_forward_network(d_model, dff):

    return tf.keras.Sequential([

        tf.keras.layers.Dense(dff, activation="relu"),

        tf.keras.layers.Dense(d_model)])



                               
def build_encoder(config_file):

    with tf.io.gfile.GFile(config_file) as reader:

        stock_params=StockBertConfig.from_json_string(reader.read())

        bert_params=stock_params.to_bert_model_layer_params()

    return BertModelLayer.from_params(bert_params,name='bert')
class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self,d_model,num_heads,dff, rate=0.1):

        super(DecoderLayer,self).__init__()

        self.d_model=d_model

        self.num_heads=num_heads

        self.dff=dff

        self.rate=rate

        self.mha1=MultiHeadAttention(d_model,num_heads)

        self.mha2=MultiHeadAttention(d_model,num_heads)

        self.ffn =point_wise_feed_forward_network(d_model,dff)

        self.layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.layernorm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.layernorm3=tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1=tf.keras.layers.Dropout(rate)

        self.dropout2=tf.keras.layers.Dropout(rate)

        self.dropout3=tf.keras.layers.Dropout(rate)

        

    def call(self ,x ,enc_output, training,

             look_ahead_mask, padding_mask):

        attn1,attn1_w=self.mha1(x,x,x,look_ahead_mask)

        out=self.dropout1(attn1, training=training)

        out1=self.layernorm1(out+x)

        attn2,attn2_w=self.mha2(enc_output,enc_output,out1, padding_mask)

        out2=self.dropout2(attn2, training=training)

        out2=self.layernorm2(out+x)

        ffn=self.ffn(out2)

        ffn_=self.dropout3(ffn, training=training)

        ffn_=self.layernorm3(ffn+out2)

        return ffn_,attn1_w,attn2_w
class Decoder(tf.keras.layers.Layer):

    def __init__(self,num_layers,d_model,num_heads,

                 dff,target_vocab_size,rate=0.1):

        super (Decoder,self).__init__()

        self.num_layers=num_layers

        self.d_model=d_model

        self.embedding=tf.keras.layers.Embedding(

            target_vocab_size,d_model)

        self.pos_encoding=positional_encoding(

            target_vocab_size,d_model)

        self.dec_layers=[DecoderLayer(d_model,num_heads,dff, rate) for _ in range(num_layers)]

        self.dropout=tf.keras.layers.Dropout(rate)

    def call(self ,x ,enc_output,

             training,look_ahead_mask, padding_mask):

        seq_len=tf.shape(x)[1]

        attention_weights={}

        x=self.embedding(x)

        x*=tf.math.sqrt(tf.cast(self.d_model,tf.float32))

        x=self.dropout(x, training=training)

        for i in range (self.num_layers):

            x,block_1,block_2=self.dec_layers[i](x,enc_output,

                                                 training,look_ahead_mask, padding_mask)

            attention_weights["decoder_layer{}_block1".format(i+1)]=block_1

            attention_weights["decoder_layer{}_block2". format (i+1)]=block_2

        return x, attention_weights

        
sample_decoder_layer = DecoderLayer(512, 8, 2048)

sample_encoder_output = tf.random.uniform((64, 128, 768))



sample_decoder_layer_output, _, _ = sample_decoder_layer(

    tf.random.uniform((64, 50, 512)), sample_encoder_output,

    False, None, None)



sample_decoder_layer_output.shape  
sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, 

                         dff=2048, target_vocab_size=8000)



output, attn = sample_decoder(tf.random.uniform((64, 26)), 

                              enc_output=sample_encoder_output, 

                              training=False, look_ahead_mask=None, 

                              padding_mask=None)



output.shape, attn['decoder_layer2_block2'].shape
class Config():

    def __init__(self,num_layers,d_model,dff,num_heads):

        self.num_layers=num_layers

        self.d_model=d_model

        self.dff=dff

        self.num_heads=num_heads

        
class Transformer (tf.keras.Model):

    def __init__(self,config,

                 target_vocab_size,bert_config_file,

                 bert_training=True,rate=0.1,name="transformer"):

        super(Transformer,self).__init__(name=name)

        self.encoder=build_encoder(bert_config_file)

        self.encoder.trainable=bert_training

        self.decoder=Decoder(config.num_layers,config.d_model,config.num_heads,

                             config.dff,target_vocab_size,rate)

        self.final_layer=tf.keras.layers.Dense(target_vocab_size)

        

    def load_stock_weights(bert:BertModelLayer,ckpt_file):

        assert isinstance(bert,BertModelLayer)

        assert tf.compat.v1.train.check_point_exists(ckpt_file)

        ckpt_reader = tf.train.load_vocab(ckpt_file)

        bert_prefix="transformer/bert"

        weights=[]

        for weight in bert.weights:

            stock_name=map_to_stock_variable_name(weight.name,bert_prefix)

            if ckpt_reader.has_tensor(stock_name):

                value=ckpt_reader.get_tensor(stock_name)

                weights.append(value)

            else:

                raise ValueError ("no  value ")

        bert.set_weights(weights)

        

    def restore_encoder(self,bert_ckpt_file):

        self.load_stock_weights(bert_ckpt_file)

        

    def call(self,inp ,tar , training,look_ahead_mask,dec_padding_mask):

        enc_output=self.encoder(inp,training=self.encoder.trainable)

        dec_output, attention_weights=self.decoder(tar,enc_output, training,look_ahead_mask,dec_padding_mask)

        output=self.final_layer(dec_output)

        return output , attention_weights

    

        
target_vocab_size = tokenizer_en.vocab_size + 2

dropout_rate = 0.1

config = Config(num_layers=6, d_model=256, dff=1024, num_heads=8)
MODEL_DIR = "chinese_L-12_H-768_A-12"

bert_config_file = os.path.join(MODEL_DIR, "bert_config.json")

bert_ckpt_file = os.path.join(MODEL_DIR, "bert_model.ckpt") 

transformer=Transformer (config , target_vocab_size,bert_config_file)

inp = tf.random.uniform((64, 128))

tar_inp = tf.random.uniform((64, 128))

fn_out, _ = transformer(inp, tar_inp, 

                        True,

                        look_ahead_mask=None,

                        dec_padding_mask=None)

print(tar_inp.shape)

fn_out.shape
transformer.summary()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self ,d_model,warmup_steps=4000):

        super(CustomSchedule,self).__init__()

        self.d_model=d_model

        self.d_model=tf.cast(self.d_model,tf.float32)

        self.warmup_steps=warmup_steps

    def __call__(self,step):

        arg1=tf.math.rsqrt(step)

        arg2=step*(self.warmup_steps**-1.5)

        return tf.math.rsqrt(d_model)*tf.math.minimum (arg1,arg2)
learning_rate=CustomSchedule(config.d_model)

optimizer=tf.keras.optimizers.Adam(learning_rate,beta_1=0.9,beta_2=0.98,epsilon=1e-9)

loss_object=tf.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")
def loss_function(real,pred):

    mask=tf.math.logical_not(tf.math.equal(real,0))

    loss_=loss_object(real,pred)

    mask=tf.cast(tf.dtype(loss_))

    loss*=mask

    return tf.reduce_sum(loss)
train_loss=tf.keras.metrics.Mean(name="train_loss")

train_accuracy=tf.keras.metrics.SparseCategoricalCrossentropy(name="train_loss")
checkpoint_path = "./checkpoints/train"



ckpt = tf.train.Checkpoint(transformer=transformer,

                           optimizer=optimizer)



ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:

    ckpt.restore(ckpt_manager.latest_checkpoint)

    print('Latest checkpoint restored!!') 
def create_masks(inp,tar):

    dec_padding_mask=create_padding_mask(inp)

    look_head_mask=look_ahead_mask(tf.shape(tar)[1])

    dec_target_padding_mask=create_padding_mask(tar)

    combine=tf.maximum(dec_target_padding_mask,look_head_mask)

    return combine,dec_padding_mask
def train_step(inp, tar):

    tar_inp = tar[:, :-1]

    tar_real = tar[:, 1:]



    combined_mask, dec_padding_mask = create_masks(inp, tar_inp)



    with tf.GradientTape() as tape:

        predictions, _ = transformer(inp, tar_inp, 

                                     True,

                                     combined_mask,

                                     dec_padding_mask)

        loss = loss_function(tar_real, predictions)



    gradients = tape.gradient(loss, transformer.trainable_variables)

    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))



    train_loss(loss)

    train_accuracy(tar_real, predictions)
import time
EPOCHS = 4



for epoch in range(EPOCHS):

    start = time.time()



    train_loss.reset_states()

    train_accuracy.reset_states()



    # inp -> chinese, tar -> english

    for (batch, (inp, tar)) in enumerate(train_dataset):

        train_step(inp, tar)



        if batch % 500 == 0:

            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(

                epoch + 1, batch, train_loss.result(), train_accuracy.result()))



    if (epoch + 1) % 1 == 0:

        ckpt_save_path = ckpt_manager.save()

        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,

                                                            ckpt_save_path))



    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,

                                                        train_loss.result(),

                                                        train_accuracy.result()))



    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))