import numpy as np

import matplotlib.pyplot as plt

lbs = [0.706] +  [0.707]*6 + [0.708]*5 + [0.709]*7 + [0.710]*8 + [0.711]*4 + [0.712]*8 + [0.713]*2 + [0.714]*2 + [0.715]*2

pp = plt.hist(lbs, bins=10)

plt.title("LB Scores")

plt.show()
jac = [0.708]*5 + [0.707]*9 + [0.706]*6 + [0.705]*12 + [0.704]*8 + [0.703]*2 + [0.702] + [0.701]*2

pp2 = plt.hist(jac, bins=8)

plt.title("Estimated (Rounded) Mean Jaccard Scores")

plt.show()
print('Number of submissions above: ' + str(len(lbs)))
import pandas as pd, numpy as np

import tensorflow as tf

import tensorflow.keras.backend as K

from sklearn.model_selection import StratifiedKFold

from transformers import *

import tokenizers

import math

import re

print('TF version',tf.__version__)
MAX_LEN = 96

PATH = '../input/tf-roberta/'

tokenizer = tokenizers.ByteLevelBPETokenizer(

    vocab_file=PATH+'vocab-roberta-base.json', 

    merges_file=PATH+'merges-roberta-base.txt', 

    lowercase=True,

    add_prefix_space=True

)

EPOCHS = 3 # originally 3

BATCH_SIZE = 32 # originally 32

PAD_ID = 1

SEED = 88888

LABEL_SMOOTHING = 0.1

tf.random.set_seed(SEED)

np.random.seed(SEED)

sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv').fillna('')

train.head()
text = train['text'].values

selected_text = train['selected_text'].values.copy()

sentiments = train['sentiment'].values



ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(train.shape[0]):

    ss = text[k].find(selected_text[k])

    if text[k][max(ss - 2, 0):ss] == '  ':

        ss -= 2

    if ss > 0  and text[k][ss - 1] == ' ':

        ss -= 1



    ee = ss + len(selected_text[k])



    if re.match(r' [^ ]', text[k]) is not None:

        ee -= 1



    ss = max(0, ss)

    if '  ' in text[k][:ss] and sentiments[k] != 'neutral':

        text1 = " ".join(text[k].split())

        sel = text1[ss:ee].strip()

        if len(sel) > 1 and sel[-2] == ' ':

            sel = sel[:-2]



        selected_text[k] = sel



    text1 = " "+" ".join(text[k].split())

    text2 = " ".join(selected_text[k].split()).lstrip(".,;:")



    idx = text1.find(text2)

    if idx != -1:

        chars = np.zeros((len(text1)))

        chars[idx:idx+len(text2)]=1

        if text1[idx-1]==' ': chars[idx-1] = 1 

    else:

        import pdb;pdb.set_trace()

        chars = np.ones((len(text1)))

    enc = tokenizer.encode(text1) 



    # ID_OFFSETS

    offsets = enc.offsets



    # START END TOKENS

    _toks = []



    for i,(a,b) in enumerate(offsets):

        sm = np.mean(chars[a:b])

        #if (sm > 0.6 and chars[a] != 0):  # こうすると若干伸びるけど...

        if (sm > 0.5 and chars[a] != 0): 

            _toks.append(i)



    toks = _toks

    s_tok = sentiment_id[sentiments[k]]

    input_ids[k, :len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask[k,:len(enc.ids)+3] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+2] = 1

        end_tokens[k,toks[-1]+2] = 1  

test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv').fillna('')



ct = test.shape[0]

input_ids_t = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask_t = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids_t = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(test.shape[0]):

        

    # INPUT_IDS

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)                

    s_tok = sentiment_id[test.loc[k,'sentiment']]

    input_ids_t[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask_t[k,:len(enc.ids)+3] = 1
import pickle



def save_weights(model, dst_fn):

    weights = model.get_weights()

    with open(dst_fn, 'wb') as f:

        pickle.dump(weights, f)





def load_weights(model, weight_fn):

    with open(weight_fn, 'rb') as f:

        weights = pickle.load(f)

    model.set_weights(weights)

    return model



def loss_fn(y_true, y_pred):

    # adjust the targets for sequence bucketing

    ll = tf.shape(y_pred)[1]

    y_true = y_true[:, :ll]

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,

        from_logits=False, label_smoothing=LABEL_SMOOTHING)

    loss = tf.reduce_mean(loss)

    return loss





def build_model():

    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)



    lens = MAX_LEN - tf.reduce_sum(padding, -1)

    max_len = tf.reduce_max(lens)

    ids_ = ids[:, :max_len]

    att_ = att[:, :max_len]

    tok_ = tok[:, :max_len]



    config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')

    bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)

    x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)

    

    x1 = tf.keras.layers.Dropout(0.1)(x[0])

    x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 

    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) 

    model.compile(loss=loss_fn, optimizer=optimizer)

    

    # this is required as `model.predict` needs a fixed size!

    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

    

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1_padded,x2_padded])

    return model, padded_model
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    if (len(a)==0) & (len(b)==0): return 0.5

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
import re

def modify_punc_length(text, selected_text):

    m = re.search(r'[!\.\?]+$', selected_text)        

    if m is None:

        return selected_text

    

    conti_punc = len(m.group())



    if conti_punc >= 4:

        selected_text = selected_text[:-(conti_punc-2)]

    elif conti_punc == 1:# 元のtextを探しに行く

        tmp = re.sub(r"([\\\*\+\.\?\{\}\(\)\[\]\^\$\|])", r"\\\g<0>", selected_text)

        pat = re.sub(r" ", " +", tmp)

        m = re.search(pat, text)

        f_idx0 = m.start()

        f_idx1 = m.end()



        if f_idx1 != len(text) and text[f_idx1] in ("!", ".", "?"):

            f_idx1 += 1

            selected_text = text[f_idx0:f_idx1]

    return selected_text



def postprocess(row):

    if row.original_text == '':

        return row.normalized_text.strip()

    original_text = row.original_text.replace('\t', '')

    y_start_char = row.y_start_char

    y_end_char = row.y_end_char

    y_selected_text = row.normalized_text[y_start_char:y_end_char].strip()

    if (y_end_char < len(row.normalized_text) and row.sentiment != 'neutral' and

        y_selected_text[-1] == '.' and

        (row.normalized_text[y_end_char] == '.' or 

         y_selected_text[-2] == '.')):

        y_selected_text = re.sub('\.+$', '..', y_selected_text)



    tmp = re.sub(r"([\\\*\+\.\?\{\}\(\)\[\]\^\$\|])", r"\\\g<0>", y_selected_text)

    pat = re.sub(r" ", " +", tmp)

    m = re.search(pat, original_text)

    if m is None:

        print(row.normalized_text[y_start_char:y_end_char].strip())

        print(row.normalized_text)

        print(y_selected_text)

        return y_selected_text

    ss2 = m.start()

    ee2 = m.end()

        

    if '  ' in original_text[:(ss2+ee2)//2]:

        ss = y_start_char

        ee = y_end_char #+ 1

        if ss > 1 and original_text[ss-1:ss+1] == '..' and  original_text[ss+1] != '.' and row.sentiment != 'neutral':

            ss -= 1

        if row.sentiment == 'neutral':

            ee -= 1

            front_spaces = re.findall("^ +[^ ]", original_text[:ee2])

            if len(front_spaces) == 1:

                ee += front_spaces[0].count(' ')

            elif len(front_spaces) > 1:

                raise Excpetion("invalid process")

            for cnt_base in re.findall("[^ ]  +[^ ]", original_text[:ee2].strip()):

                ee += cnt_base[2:].count(' ')            

        else:

            ee += 1

        st = original_text[ss:ee].lstrip(' ½¿')

        y_selected_text = re.sub(r' .$', '', st)#.strip('`') ###  この一行追加

    else:

        if (ee2 < len(original_text)-1 and original_text[ee2:ee2+2] in ('..', '!!', '??', '((', '))')):

            ee2 += 1

        # 先頭の空白分後退

        if original_text[0] == ' ':

            ss2 -= 1



        y_selected_text = original_text[ss2:ee2].lstrip(' ½')



        if  row.normalized_text[:y_end_char + 5] == " " + row.original_text[:ee2 + 4] and row.sentiment != 'neutral': # 簡単のため、長さが同じ場合に限定している

            y_selected_text = modify_punc_length(original_text, y_selected_text)

            

            

    return y_selected_text
text = train['text'].values

selected_text = train['selected_text'].values.copy()

sentiments = train['sentiment'].values

ids = train['textID'].values



jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED) #originally 5 splits

for fold,(idxT,idxV) in enumerate(skf.split(input_ids,train.sentiment.values)):



    print('#'*25)

    print('### FOLD %i'%(fold+1))

    print('#'*25)

    

    K.clear_session()

    model, padded_model = build_model()

        

    #sv = tf.keras.callbacks.ModelCheckpoint(

    #    '%s-roberta-%i.h5'%(VER,fold), monitor='val_loss', verbose=1, save_best_only=True,

    #    save_weights_only=True, mode='auto', save_freq='epoch')

    inpT = [input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]]

    targetT = [start_tokens[idxT,], end_tokens[idxT,]]

    inpV = [input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]]

    targetV = [start_tokens[idxV,], end_tokens[idxV,]]

    # sort the validation data

    shuffleV = np.int32(sorted(range(len(inpV[0])), key=lambda k: (inpV[0][k] == PAD_ID).sum(), reverse=True))

    inpV = [arr[shuffleV] for arr in inpV]

    targetV = [arr[shuffleV] for arr in targetV]

    weight_fn = '%s-roberta-%i.h5'%(VER,fold)

    for epoch in range(1, EPOCHS + 1):

        # sort and shuffle: We add random numbers to not have the same order in each epoch

        shuffleT = np.int32(sorted(range(len(inpT[0])), key=lambda k: (inpT[0][k] == PAD_ID).sum() + np.random.randint(-3, 3), reverse=True))

        # shuffle in batches, otherwise short batches will always come in the beginning of each epoch

        num_batches = math.ceil(len(shuffleT) / BATCH_SIZE)

        batch_inds = np.random.permutation(num_batches)

        shuffleT_ = []

        for batch_ind in batch_inds:

            shuffleT_.append(shuffleT[batch_ind * BATCH_SIZE: (batch_ind + 1) * BATCH_SIZE])

        shuffleT = np.concatenate(shuffleT_)

        # reorder the input data

        inpT = [arr[shuffleT] for arr in inpT]

        targetT = [arr[shuffleT] for arr in targetT]

        model.fit(inpT, targetT, 

            epochs=epoch, initial_epoch=epoch - 1, batch_size=BATCH_SIZE, verbose=DISPLAY, callbacks=[],

            validation_data=(inpV, targetV), shuffle=False)  # don't shuffle in `fit`

        save_weights(model, weight_fn)



    print('Loading model...')

    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))

    load_weights(model, weight_fn)



    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], verbose=DISPLAY, batch_size=128)

    

    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        text0 = text[k]

        text1 = " " + " ".join(text[k].split())

        enc = tokenizer.encode(text1)



        aa = np.argmax(oof_start[k])

        if aa - 2 >= len(enc.offsets):

            aa = 2



        oof_end[k, :aa] = 0

        bb = np.argmax(oof_end[k])



        try:

            ss = enc.offsets[aa - 2][0]

            ee = enc.offsets[bb - 2][1] 

            st = text1[ss:ee].strip()

        except IndexError:

            ss = 0

            ee = len(text1)

            st = text1





        row = pd.Series(dict(

            original_text=text0,

            normalized_text=text1,

            sentiment=sentiments[k],

            y_start_char=ss,

            y_end_char=ee,

        ))

        try:

            st = postprocess(row)

        except:

            print(k)

            st = tokenizer.decode(enc.ids[aa-2:bb-1])

        sc = jaccard(st,selected_text[k])

        all.append(sc)

    

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))
print(jac) # Jaccard CVs
all = []

for k in range(input_ids_t.shape[0]):

    text1 = " "+" ".join(test.loc[k,'text'].split())

    enc = tokenizer.encode(text1)

    

    aa = np.argmax(preds_start[k])

    if aa - 2 >= len(enc.offsets):

        aa = 2



    preds_end[k, :aa] = -1

    bb = np.argmax(preds_end[k])



    try:

        ss = enc.offsets[aa - 2][0]

        ee = enc.offsets[bb - 2][1] 

        st = text1[ss:ee].strip()

    except IndexError:

        ss = 0

        ee = len(text1)

        st = text1





    row = pd.Series(dict(

        original_text=test.loc[k,'text'],

        normalized_text=text1,

        sentiment=test.loc[k,'sentiment'],

        y_start_char=ss,

        y_end_char=ee,

    ))

    try:

        st = postprocess(row)

    except:

        print(k)

        st = tokenizer.decode(enc.ids[aa-2:bb-1])

    all.append(st)
test['selected_text'] = all

test[['textID','selected_text']].to_csv('submission.csv',index=False)

pd.set_option('max_colwidth', 60)

test.sample(25)