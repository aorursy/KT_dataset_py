import numpy as np

import pandas as pd

import matplotlib

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import seaborn as sns; sns.set(style='white')

from mpl_toolkits.mplot3d import Axes3D



from wordcloud import WordCloud

from sklearn.decomposition import PCA, TruncatedSVD

import math

import pickle



from scipy.cluster import hierarchy

from scipy.spatial.distance import pdist

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans



import tensorflow as tf

import tensorflow.keras.backend as K

from transformers import *

import tokenizers

from sklearn.model_selection import StratifiedKFold



pd.set_option('max_colwidth', 40)
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
Dropout_new = 0.15     # originally 0.1

n_split = 5            # originally 5

lr = 3e-5              # originally 3e-5
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

    

    x1 = tf.keras.layers.Dropout(Dropout_new)(x[0])

    x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)

    x1 = tf.keras.layers.LeakyReLU()(x1)

    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)

    x1 = tf.keras.layers.Dense(1)(x1)

    x1 = tf.keras.layers.Flatten()(x1)

    x1 = tf.keras.layers.Activation('softmax')(x1)

    

    x2 = tf.keras.layers.Dropout(Dropout_new)(x[0]) 

    x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)

    x2 = tf.keras.layers.LeakyReLU()(x2)

    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)

    x2 = tf.keras.layers.Dense(1)(x2)

    x2 = tf.keras.layers.Flatten()(x2)

    x2 = tf.keras.layers.Activation('softmax')(x2)



    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr) 

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
ct = train.shape[0]

input_ids = np.ones((ct,MAX_LEN),dtype='int32')

attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')

token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')

start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')



for k in range(train.shape[0]):

    

    # FIND OVERLAP

    text1 = " "+" ".join(train.loc[k,'text'].split())

    text2 = " ".join(train.loc[k,'selected_text'].split())

    idx = text1.find(text2)

    chars = np.zeros((len(text1)))

    chars[idx:idx+len(text2)]=1

    if text1[idx-1]==' ': chars[idx-1] = 1 

    enc = tokenizer.encode(text1) 

        

    # ID_OFFSETS

    offsets = []; idx=0

    for t in enc.ids:

        w = tokenizer.decode([t])

        offsets.append((idx,idx+len(w)))

        idx += len(w)

    

    # START END TOKENS

    toks = []

    for i,(a,b) in enumerate(offsets):

        sm = np.sum(chars[a:b])

        if sm>0: toks.append(i) 

        

    s_tok = sentiment_id[train.loc[k,'sentiment']]

    input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]

    attention_mask[k,:len(enc.ids)+3] = 1

    if len(toks)>0:

        start_tokens[k,toks[0]+2] = 1

        end_tokens[k,toks[-1]+2] = 1
%%time

jac = []; VER='v0'; DISPLAY=1 # USE display=1 FOR INTERACTIVE

oof_start = np.zeros((input_ids.shape[0],MAX_LEN))

oof_end = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start_train = np.zeros((input_ids.shape[0],MAX_LEN))

preds_end_train = np.zeros((input_ids.shape[0],MAX_LEN))

preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))

preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))



skf = StratifiedKFold(n_splits=n_split,shuffle=True,random_state=SEED)

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

    #weight_fn = '%s-roberta-%i.h5'%(VER,fold)

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

        #save_weights(model, weight_fn)



    #print('Loading model...')

    # model.load_weights('%s-roberta-%i.h5'%(VER,fold))

    #load_weights(model, weight_fn)



    print('Predicting OOF...')

    oof_start[idxV,],oof_end[idxV,] = padded_model.predict([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],verbose=DISPLAY)

    

    print('Predicting all Train for Outlier analysis...')

    preds_train = padded_model.predict([input_ids,attention_mask,token_type_ids],verbose=DISPLAY)

    preds_start_train += preds_train[0]/skf.n_splits

    preds_end_train += preds_train[1]/skf.n_splits



    print('Predicting Test...')

    preds = padded_model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)

    preds_start += preds[0]/skf.n_splits

    preds_end += preds[1]/skf.n_splits

    

    # DISPLAY FOLD JACCARD

    all = []

    for k in idxV:

        a = np.argmax(oof_start[k,])

        b = np.argmax(oof_end[k,])

        if a>b: 

            st = train.loc[k,'text'] # IMPROVE CV/LB with better choice here

        else:

            text1 = " "+" ".join(train.loc[k,'text'].split())

            enc = tokenizer.encode(text1)

            st = tokenizer.decode(enc.ids[a-2:b-1])

        all.append(jaccard(st,train.loc[k,'selected_text']))

    jac.append(np.mean(all))

    print('>>>> FOLD %i Jaccard ='%(fold+1),np.mean(all))

    print()
print('>>>> OVERALL 5Fold CV Jaccard =',np.mean(jac))

print(jac) # Jaccard CVs
all = []

for k in range(input_ids_t.shape[0]):

    a = np.argmax(preds_start[k,])

    b = np.argmax(preds_end[k,])

    if a>b: 

        st = test.loc[k,'text']

    else:

        text1 = " "+" ".join(test.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-2:b-1])

    all.append(st)
test['selected_text'] = all

test[['textID','selected_text']].to_csv('submission.csv',index=False)

test.sample(10)
# Visualization training prediction results

all = []

start = []

end = []

start_pred = []

end_pred = []

for k in range(input_ids.shape[0]):

    a = np.argmax(preds_start_train[k,])

    b = np.argmax(preds_end_train[k,])

    start.append(np.argmax(start_tokens[k]))

    end.append(np.argmax(end_tokens[k]))        

    if a>b:

        st = train.loc[k,'text']

        start_pred.append(0)

        end_pred.append(len(st))

    else:

        text1 = " "+" ".join(train.loc[k,'text'].split())

        enc = tokenizer.encode(text1)

        st = tokenizer.decode(enc.ids[a-2:b-1])

        start_pred.append(a)

        end_pred.append(b)

    all.append(st)

train['start'] = start

train['end'] = end

train['start_pred'] = start_pred

train['end_pred'] = end_pred

train['selected_text_pred'] = all

train.sample(10)
def metric_tse(df,col1,col2):

    # Calc metric of tse-competition - according to https://www.kaggle.com/c/tweet-sentiment-extraction/overview/evaluation

    return df.apply(lambda x: jaccard(x[col1],x[col2]),axis=1)
# Analytics

train = train.replace({'sentiment': {'negative': -1, 'neutral': 0, 'positive': 1}})

train['len_text'] = train['text'].str.len()

train['len_selected_text'] = train['selected_text'].str.len()

train['diff_num'] = train['end']-train['start']

train['share'] = train['len_selected_text']/train['len_text']
# Prediction analytics

train['selected_text_pred'] = train['selected_text_pred'].map(lambda x: x.lstrip(' '))

train['len_selected_text_pred'] = train['selected_text_pred'].str.len()

train['diff_num_pred'] = train['end_pred']-train['start_pred']

train['share_pred'] = train['len_selected_text_pred']/train['len_text']

# len_equal

train['len_equal'] = 0

train.loc[(train['start'] == train['start_pred']) & (train['end'] == train['end_pred']), 'len_equal'] = 1

# metric

train['metric'] = metric_tse(train,'selected_text','selected_text_pred')

# res

train['res'] = 0

train.loc[train['metric'] == 1, 'res'] = 1
def rep_3chr(text):

    # Checks if there are 3 or more repetitions of characters in words

    chr3 = 0

    for word in text.split():

        for c in set(word):

            if word.rfind(c+c+c) > -1:

                chr3 = 1                

    return chr3
# Analysis of 3 or more repetitions of characters in words

train['text_chr3'] = train['text'].apply(rep_3chr)

train['selected_text_chr3'] = train['selected_text'].apply(rep_3chr)

train['selected_text_pred_chr3'] = train['selected_text_pred'].apply(rep_3chr)
# result

col_interesting = ['sentiment', 'len_text', 'text_chr3', 'selected_text', 'len_selected_text', 'diff_num', 'share', 

                   'selected_text_chr3', 'selected_text_pred', 'len_selected_text_pred', 'diff_num_pred', 'share_pred',

                   'selected_text_pred_chr3', 'len_equal', 'metric', 'res']

train[col_interesting].head(10)
print('Total metric =',train['metric'].mean())
train.describe()
# Outlier

train_outlier = train[train['res'] == 0].reset_index(drop=True)

train_outlier
train_outlier.describe()
sh_out = str(round(len(train_outlier)*100/len(train),1))

print('Number of outliers is ' + sh_out + '% from training data')
# Good prediction

train_good = train[train['res'] == 1].reset_index(drop=True)

train_good
train_good.describe()
print('Share of all data')

train[['share', 'share_pred']].hist(bins=10)
print('Share of outlier data')

train_outlier[['share', 'share_pred']].hist(bins=10)
# Only one word in 'selected_text'

train_outlier[train_outlier['diff_num']==0]
train_good[train_good['diff_num']==0]
# 'selected_text' = 'text'

train_outlier[train_outlier['share']==1]
train_good[train_good['share']==1]
# Only one word in 'text'

train_outlier[train_outlier["text"].str.find(' ') == -1].head(5)
len(train_outlier[train_outlier["text"].str.find(' ') == -1])
train_good[train_good["text"].str.find(' ') == -1].head(5)
len(train_good[train_good["text"].str.find(' ') == -1])
def plot_word_cloud(x, col):

    corpus=[]

    for k in x[col].str.split():

        for i in k:

            corpus.append(i)

    plt.figure(figsize=(12,8))

    word_cloud = WordCloud(

                              background_color='black',

                              max_font_size = 80

                             ).generate(" ".join(corpus[:50]))

    plt.imshow(word_cloud)

    plt.axis('off')

    plt.show()

    return corpus[:50]
# All training text

print('Word Cloud for all text in training data')

train_all = plot_word_cloud(train, 'text')
train_all
# All test

print('Word Cloud for all text in test')

test_all = plot_word_cloud(test, 'text')
test_all
# All training selected_text

print('Word Cloud for selected_text in training data')

train_selected_text = plot_word_cloud(train, 'selected_text')
train_selected_text
# Oitlier WordCloud

print('Word Cloud for Outliers')

outlier_max = plot_word_cloud(train_outlier, 'selected_text')
outlier_max
# Worst oitlier WordCloud

print('Word Cloud for the 100 worst outliers')

outlier_max100 = plot_word_cloud(train_outlier.nsmallest(100, 'metric', keep='all'), 'selected_text')
outlier_max100
# Worst oitlier WordCloud

print('Word Cloud for the 1000 worst outliers')

outlier_max1000 = plot_word_cloud(train_outlier.nsmallest(1000, 'metric', keep='all'), 'selected_text')
outlier_max1000
# Good prediction WordCloud

print('Word Cloud for good prediction')

good_max = plot_word_cloud(train_good, 'selected_text')
good_max
def subtext_analysis(col, subtext, df1, str1, df2, str2):

    # Calc statistics as table for subtext in the df1[col] (smaller) in compare to df2[col] (bigger) 

    

    result = pd.DataFrame(columns = ['subtext', str1, str2, 'share,%'])

    if (len(df1) > 0) and (len(df2) > 0):

        for i in range(len(subtext)):

            result.loc[i,'subtext'] = subtext[i]

            num1 = len(df1[df1[col].str.find(subtext[i]) > -1])

            result.loc[i, str1] = num1

            num2 = len(df2[df2[col].str.find(subtext[i]) > -1])

            result.loc[i, str2] = num2

            result.loc[i,'share,%'] = round(num1*100/num2,1) if num2 != 0 else 0

    print('Number of all data is', len(df2))

    display(result.sort_values(by=['share,%', str1], ascending=False))
def subtext_analysis_one_df(col, subtext, df, str):

    # Calc statistics as table for subtext in the df[col]

    

    result = pd.DataFrame(columns = ['subtext', str, 'share of all,%'])

    num_all = len(df)

    if (num_all > 0):

        for i in range(len(subtext)):

            result.loc[i,'subtext'] = subtext[i]

            num = len(df[df[col].str.find(subtext[i]) > -1])

            result.loc[i, str] = num

            result.loc[i,'share of all,%'] = round(num*100/num_all,1)

    print('Number of all data is', len(df))

    display(result.sort_values(by='share of all,%', ascending=False))    
subtext_test = ['SAD', 'bullying', 'Uh', 'oh', 'onna', 'fun', 'addicted', 'Power', 'well', 'unhappy', 'funny', 'Tears', 'Fears', 'sleeeeepy', ' ', ',', '?', '!' ,'!!', '!!!', ':/', '...', 'http', '****']
subtext_analysis("selected_text", subtext_test, train_outlier, 'train_outliers', train, 'train_all')
subtext_analysis("selected_text", subtext_test, train_good, 'train_good', train, 'train_all')
subtext_analysis_one_df("selected_text", subtext_test, train, 'test_all')
subtext_analysis_one_df("selected_text", subtext_test, test, 'test_all')
test['text_chr3'] = test['text'].apply(rep_3chr)

test.head(10)
test.describe()
print('Metric of prediction for training data')

train[['metric']].hist(bins=10)
print('Metric of prediction for outliers of training data')

train_outlier[['metric']].hist(bins=10)
train_outlier1 = train_outlier.nsmallest(1000, 'metric', keep='all')

train_outlier2 = train_outlier.nsmallest(2000, 'metric', keep='all')

train_outlier3 = train_outlier.nsmallest(3000, 'metric', keep='all')

train_outlier5 = train_outlier.nsmallest(5000, 'metric', keep='all')

train_outlier8 = train_outlier.nsmallest(8000, 'metric', keep='all')
subtext_analysis("selected_text", subtext_test, train_outlier1, 'in worst 1000 outliers', train_outlier, 'in all outliers')
subtext_analysis("selected_text", subtext_test, train_outlier2, 'in worst 2000 outliers', train_outlier, 'in all outliers')
subtext_analysis("selected_text", subtext_test, train_outlier3, 'in worst 3000 outliers', train_outlier, 'in all outliers')
subtext_analysis("selected_text", subtext_test, train_outlier5, 'in worst 5000 outliers', train_outlier, 'in all outliers')
subtext_analysis("selected_text", subtext_test, train_outlier8, 'in worst 8000 outliers', train_outlier, 'in all outliers')
train_outlier1.describe()
train_outlier2.describe()
train_outlier3.describe()
train_outlier5.describe()
train_outlier8.describe()
# Histograms of interesting features in training data

col_hist = ['sentiment', 'start', 'end', 'start_pred', 'end_pred', 'len_text', 'len_selected_text', 

            'text_chr3', 'selected_text_chr3', 'selected_text_pred_chr3', 'metric']
print('Statistics for 1000 worst outliers')

fig = plt.figure(figsize = (8,8))

ax = fig.gca()

train_outlier1[col_hist].hist(ax=ax)

plt.show()
print('Statistics for 2000 worst outliers')

fig = plt.figure(figsize = (8,8))

ax = fig.gca()

train_outlier2[col_hist].hist(ax=ax)

plt.show()
print('Statistics for 3000 worst outliers')

fig = plt.figure(figsize = (8,8))

ax = fig.gca()

train_outlier3[col_hist].hist(ax=ax)

plt.show()
print('Statistics for 5000 worst outliers')

fig = plt.figure(figsize = (8,8))

ax = fig.gca()

train_outlier5[col_hist].hist(ax=ax)

plt.show()
print('Statistics for 8000 worst outliers')

fig = plt.figure(figsize = (8,8))

ax = fig.gca()

train_outlier8[col_hist].hist(ax=ax)

plt.show()
def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", plot=True, title=None):

        lsa = TruncatedSVD(n_components=2)

        lsa.fit(test_data)

        lsa_scores = lsa.transform(test_data)

        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}

        color_column = [color_mapper[label] for label in test_labels]

        colors = ['orange','blue']

        if plot:

            plt.title(title)

            plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))

            orange_patch = mpatches.Patch(color='orange', label='Good')

            blue_patch = mpatches.Patch(color='blue', label='Outlier')

            plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})
fig = plt.figure(figsize=(16, 16))          

plot_LSA(preds_start_train, train['res'], title='Predicted start places of selected text in training data')

plt.show()
fig = plt.figure(figsize=(16, 16))          

plot_LSA(preds_end_train, train['res'], title='Predicted end places of selected text in training data')

plt.show()
data = train[['sentiment', 'start', 'end', 'start_pred', 'end_pred', 'len_text', 'len_selected_text', 'diff_num', 'share', 'metric', 'res']].dropna()

data
# Thanks to https://www.kaggle.com/kashnitsky/topic-7-unsupervised-learning-pca-and-clustering

inertia = []

pca = PCA(n_components=2)

# fit X and apply the reduction to X 

x_3d = pca.fit_transform(data)

for k in range(1, 8):

    kmeans = KMeans(n_clusters=k, random_state=1).fit(x_3d)

    inertia.append(np.sqrt(kmeans.inertia_))

plt.plot(range(1, 8), inertia, marker='s');

plt.xlabel('$k$')

plt.ylabel('$J(C_k)$');
# Thanks to https://www.kaggle.com/arthurtok/a-cluster-of-colors-principal-component-analysis

# Set a 3 KMeans clustering

kmeans = KMeans(n_clusters=5, random_state=0)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_3d)

LABEL_COLOR_MAP = {0 : 'r',

                   1 : 'g',

                   2 : 'b',

                   3 : 'y',

                   4 : 'c'}



label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

plt.figure(figsize = (7,7))

plt.scatter(x_3d[:,0],x_3d[:,1], c= label_color, alpha=0.9)

plt.show()