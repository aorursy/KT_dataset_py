import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
import re
import string
import time


from pyphen import Pyphen
from gensim.models import KeyedVectors as wv
from sklearn.feature_extraction.text import TfidfVectorizer
 
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from keras.preprocessing.text import text_to_word_sequence, Tokenizer

import Levenshtein as lv

pd.options.display.max_rows = 8
pd.options.display.max_columns = 999
print(os.listdir("../input"))

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'【{name}】 done in 【{time.time() - t0:.0f}】 s')
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
print(train_df.shape, test_df.shape, submission.shape)
s = test_df.qid
print(s.shape, s.nunique())
s1 = submission.qid
print(s1.shape, s1.nunique())
print(np.sum(s!=s1))
del s,s1,submission
gc.collect()
y = train_df.target
print(y.shape, y.loc[1==y].shape, y.loc[1==y].shape[0]/y.shape[0])
train_df.loc[0==y]
train_df.loc[1==y]
s = train_df.loc[1==y, 'question_text'].str.len()
print(s.describe())
print()
s = train_df.loc[0==y, 'question_text'].str.len()
print(s.describe())
s = train_df.question_text.str.len()
print(s.describe())
print()
s = test_df.question_text.str.len()
print(s.describe())
s = train_df.loc[1==y, 'question_text'].str.count(' ')
print(s.describe())
print()
s = train_df.loc[0==y, 'question_text'].str.count(' ')
print(s.describe())
s = train_df.question_text.str.count(' ')
print(s.describe())
print()
s = test_df.question_text.str.count(' ')
print(s.describe())
s = train_df.loc[train_df.target==1,'question_text'].str.count(' ').sort_values()
print(s.quantile([0.9,0.95,0.99,0.995,0.999,0.9995,0.9999,0.9999,0.99995,0.99999,0.999995,0.999999]).to_dict())
print(s.iloc[-30:].to_dict())
s = train_df.loc[train_df.target==0,'question_text'].str.count(' ').sort_values()
print(s.quantile([0.9,0.95,0.99,0.995,0.999,0.9995,0.9999,0.9999,0.99995,0.99999,0.999995,0.999999]).to_dict())
print(s.iloc[-30:].to_dict())
s = train_df.question_text.str.count(' ').sort_values()
print(s.quantile([0.9,0.95,0.99,0.995,0.999,0.9995,0.9999,0.9999,0.99995,0.99999,0.999995,0.999999]).to_dict())
print(s.iloc[-30:].to_dict())
s = test_df.question_text.str.count(' ').sort_values()
print(s.quantile([0.9,0.95,0.99,0.995,0.999,0.9995,0.9999,0.9999,0.99995,0.99999,0.999995,0.999999]).to_dict())
print(s.iloc[-30:].to_dict())
for min_df in [1,2,3,4,5,10,20,30]:
    tvr = TfidfVectorizer(token_pattern=r'(?u)\w+|[^\w\s]', strip_accents='unicode', min_df=min_df)
    gc.collect()
    tvr.fit(train_df.question_text)
    tr_words = tvr.get_feature_names()
    tvr.fit(test_df.question_text)
    ts_words = tvr.get_feature_names()
    rs_words = np.setdiff1d(ts_words, tr_words)
    tt_words1 = np.union1d(tr_words, ts_words)
    tvr.fit(train_df.question_text.append(test_df.question_text))
    tt_words2 = tvr.get_feature_names()
    tvr.fit(train_df.loc[train_df.target>0, 'question_text'])
    pos_words = tvr.get_feature_names()
    tvr.fit(train_df.loc[0==train_df.target, 'question_text'])
    neg_words = tvr.get_feature_names()
    print(f'min_df={min_df}: tr_words={len(tr_words)}, ts_words={len(ts_words)}, rs_words={rs_words.shape}, '
          +f'tt_words1={tt_words1.shape}, tt_words2={len(tt_words2)}, pos_words={len(pos_words)}, neg_words={len(neg_words)}')
    del tvr,tr_words,ts_words,rs_words,tt_words1,tt_words2,pos_words,neg_words
    gc.collect()
@lru_cache(maxsize=65536)
def remove_punctuation(text):
    return ''.join(ch for ch in text if ch not in string.punctuation)


@lru_cache(maxsize=65536)
def count_syllable(word):
    dic = Pyphen(lang='en_US')
    word_hyphenated = dic.inserted(word)
    return max(1, word_hyphenated.count("-") + 1)


@lru_cache(maxsize=65536)
def syllable_count(text, lang='en_US'):
    text = text.lower()
    text = remove_punctuation(text)

    if not text:
        return 0

    dic = Pyphen(lang=lang)
    count = 0
    for word in text.split(' '):
        count += count_syllable(word)
    return count


@lru_cache(maxsize=65536)
def lexicon_count(text, removepunct=True):
    if removepunct:
        text = remove_punctuation(text)
    count = len(text.split())
    return count


@lru_cache(maxsize=65536)
def sentence_count(text):
    ignore_count = 0
    sentences = re.split(r' *[.?!][\'")\]]*[ |\n](?=[A-Z])', text)
    for sentence in sentences:
        if lexicon_count(sentence) <= 2:
            ignore_count += 1
    return max(1, len(sentences) - ignore_count)


@lru_cache(maxsize=65536)
def polysyllabcount(text):
    count = 0
    for word in text.split():
        wrds = syllable_count(word)
        if wrds >= 3:
            count += 1
    return count


@lru_cache(maxsize=65536)
def linsear_write_formula(text):
    easy_word = 0
    difficult_word = 0
    text_list = text.split()[:100]

    for word in text_list:
        if syllable_count(word) < 3:
            easy_word += 1
        else:
            difficult_word += 1

    text = ' '.join(text_list)

    number = easy_word * 1 + difficult_word * 3 / sentence_count(text)

    if number <= 20:
        number -= 2

    return number / 2


@lru_cache(maxsize=65536)
def lix(text, avg_sentence_length):
    words = text.split()

    words_len = len(words)
    long_words = len([wrd for wrd in words if len(wrd) > 6])

    per_long_words = long_words * 100 / words_len

    return avg_sentence_length + per_long_words
def encode_text(df):
    def count_chars(txt):
        _len = 0
        digit_cnt, number_cnt = 0, 0
        lower_cnt, upper_cnt, letter_cnt, word_cnt = 0, 0, 0, 0
        char_cnt, term_cnt = 0, 0
        conj_cnt, blank_cnt, punc_cnt = 0, 0, 0
        sign_cnt, marks_cnt = 0, 0

        flag = 10
        for ch in txt:
            _len += 1
            if ch in string.ascii_lowercase:
                lower_cnt += 1
                letter_cnt += 1
                char_cnt += 1
                if flag:
                    word_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 0
            elif ch in string.ascii_uppercase:
                upper_cnt += 1
                letter_cnt += 1
                char_cnt += 1
                if flag:
                    word_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 0
            elif ch in string.digits:
                digit_cnt += 1
                char_cnt += 1
                if 1 != flag:
                    number_cnt += 1
                    if flag > 2:
                        term_cnt += 1
                    flag = 1
            elif '_' == ch:
                conj_cnt += 1
                char_cnt += 1
                if flag > 2:
                    term_cnt += 1
                flag = 2
            elif ch in string.whitespace:
                blank_cnt += 1
                flag = 3
            elif ch in string.punctuation:
                punc_cnt += 1
                flag = 4
            else:
                sign_cnt += 1
                if flag != 5:
                    marks_cnt += 1
                    flag = 5

        syllable_cnt = syllable_count(txt)
        sentence_cnt = sentence_count(txt)
        avg_sentence_length = word_cnt / sentence_cnt
        avg_sentence_per_word = sentence_cnt / max(1, word_cnt)
        avg_syllables_per_word = syllable_cnt / max(1, word_cnt)
        avg_character_per_word = char_cnt / max(1, word_cnt)
        avg_letter_per_word = letter_cnt / max(1, word_cnt)
        flesch_reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        flesch_kincaid_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        polysyllable_cnt = polysyllabcount(txt)
        smog_index = (1.043 * (30 * polysyllable_cnt / sentence_cnt) ** .5) + 3.1291
        coleman_liau_index = 5.8 * avg_letter_per_word - 29.6 * avg_sentence_per_word - 15.8
        readability = 4.71 * avg_character_per_word + 0.5 * avg_sentence_length - 21.43
        linsear_write_metric = linsear_write_formula(txt)
        lix_metric = lix(txt, avg_sentence_length)

        return (_len, digit_cnt, number_cnt, digit_cnt / max(1, number_cnt), lower_cnt, upper_cnt, letter_cnt,
                word_cnt, avg_letter_per_word, char_cnt, term_cnt, char_cnt / max(1, term_cnt), conj_cnt,
                blank_cnt, punc_cnt, sign_cnt, marks_cnt, sign_cnt / max(1, marks_cnt), syllable_cnt,
                sentence_cnt, avg_sentence_length, avg_sentence_per_word, avg_syllables_per_word,
                avg_character_per_word, flesch_reading_ease, flesch_kincaid_grade, polysyllable_cnt, smog_index,
                coleman_liau_index, readability, linsear_write_metric, lix_metric)

    (df['char_len'], df['digit_cnt'], df['number_cnt'], df['digit_cnt/number_cnt'], df['lower_cnt'], df['upper_cnt'],
     df['letter_cnt'], df['word_cnt'], df['avg_letter_per_word'], df['char_cnt'], df['term_cnt'], df['char_cnt/term_cnt'],
     df['conj_cnt'], df['blank_cnt'], df['punc_cnt'], df['sign_cnt'], df['marks_cnt'], df['sign_cnt/marks_cnt'], df['syllable_cnt'], 
     df['sentence_cnt'], df['avg_sentence_length'], df['avg_sentence_per_word'], df['avg_syllables_per_word'], 
     df['avg_character_per_word'],df['flesch_reading_ease'],df['flesch_kincaid_grade'],df['polysyllable_cnt'],df['smog_index'],
     df['coleman_liau_index'],df['readability'],df['linsear_write_metric'],df['lix_metric']) = zip(
        *df.question_text.apply(count_chars))

    return df
with timer('encode train text'):
    train_df = encode_text(train_df)
train_df
test_df = encode_text(test_df)
test_df
train_df.info()
test_df.info()
train_df = train_df.fillna('the')
test_df = test_df.fillna('the')
gc.collect()

with timer('reserve punctuation'):
    def reserve_puncts(text):
        puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%',
                  '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', '<', '→',
                  '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–',
                  '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓',
                  '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯',
                  '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
                  '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
        punct_dic = {punct: f' {punct} ' for punct in puncts}
        punct_dic.update({'\t': ' ', '\n': ' ', '\r': ' ', '\u200b': ''})
        return text.translate(str.maketrans(punct_dic))

    train_df['question_text'] = train_df.question_text.apply(reserve_puncts)
    test_df['question_text'] = test_df.question_text.apply(reserve_puncts)
    gc.collect()
tr_tkr = Tokenizer(filters='')
tr_tkr.fit_on_texts(train_df.question_text)
ts_tkr = Tokenizer(filters='')
ts_tkr.fit_on_texts(test_df.question_text)
tt_tkr = Tokenizer(filters='')
tt_tkr.fit_on_texts(train_df.question_text.append(test_df.question_text))
pos_tkr = Tokenizer(filters='')
pos_tkr.fit_on_texts(train_df.loc[train_df.target==1,'question_text'])
neg_tkr = Tokenizer(filters='')
neg_tkr.fit_on_texts(train_df.loc[train_df.target==0,'question_text'])
gc.collect()

for min_df in [1,2,3,4,5,10,20,30]:
    tr_words = [word for word,cnt in tr_tkr.word_counts.items() if cnt>=min_df]
    ts_words = [word for word,cnt in ts_tkr.word_counts.items() if cnt>=min_df]
    rs_words = np.setdiff1d(ts_words, tr_words)
    tt_words1 = np.union1d(tr_words, ts_words)
    tt_words2 = [word for word,cnt in tt_tkr.word_counts.items() if cnt>=min_df]
    pos_words = [word for word,cnt in pos_tkr.word_counts.items() if cnt>=min_df]
    neg_words = [word for word,cnt in neg_tkr.word_counts.items() if cnt>=min_df]
    print(f'min_df={min_df}: tr_words={len(tr_words)}, ts_words={len(ts_words)}, rs_words={rs_words.shape}, '
          +f'tt_words1={tt_words1.shape}, tt_words2={len(tt_words2)}, pos_words={len(pos_words)}, neg_words={len(neg_words)}')
    del tr_words,ts_words,rs_words,tt_words1,tt_words2,pos_words,neg_words
    gc.collect()

tr_words = list(tr_tkr.word_index.keys())
ts_words = list(ts_tkr.word_index.keys())
rs_words = np.setdiff1d(ts_words, tr_words)
tt_words = list(tt_tkr.word_index.keys())
pos_words = list(pos_tkr.word_index.keys())
neg_words = list(neg_tkr.word_index.keys())
def load_embed_dic(embed_id, embed_root_dir='../input/embeddings'):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    file_path_dic = {
        'glove': os.path.join(embed_root_dir, 'glove.840B.300d', 'glove.840B.300d.txt'),
        'wiki': os.path.join(embed_root_dir, 'wiki-news-300d-1M', 'wiki-news-300d-1M.vec'),
        'para': os.path.join(embed_root_dir, 'paragram_300_sl999', 'paragram_300_sl999.txt'),
        'google': os.path.join(embed_root_dir, 'GoogleNews-vectors-negative300', 'GoogleNews-vectors-negative300.bin')
    }
    if 'wiki' == embed_id:
        embed_dic = dict(get_coefs(*line.split(' ')) for line in open(
            file_path_dic[embed_id], encoding='utf8', errors='ignore') if len(line) > 100)
    elif 'google' == embed_id:
        embed_dic = wv.load_word2vec_format(file_path_dic[embed_id], binary=True)
    else:
        embed_dic = dict(get_coefs(*line.split(' ')) for line in open(
            file_path_dic[embed_id], encoding='utf8', errors='ignore'))

    return embed_dic


def set_diff(s1,s2,batch_num=100):
    batch_size = len(s2) // batch_num + 1
    for i in range(batch_num):
        s = s2[i*batch_size: (i+1)*batch_size]
        s1 = np.setdiff1d(s1, s)
        del s
        gc.collect()
    return s1


@lru_cache(maxsize=65536)
def word_distance(_word1, _word2):
    return lv.distance(_word1, _word2)


def similar(_word1, _word2):
    _len1 = len(_word1)
    _len2 = len(_word2)
    _len = min(_len1, _len2)
    if _len<5 or (_len>=5 and abs(_len1-_len2)>1):
        return False
    return word_distance(_word1, _word2)<=1


_digits = re.compile('\d')
@lru_cache(maxsize=65536)
def contains_digits(d):
    return bool(_digits.search(d))
embed_dic = load_embed_dic('glove')
glove_words = list(embed_dic.keys())
del embed_dic
gc.collect()
print(f'glove_words: {len(glove_words)}')
with timer('train lv'):
    tr_miss_words = set_diff(tr_words, glove_words)
    tr_embed_words = np.setdiff1d(tr_words, tr_miss_words)
    len_dic = {}
    for word in tr_embed_words:
        _len = len(word)
        if _len in len_dic:
            len_dic[_len].append(word)
        else:
            len_dic[_len] = [word]
            
    tr_miss_word_dic = {}
    for word1 in tr_miss_words:
        if not contains_digits(word1):
            _len = len(word1)
            _embed_words = []
            if _len-1 in len_dic:
                _embed_words += len_dic[_len-1]
            if _len in len_dic:
                _embed_words += len_dic[_len]
            if _len+1 in len_dic:
                _embed_words += len_dic[_len+1]

            cand_word = None
            for word2 in _embed_words:
                if similar(word1, word2):
                    if word2>=word1:
                        tr_miss_word_dic[word1] = word2
                        break
                    else:
                        cand_word = word2
            if word1 not in tr_miss_word_dic and cand_word is not None:
                tr_miss_word_dic[word1] = cand_word
    print(f'tr_miss_words: {len(tr_miss_words)}, tr_embed_words: {tr_embed_words.shape}, tr_miss_word_dic: {len(tr_miss_word_dic)}')
print(tr_miss_word_dic) 
with timer('test lv'):
    ts_miss_words = set_diff(ts_words, glove_words)
    print(f'before, ts_miss_words: {ts_miss_words.shape}', end='; ')
    ts_miss_words = np.setdiff1d(ts_miss_words, list(tr_miss_word_dic.keys()))
    print(f'after, ts_miss_words: {ts_miss_words.shape}')
    len_dic = {}
    for word in tr_embed_words:
        _len = len(word)
        if _len in len_dic:
            len_dic[_len].append(word)
        else:
            len_dic[_len] = [word]
            
    ts_miss_word_dic = {}
    for word1 in tqdm(ts_miss_words):
        if not contains_digits(word1):
            _len = len(word1)
            _embed_words = []
            if _len-1 in len_dic:
                _embed_words += len_dic[_len-1]
            if _len in len_dic:
                _embed_words += len_dic[_len]
            if _len+1 in len_dic:
                _embed_words += len_dic[_len+1]

            cand_word = None
            for word2 in _embed_words:
                if similar(word1, word2):
                    if word2>=word1:
                        ts_miss_word_dic[word1] = word2
                        break
                    else:
                        cand_word = word2
            if word1 not in ts_miss_word_dic and cand_word is not None:
                ts_miss_word_dic[word1] = cand_word
    print(f'ts_miss_word_dic: {len(ts_miss_word_dic)}')
print(ts_miss_word_dic)
del tr_miss_words,tr_embed_words,len_dic,tr_miss_word_dic,ts_miss_words,ts_miss_word_dic
gc.collect()
embed_dic = load_embed_dic('glove')
glove_words = list(embed_dic.keys())
wvs = np.stack(embed_dic.values())
print(wvs.shape)
del embed_dic
gc.collect()

wvs = np.sort(np.ravel(wvs))
gc.collect()
print(wvs.shape, np.mean(wvs), np.std(wvs))

y = wvs[::100]
del wvs
gc.collect()
fig = plt.figure(figsize=(18, 9))
sns.distplot(y)
del y
gc.collect()
plt.show()
embed_dic = load_embed_dic('wiki')
wiki_words = list(embed_dic.keys())
wvs = np.stack(embed_dic.values())
print(wvs.shape)
del embed_dic
gc.collect()

wvs = np.sort(np.clip(np.ravel(wvs),-5,5))
gc.collect()
print(wvs.shape, np.mean(wvs), np.std(wvs))

y = wvs[::100]
del wvs
gc.collect()
fig = plt.figure(figsize=(18, 9))
sns.distplot(y)
del y
gc.collect()
plt.show()
embed_dic = load_embed_dic('para')
para_words = list(embed_dic.keys())
wvs = np.stack(embed_dic.values())
print(wvs.shape)
del embed_dic
gc.collect()

wvs = np.sort(np.ravel(wvs))
gc.collect()
print(wvs.shape, np.mean(wvs), np.std(wvs))

y = wvs[::100]
del wvs
gc.collect()
fig = plt.figure(figsize=(18, 9))
sns.distplot(y)
del y
gc.collect()
plt.show()
embed_dic = load_embed_dic('google')
google_words = list(embed_dic.vocab.keys())
wvs = embed_dic.vectors
print(wvs.shape)
del embed_dic
gc.collect()

wvs = np.sort(np.ravel(wvs))
gc.collect()
print(wvs.shape, np.mean(wvs), np.std(wvs))

y = wvs[::100]
del wvs
gc.collect()
fig = plt.figure(figsize=(18, 9))
sns.distplot(y)
del y
gc.collect()
plt.show()
wv_names = ['glove','para','wiki','google']
wv_words = [glove_words,para_words,wiki_words,google_words]
data_names = ['tr','ts','rs','pos','neg']
data_words = [tr_words,ts_words,rs_words,pos_words,neg_words]
miss_words = [[set_diff(data_word, wv_word) for data_word in data_words] for wv_word in wv_words]
print(len(miss_words), len(miss_words[0]))
def print_embed_info(embed_ids):
    for i in range(len(embed_ids)):
        for j in range(len(embed_ids[i])-1):
            print(f'{wv_names[embed_ids[i][j]]} &', end=' ')
        print(f'{wv_names[embed_ids[i][-1]]}:', end=' ')
        
        for j in range(len(data_names)-1):
            cur_miss_words = miss_words[embed_ids[i][0]][j]
            for k in range(1, len(embed_ids[i])):
                cur_miss_words = np.intersect1d(cur_miss_words, miss_words[embed_ids[i][k]][j])
            print(f'{data_names[j]}({len(cur_miss_words)}),', end=' ')
        cur_miss_words = miss_words[embed_ids[i][0]][-1]
        for k in range(1, len(embed_ids[i])):
            cur_miss_words = np.intersect1d(cur_miss_words, miss_words[embed_ids[i][k]][-1])
        print(f'{data_names[-1]}({len(cur_miss_words)})')
print_embed_info([[0],[1],[2],[3]])
print_embed_info([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
print_embed_info([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
print_embed_info([[0,1,2,3]])
