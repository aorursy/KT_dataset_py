import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))
dictionary = open('../input/cmu-pronouncing-dictionary/cmudict.dict', 'r')
%%time

with dictionary as f:
    phonics = [line.rstrip('\n') for line in f]

word = []
pronunciation = []
pronunciation_sorted = []

for x in phonics:
    x = x.split(' ')
    word.append(x[0])
    p = ' '.join(x[1:])
    # removing numbers from pronunciation
    p = p.replace('0','')
    p = p.replace('1','')
    p = p.replace('2','')
    pronunciation.append(p)
    a = p.split(' ')
    a.sort()
#     a = ' '.join(a)
    pronunciation_sorted.append(a)

df = pd.DataFrame({
        "word": word,
        "pronunciation": pronunciation,
        "pronunciation_sorted": pronunciation_sorted
    })

print(df.shape)
unigram_freq = pd.read_csv('../input/english-word-frequency/unigram_freq.csv')

# unigram_freq = unigram_freq.loc[unigram_freq['word'].isin(word)]
# df = df.loc[df['word'].isin(fword)]
df = pd.merge(df,unigram_freq, on='word')
print(df.shape)
Phrase = "a dog ate a taco"
Phrase = Phrase.lower().split()
Data_List = df[['word','pronunciation_sorted']].values.tolist()

ARPA_Phonics = []

for x in Phrase:
    Word_Array = df.loc[df['word'] == x]['pronunciation_sorted'].values[0]
    ARPA_Phonics += Word_Array

print(ARPA_Phonics)
%%time
def ARPA_Phrases(Data, phrase, word_total, First):
    count = []
    for i ,line in enumerate(Data):
#         if First and i % 1 == 0:
#             print(i)
        word = line[0]
        pron = line[1]
        curword_total = word_total.copy()
        curphrase = phrase.copy()
        if all(x in curphrase for x in pron):
            try:
                for x in pron:
                    del curphrase[curphrase.index(x)]
                curword_total.append(word)
                if curphrase == []:
                    count.append(curword_total)
                else:
                    nlist = []               
                    for nline in Data[i:]:
                        if all(x in curphrase for x in nline[1]):
                            nlist.append(nline)
                    count += ARPA_Phrases(nlist, curphrase, curword_total, False)
            except:
                word
    return count

New_List = []               
for line in Data_List:
    if all(x in ARPA_Phonics for x in line[1]):
        New_List.append(line)

Data_List = New_List
count = ARPA_Phrases(Data_List, ARPA_Phonics, [], True)

print(len(count))
count[:100]