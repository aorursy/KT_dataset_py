import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))
dictionary = open('../input/cmudict.dict', 'r')
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
    a = ' '.join(a)
    pronunciation_sorted.append(a)

df = pd.DataFrame({
        "word": word,
        "pronunciation": pronunciation,
        "pronunciation_sorted": pronunciation_sorted
    })

# add placeholder columns
df['ARPAbetagrams'] = ''
df['index'] = df.index
df[:10]
%%time
def fillARPAbetagrams(line):
    word = line[0]
    cp = line[1]
    cpa = line[2]
    p = 0
    i = line[3]
    if i % 1350 == 0:
        print(str(i/1350)+'% done')
    
    pg = df.loc[(df['pronunciation_sorted'] == cpa) & (df['pronunciation'] != cp)]['word'].values.tolist()
    
    pg = ','.join(pg)
    h = ''
    return pg
df['ARPAbetagrams'] = df[['word', 'pronunciation', 'pronunciation_sorted', 'index']].apply(fillARPAbetagrams, axis = 1)

df.drop(['index'], axis=1)
# df.loc[(df['word'] == 'accord')]
df[:50]
df.to_csv("ARPAbetagrams_Dataset.csv", index=False, header=True)