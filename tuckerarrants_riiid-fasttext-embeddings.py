import fasttext, string, collections

import pandas as pd, numpy as np

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from tqdm import tqdm
#full length of dataset is 101,230,332

N_SAMPLES = 1000000



train = pd.read_pickle('../input/riiid-answer-correctness-prediction-files/train.pkl')

train = train[train['content_type_id'] == 0]



questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv').rename({'question_id':'content_id'},

                                                                                        axis=1)



print(train.shape)

train = train.sample(N_SAMPLES, random_state=34)

print(train.shape)

train = pd.merge(train, questions, how='left', on='content_id')

print(train.shape)

print(train['user_id'].nunique())

train.head()
print(f"There are {train['content_id'].nunique()} unique content ids in train.csv")

print(f"On average, students get {round(train['answered_correctly'].mean(), 3)} of questions correct")
train['content_id'].value_counts()
max_ = 0

iter_ = 0

for i, row in enumerate(train['tags'].values):

    if len(row.split()) > max_: 

        max_ = len(row.split())

        iter_ = i

train.iloc[iter_]
tags = []



for row in tqdm(train.index):

    tags.append(train.iloc[row]['tags'].split())
#check nunique tags

len(set([tag for sublist in tags for tag in sublist]))
questions['part'].value_counts(normalize=True)
print(questions['bundle_id'].nunique())

questions['bundle_id'].value_counts()
greek = '\u03A9\u0394\u03BC\u00B0\u0302\u03C0\u03F4\u03BB\u03B1\u03B3\u03B4\u03B5\u03B6\u03B7\u03B8\u03B9\u03BA\u03BD\u03BE\u03C1\u03C2\u03C3\u03C4\u03C5\u03C6\u03C7\u03C8\u03C9\u0391\u0395\u0396\u0397\u0398\u0399\u039A\u039B\u039C\u039D\u039E\u039F\u03A0\u03A1\u03A3\u03A4\u03A5\u03A6\u03A7\u03A8'

greek
cyrillic = '\u0410\u0430\u0411\u0413\u0414\u0415\u0416\u0417\u0418\u0419\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u041a\u041b\u041c\u041d\u041e\u041f\u043a\u043b\u043c\u043d\u043e\u043f\u0420\u0421\u0422\u0423\u0424\u0425\u0426\u0427\u0428\u0429'

cyrillic
latin = '\u00A1\u00A2\u00A3\u00A4\u00A5\u00A6\u00A7\u00A8\u00A9\u00B0\u00B1\u00B2\u00B3\u00B4\u00B5\u00B6\u00B7\u00B8\u00B9\u00C0\u00C1\u00C3\u00C5\u00C6\u00C7\u00C8\u00C9\u00D0\u00D1\u00D2\u00D4\u00D5\u00D6\u00D7\u00D8\u00D9'

latin
#not using 012 because they are reserved for labels

chars = '3456789' + string.ascii_letters + greek + cyrillic + latin

len(chars)
most_common = [word for word, word_count in collections.Counter([tag for sublist in tags for tag in sublist]).most_common(len(chars))]

least_common = [word for word, word_count in collections.Counter([tag for sublist in tags for tag in sublist]).most_common()[len(chars):]]

len(most_common)
char_dict = {}



for tag, char in zip(most_common, chars):

    char_dict[tag] = char
#remove low frequency tags and maps to unique character

def filter_tags(tag):

    return_tag = []

    

    for tag in tag.split():

        if tag not in least_common:

            tag = char_dict.get(tag)

            return_tag.append(tag)

            

    return " ".join(return_tag)

     

train['tags'] = train['tags'].apply(filter_tags)
#I know I know, this is sloppy

def pad_tags(tag):

    if len(tag.split()) == 1:

        return "".join([tag]*6)

    

    if len(tag.split()) == 2:

        tag1 = tag.split()[0]

        tag2 = tag.split()[1]

        return "".join([tag1, tag2]*3)

    

    if len(tag.split()) == 3:

        tag1 = tag.split()[0]

        tag2 = tag.split()[1]

        tag3 = tag.split()[2]

        return "".join([tag1, tag2, tag3]*2)

    

    if len(tag.split()) == 4:

        tag1 = tag.split()[0]

        tag2 = tag.split()[1]

        tag3 = tag.split()[2]

        tag4 = tag.split()[3]

        return "".join([tag1, tag2, tag3, tag4, tag1, tag2])

    

    if len(tag.split()) == 5:

        tag1 = tag.split()[0]

        tag2 = tag.split()[1]

        tag3 = tag.split()[2]

        tag4 = tag.split()[3]

        tag5 = tag.split()[4]

        return "".join([tag1, tag2, tag3, tag4, tag5, tag1])

    

    else: return tag      

    

train['tags'] = train['tags'].apply(pad_tags)
train['answered_correctly'] = train['answered_correctly'].apply(str)

train['answered_correctly'] = train['answered_correctly'].replace({'-1':'2'})

train['interaction_enc'] = train['tags'] + train['answered_correctly']
#debugging step

i = 0

for i in range(100000):

    if len([_ for _ in train.iloc[i]['interaction_enc']]) != 7: print('stop'); print(i); break
with open('corpus.txt', 'w') as file:

    for user in tqdm(train['user_id'].unique()):

        user_df = train[train['user_id'] == user]

        line=' '.join(user_df['interaction_enc'].values)

        file.write(line+'\n')
cbow = fasttext.train_unsupervised('corpus.txt', model='cbow',

                                    dim=200, minn=1, maxn=1, ws=6)

skipgram = fasttext.train_unsupervised('corpus.txt', model='skipgram',

                                    dim=200, minn=1, maxn=1, ws=6)

print(f"{cbow.get_output_matrix().mean()}")                                                                                                                    

print(f"{skipgram.get_output_matrix().mean()}")
#sanity check

cbow.get_subwords('Xy5Xy51')
# https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne/comments

def tsne_plot(model):

    labels = []

    tokens = []



    for word in model.words:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2000,

                      n_jobs=-1, random_state=34)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(15, 15)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        

        if i%50==0:

            plt.annotate(labels[i],

                         xy=(x[i], y[i]),

                         xytext=(5, 2),

                         textcoords='offset points',

                         ha='right',

                         va='bottom')

    plt.show()
tsne_plot(cbow)
tsne_plot(skipgram)
question_vectors = list(zip(train['tags'].values, [cbow.get_word_vector(word) for word in train['tags']]))

question_vectors