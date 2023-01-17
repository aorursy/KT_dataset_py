# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(ds) * 5
import random

from random import shuffle

random.seed(1)



stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 

            'ours', 'ourselves', 'you', 'your', 'yours', 

            'yourself', 'yourselves', 'he', 'him', 'his', 

            'himself', 'she', 'her', 'hers', 'herself', 

            'it', 'its', 'itself', 'they', 'them', 'their', 

            'theirs', 'themselves', 'what', 'which', 'who', 

            'whom', 'this', 'that', 'these', 'those', 'am', 

            'is', 'are', 'was', 'were', 'be', 'been', 'being', 

            'have', 'has', 'had', 'having', 'do', 'does', 'did',

            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',

            'because', 'as', 'until', 'while', 'of', 'at', 

            'by', 'for', 'with', 'about', 'against', 'between',

            'into', 'through', 'during', 'before', 'after', 

            'above', 'below', 'to', 'from', 'up', 'down', 'in',

            'out', 'on', 'off', 'over', 'under', 'again', 

            'further', 'then', 'once', 'here', 'there', 'when', 

            'where', 'why', 'how', 'all', 'any', 'both', 'each', 

            'few', 'more', 'most', 'other', 'some', 'such', 'no', 

            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 

            'very', 's', 't', 'can', 'will', 'just', 'don', 

            'should', 'now', '']



import re

def get_only_chars(line):



    clean_line = ""



    line = line.replace("’", "")

    line = line.replace("'", "")

    line = line.replace("-", " ") #replace hyphens with spaces

    line = line.replace("\t", " ")

    line = line.replace("\n", " ")

    line = line.lower()



    for char in line:

        if char in 'qwertyuiopasdfghjklzxcvbnm ':

            clean_line += char

        else:

            clean_line += ' '



    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces

    #if clean_line[0] == ' ':

    #    clean_line = clean_line[1:]

    return clean_line

from nltk.corpus import wordnet 



def synonym_replacement(words, n):

    new_words = words.copy()

    random_word_list = list(set([word for word in words if word not in stop_words]))

    random.shuffle(random_word_list)

    num_replaced = 0

    for random_word in random_word_list:

        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:

            synonym = random.choice(list(synonyms))

            new_words = [synonym if word == random_word else word for word in new_words]

            #print("replaced", random_word, "with", synonym)

            num_replaced += 1

        if num_replaced >= n: #only replace up to n words

            break



    #this is stupid but we need it, trust me

    sentence = ' '.join(new_words)

    new_words = sentence.split(' ')



    return new_words



def get_synonyms(word):

    synonyms = set()

    for syn in wordnet.synsets(word): 

        for l in syn.lemmas(): 

            synonym = l.name().replace("_", " ").replace("-", " ").lower()

            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])

            synonyms.add(synonym) 

    if word in synonyms:

        synonyms.remove(word)

    return list(synonyms)



def random_deletion(words, p):



    #obviously, if there's only one word, don't delete it

    if len(words) == 1:

        return words



    #randomly delete words with probability p

    new_words = []

    for word in words:

        r = random.uniform(0, 1)

        if r > p:

            new_words.append(word)



    #if you end up deleting all words, just return a random word

    if len(new_words) == 0:

        rand_int = random.randint(0, len(words)-1)

        return [words[rand_int]]



    return new_words



def random_swap(words, n):

    new_words = words.copy()

    for _ in range(n):

        new_words = swap_word(new_words)

    return new_words



def swap_word(new_words):

    random_idx_1 = random.randint(0, len(new_words)-1)

    random_idx_2 = random_idx_1

    counter = 0

    while random_idx_2 == random_idx_1:

        random_idx_2 = random.randint(0, len(new_words)-1)

        counter += 1

        if counter > 3:

            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 

    return new_words



def random_insertion(words, n):

    new_words = words.copy()

    for _ in range(n):

        add_word(new_words)

    return new_words



def add_word(new_words):

    synonyms = []

    counter = 0

    while len(synonyms) < 1:

        random_word = new_words[random.randint(0, len(new_words)-1)]

        synonyms = get_synonyms(random_word)

        counter += 1

        if counter >= 10:

            return

    random_synonym = synonyms[0]

    random_idx = random.randint(0, len(new_words)-1)

    new_words.insert(random_idx, random_synonym)



def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):



    sentence = get_only_chars(sentence)

    words = sentence.split(' ')

    words = [word for word in words if word is not '']

    num_words = len(words)



    if num_words == 0:

        return [""] * num_aug

    augmented_sentences = []

    num_new_per_technique = int(num_aug/4)+1

    n_sr = max(1, int(alpha_sr*num_words))

    n_ri = max(1, int(alpha_ri*num_words))

    n_rs = max(1, int(alpha_rs*num_words))



    #sr

    for _ in range(num_new_per_technique):

        a_words = synonym_replacement(words, n_sr)

        augmented_sentences.append(' '.join(a_words))



    #ri

    for _ in range(num_new_per_technique):

        a_words = random_insertion(words, n_ri)

        augmented_sentences.append(' '.join(a_words))



    #rs

    for _ in range(num_new_per_technique):

        a_words = random_swap(words, n_rs)

        augmented_sentences.append(' '.join(a_words))



#rd

    for _ in range(num_new_per_technique):

        a_words = random_deletion(words, p_rd)

        augmented_sentences.append(' '.join(a_words))



    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]

    shuffle(augmented_sentences)



#trim so that we have the desired number of augmented sentences

    if num_aug >= 1:

        augmented_sentences = np.random.choice(augmented_sentences, num_aug, replace=False)

    else:

        keep_prob = num_aug / len(augmented_sentences)

        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]





    return augmented_sentences
#lets look what function does

sentence = "Hello! This function creates new examples from data"

augs = eda(sentence)

for element in augs:

    print(element)

    print()
sentence = " Hello! This function creates new examples from data"

augs = eda(sentence, alpha_sr=0.5, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1)

for element in augs:

    print(element)

    print()
#try with toxic

sentence = "What a fuck are you doing man. It's a fucking bullshit"

augs = eda(sentence, alpha_sr=0.3, alpha_ri=0.3, alpha_rs=0.3, p_rd=0.3)

for element in augs:

    print(element)

    print()
#try with toxic

sentence = "What a fuck are you doing man. It's a fucking bullshit"

augs = eda(sentence, alpha_sr=0.3, alpha_ri=0.2, alpha_rs=0.2, p_rd=0.2)

for element in augs:

    print(element)

    print()
def augment_data(toxic_df, alpha_sr=0.3, alpha_ri=0.2, alpha_rs=0.2, p_rd=0.2, num_aug=4):

    pos = 0

    sentences = []

    while pos < len(toxic_df):

        for sent in toxic_df["comment_text"][pos:pos+10000]:

            sentences.extend(eda(sent, num_aug=num_aug))

        pos += 10000

        print("Processed", pos, "sentences")

    #sentences = np.concatenate(sentences).tolist()

    labels = len(sentences) * [1]

    return sentences, labels
large_ds = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text","toxic"]).query("toxic > 0.5")

small_ds = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text","toxic"]).query("toxic==1")



ds = pd.concat((large_ds,small_ds))

ds["length"] = ds.comment_text.str.split().apply(len)

del large_ds

del small_ds

ds = ds[ds["length"] > 0]
aug_sents, aug_labels = augment_data(ds)
aug_df = pd.DataFrame({"comment_text":aug_sents, "toxic":aug_labels})

aug_df.to_csv("aug.csv")