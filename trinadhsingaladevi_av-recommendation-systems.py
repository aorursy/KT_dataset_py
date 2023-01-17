
# import the required libraries

from collections import Counter
from scipy import sparse
import numpy as np
import pandas as pd
import pickle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# reading train and test file

train = pd.read_csv("../input/av-recommendation-systems/train_mddNHeX/train.csv")
test = pd.read_csv("../input/av-recommendation-systems/test_HLxMpl7/test.csv")
print(train.shape)
train.head()

# convert the train in the long format to wide format

wide_train = train.pivot_table(index = "user_id", columns="challenge_sequence", values="challenge", aggfunc= lambda x : x).reset_index()
# dropping the user_id, since we won't be needing those for our co-occurrence matrix
wide_train.drop('user_id',axis = 1,inplace = True)

wide_train.head(20)
# convert each row for a user into a string

rows = []
for index, row in wide_train.iterrows():
    r = " ".join(row.map(str))
    rows.append(r)

# converting test to wide format

wide_test = test.pivot_table(index = "user_id", columns="challenge_sequence", values="challenge", aggfunc= lambda x : x).reset_index()
wide_test.shape
# saving test user_id for future use

test_ids = wide_test['user_id']
# dropping user_id from wide test

wide_test.drop(["user_id"], axis =1, inplace = True)
for index, row in wide_test.iterrows():
    r = " ".join(row.map(str))
    rows.append(r)
# creating a corpus
thefile = open("corpus.txt","w")
for element in rows:
    thefile.write("%s\n"%element)
thefile.close()
# reading the corpus

corpus = open("corpus.txt","r")
corpus
# creating a dictionary with key = challenge_name and value = frequency
vocab = Counter()
for line in corpus:
    tokens = line.strip().split()
    vocab.update(tokens)
vocab = {word:(i,freq) for i,(word,freq) in enumerate(vocab.items())}
    
vocab
id2word = dict((i, word) for word, (i, _) in enumerate(vocab.items()))

id2word
vocab_size = len(vocab)
print(vocab_size)
cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),dtype=np.float64)
cooccurrences

# context window size

window_size = 10
corpus = open("../input/recommendation-corpus/corpus.txt","r")

# Tuneable parameters : window_size, distance

for i, line in enumerate(corpus):
    tokens = line.strip().split()
    token_ids = [vocab[word][0] for word in tokens]
    
    for center_i, center_id in enumerate(token_ids):
        # Collect all word IDs in left window of center word
        context_ids = token_ids[max(0, center_i - window_size) : center_i]
        contexts_len = len(context_ids)

        for left_i, left_id in enumerate(context_ids):
            # Distance from center word
            
            distance = contexts_len - left_i

            # Weight by inverse of distance between words
            increment = 1.0 / float(distance)

            # Build co-occurrence matrix symmetrically (pretend we
            # are calculating right contexts as well)
            cooccurrences[center_id, left_id] += increment
            cooccurrences[left_id, center_id] += increment
# If anything other than None will exclude challenges whose frequencies are below this value.

min_count = None
#min_count = 20
print(min_count)
# filling the values in a matrix form

co_matrix = np.zeros([len(id2word),len(id2word)])

for i, (row, data) in enumerate(zip(cooccurrences.rows,cooccurrences.data)):
    if min_count is not None and vocab[id2word[i]][0] < min_count:
        continue
        
    for data_idx, j in enumerate(row):
        if min_count is not None and vocab[id2word[j]][0] < min_count:
            continue
            
        co_matrix[i,j] = data[data_idx]
co_matrix
#saving the mapping to a dictionary
pickle_path = "../input/vocab-mapping/vocab_mapping.pkl"
#pickle_mapping = open(pickle_path,"wb")
#pickle.dump(id2word, pickle_mapping)
#pickle_mapping.close()
# saving the co-occurence matrix as a dataframe

co_occurence_dataframe = pd.DataFrame(co_matrix)
co_occurence_dataframe.head()

res = {v:k for k,v in id2word.items()}
co_occurence_dataframe =co_occurence_dataframe.rename(columns=res)
co_occurence_dataframe = co_occurence_dataframe.rename(index=res)
co_occurence_dataframe.to_csv("co_matrix_with_window_size_1.csv", index = False)
co_occurence_dataframe.head()
wide_test.head()
wide_test.shape

final_predictions = []

for i in range(0,39732):
    predictions = [wide_test.loc[i,10]]
    counter = 0
    for stimulus in predictions:
        predictions.append(co_occurence_dataframe[stimulus].idxmax())
        counter+=1
        if counter == 3:
            break
            
    final_predictions.append(predictions[1:])
# making predictions with the co-occurence_matrix based on 10th challenge only
final_predictions_new = []

for i in range(0,39732):
    stimulus = wide_test.loc[i,10]
    
    final_predictions_new.append(list(co_occurence_dataframe[stimulus].nlargest(3).index))
largest_3 = pd.DataFrame(final_predictions_new)
largest_3['user_id'] = test_ids

largest_3.head()
largest_3_long = pd.melt(largest_3,id_vars="user_id",var_name="sequence", value_name="challenge" )
final_predictions
sub = pd.read_csv('../input/av-recommendation-systems/sample_submission_J0OjXLi_DDt3uQN.csv')
seq = []
for i in final_predictions:
    for j in i:
        seq.append(j)
sub['challenge'] = seq
sub.to_csv('nlp_corr.csv',index = False)
df = pd.read_csv("../input/av-recommendation-systems/train_mddNHeX/challenge_data.csv")
print(df.shape)
df.head()
print(train.shape)
train.head()
train.rename(columns = {'challenge' : 'challenge_ID'},inplace = True)
test.rename(columns = {'challenge' : 'challenge_ID'},inplace = True)
train.head()
df2 = df.merge(train,on = 'challenge_ID')
df2_test = df.merge(test,on = 'challenge_ID')
print(df2.shape)
df2.head()
print(df2_test.shape)
df2_test.head()
print("No. of challenges: ", df2['challenge_ID'].nunique())
print("No. of progaming languages: ", df2['programming_language'].nunique())
print("Minimum submissions: ", df2['total_submissions'].min())
print("Max submissions: ", df2['total_submissions'].max())
print("No. of Authors: ", df2['author_ID'].nunique())
print("No. org of Authors: ", df2['author_org_ID'].nunique())
print("No. categories: ", df2['category_id'].nunique())
df2.isnull().sum()
df[['challenge_ID','programming_language']].groupby('challenge_ID').count()
df2[['user_id','programming_language']].groupby('user_id').nunique().sort_values(by = 'programming_language',ascending = False)
#df2[df2['user_id'] == 77954]
df2['programming_language'] = df2['programming_language'].astype(str)
df2['total_submissions'] = df2['total_submissions'].astype(str)
df2['category_id'] = df2['category_id'].astype(str)
def create_soup(x):
    return ' '.join(x['programming_language']) + ' ' + ' '.join(x['total_submissions']) + ' ' + x['category_id'] 
df2['soup'] = df2.apply(create_soup, axis=1)
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

type(count_matrix)
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
