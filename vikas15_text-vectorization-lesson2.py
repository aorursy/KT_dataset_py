from IPython.core.display import display, HTML

from IPython.display import Image

import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import normalize
# Document and Word Vectors

Image('https://mlwhiz.com/images/countvectorizer.png',width=800, height=400)
# create a list of documents

# text = ['This is the first document'

#         , 'This is the second second document'

#         , 'And the third one'

#         , 'Is it the first document again']



text=['This is good',

     'This is bad',

     'This is awesome']
from sklearn.feature_extraction.text import CountVectorizer
# create an instance of countvectorizer

vect = CountVectorizer()  # shift tab 
# when we print vect, we see its hyperparameters

print(vect)
# The vectorizer learns the vocabulary when we fit it with our documents. 

# This means it learns the distinct tokens (terms) in the text of the documents. 

# We can observe these with the method get_feature_names



vect.fit(text)
print('ORIGINAL_SENTENCES: \n {} \n'.format(text))

print('FEATURE_NAMES: \n {}'.format(vect.get_feature_names()))
# Transform creates a sparse matrix, identifying the indices where terms are stores in each document

# This sparse matrix has 4 rows and 11 columns



pd.DataFrame(vect.transform(text).toarray(),columns= ['awesome', 'bad', 'good', 'is', 'this'])[ ['this','is','good','bad','awesome']]
print(vect.transform(text))
sparse_matrix_url = 'https://op2.github.io/PyOP2/_images/csr.svg'

iframe = '<iframe src={} width=1000 height=200></iframe>'.format(sparse_matrix_url)

HTML(iframe)
# This is easier to understand when we covert the sparse matrix into a dense matrix or pandas DataFrame

vect.transform(text).toarray()
import pandas as pd



# store the dense matrix

data = vect.transform(text).toarray()



# store the learned vocabulary

columns = vect.get_feature_names()



# combine the data and columns into a dataframe

pd.DataFrame(data, columns=columns)[['this','is','good','bad','awesome']]
example_text = ['again we observe a document'

               , 'the second time we have see this text']
# TODO

vect = CountVectorizer()

vect.fit_transform(text).toarray()
Image('http://karlrosaen.com/ml/learning-log/2016-06-20/pipeline-diagram.png')
text = ['This is the first document'

        , 'This is the second second document'

        , 'And the third one'

        , 'Is it the first document again']
vect = CountVectorizer()
# by instantiating CountVectorizer with differnt parameters, we can change the vocabulary

# lowercase determines if all words should be lowercase, setting it to False includes uppercase words



vect = CountVectorizer(lowercase=False)

vect.fit(text)

print(vect.get_feature_names())
# stops words determine if we should include common words (e.g. and, is, the) which show up in most documents

vect = CountVectorizer(stop_words='english')

vect.fit(text)

print(vect.get_feature_names())
# stops words determine if we should include common words (e.g. and, is, the) which show up in most documents

vect = CountVectorizer(stop_words=['first','second','third'])

vect.fit(text)

print(vect.get_feature_names())
# stops words determine if we should include common words (e.g. and, is, the) which show up in most documents

vect = CountVectorizer(vocabulary=['first','second','third'])

vect.fit(text)

print(vect.get_feature_names())
vect.transform(text).toarray()
vect = CountVectorizer(max_features=5)

vect.fit(text)

print(vect.get_feature_names())
vect = CountVectorizer(max_df=.5)

vect.fit(text)

print(vect.get_feature_names())
vect = CountVectorizer(min_df=.5)

vect.fit(text)

print(vect.get_feature_names())
# max features determines the maximum number of features to display

vect = CountVectorizer(ngram_range=(1,2), max_features=5)

vect.fit(text)

print(vect.get_feature_names())
# max features determines the maximum number of features to display

vect = CountVectorizer(binary=True)

vect.fit_transform(['Two Two different words words']).toarray()
# max features determines the maximum number of features to display

vect = CountVectorizer(analyzer='char', ngram_range=(2,2))

vect.fit(text)

print(vect.get_feature_names())
vect = CountVectorizer(max_features=5)

vect.fit(text)

print(vect.get_feature_names())
vect.vocabulary_
vect.stop_words_
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
text
tfidf_vect = TfidfVectorizer()

pd.DataFrame(tfidf_vect.fit_transform(text).toarray(), columns=tfidf_vect.get_feature_names())
vect = CountVectorizer()

tf = vect.fit_transform(text).toarray()

pd.DataFrame(tf,columns= vect.get_feature_names())
Image('http://www.science4all.org/wp-content/uploads/2013/10/Graph-of-Logarithm-and-Exponential1.png')
tf
len(tf)
vect = CountVectorizer(binary=True)

count_vec = vect.fit_transform(text).toarray()

pd.DataFrame(count_vec,columns= vect.get_feature_names())
len(count_vec)


# idf calculation

print( np.log(len(count_vec) / count_vec.sum(axis=0)) )
list(zip(vect.get_feature_names(),np.log(len(count_vec) / count_vec.sum(axis=0))))
# when we use sum(axis=0) we take the sum of each column

# as opposed to a scalar sum (single # result) of all values

count_vec.sum(axis=0)
idf = np.log( (len(count_vec)+1) / (count_vec.sum(axis=0)+1) ) + 1

print(idf)
# value as stored from sklearn in tfidf_vect

print(tfidf_vect.idf_)
tfidf = pd.DataFrame(tf*idf,columns=tfidf_vect.get_feature_names())

tfidf
# tf*idf is equivalent to using TfidfVectorizer without a norm

tfidf_vect = TfidfVectorizer(norm=None)

pd.DataFrame(tfidf_vect.fit_transform(text).toarray())
from sklearn.preprocessing import normalize



pd.DataFrame(normalize(tfidf, norm='l2'))
# normalize()
# TFIDF Weighting in Sklearn

'http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting'



# tf*idf is equivalent to using TfidfVectorizer without a norm

tfidf_vect = TfidfVectorizer(norm='l2')

pd.DataFrame(tfidf_vect.fit_transform(text).toarray())
import os

os.listdir('../input')
path='../input/usinlppracticum/imdb_train.csv'

data= pd.read_csv(path)

data.head()
labels=data['sentiment'].unique().tolist()

label2id={ lbl:i for i,lbl in enumerate(labels)}

id2label={ i:lbl for i,lbl in enumerate(labels)}

print(label2id), print(id2label)
data['label']=data['sentiment'].map(label2id)

data.head()
data.shape
import spacy, string

nlp = spacy.load('en')

punctuations = string.punctuation

from spacy.lang.en.stop_words import STOP_WORDS

def cleanup_text(doc):

    doc = nlp(doc, disable=['parser', 'ner'])

    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

#   print (tokens)

    tokens = " ".join([i for i in tokens if i not in STOP_WORDS and len(i)>2]) 

#     tokens = ' '.join(tokens)

    return tokens
print(cleanup_text(data['review'][1]))
data= data.sample(1000).reset_index(drop=True)
from tqdm import tqdm

tqdm.pandas()
data['clean_review']=data['review'].progress_apply(lambda x:cleanup_text(x))

data.head()
# split the dataset into training and validation datasets 

from sklearn import model_selection

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['clean_review'], data['label'],test_size=0.2, random_state=42)
train_x.head()
# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word',max_df=0.95, min_df=5,ngram_range=(1,1),

                             max_features=1500)



tfidf_vect.fit(train_x) #--- -

xtrain_tfidf =  tfidf_vect.transform(train_x)

xvalid_tfidf =  tfidf_vect.transform(valid_x)
xtrain_tfidf.shape,xvalid_tfidf.shape
features_name=tfidf_vect.get_feature_names()

features_name[:30]
from sklearn import ensemble

# RF on Word Level TF IDF Vectors

model=ensemble.RandomForestClassifier(n_estimators=50, random_state=0)

model.fit(xtrain_tfidf, train_y)

from sklearn import metrics

predictions = model.predict(xvalid_tfidf)

accuracy=metrics.accuracy_score(predictions, valid_y)

print ("Random Forest accuracy for validation: ", accuracy)
Image('http://cs.carleton.edu/cs_comps/0910/netflixprize/final_results/knn/img/knn/cos.png')
from sklearn.metrics.pairwise import linear_kernel



def find_similar(tfidf_matrix, index, top_n = 5):

    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()

    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]

    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]
find_similar(xtrain_tfidf, 1)