# Use LDA model in Gensim package
import pandas as pd
import nltk
from gensim.models import Phrases
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis.gensim
df=pd.read_csv('../input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv')
positive=df.Positive_Review.tolist()
negative=df.Negative_Review.tolist()
from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer(r'\w+')

# tokenize
for idx in range(len(positive)):
    positive[idx]=positive[idx].lower()
    positive[idx]=tokenizer.tokenize(positive[idx])
    
# tokenize
for idx in range(len(negative)):
    negative[idx]=negative[idx].lower()
    negative[idx]=tokenizer.tokenize(negative[idx])

    # remove the tokens whose length is one and which is a number
positive=[[token for token in pos if len(token)>1] for pos in positive]
positive=[[token for token in pos if not token.isnumeric()] for pos in positive]

negative=[[token for token in neg if len(token)>1] for neg in negative]
negative=[[token for token in neg if not token.isnumeric()] for neg in negative]


# Romove stopwords
stpwd=set(stopwords.words('english'))
positive=[[token for token in pos if token not in stpwd] for pos in positive]
negative=[[token for token in neg if token not in stpwd ] for neg in negative]

# lemmatize
from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()
positive=[[lemmatizer.lemmatize(token) for token in pos] for pos in positive]
negative=[[lemmatizer.lemmatize(token) for token in neg] for neg in negative]

#find the bigram 
bigram_pos=Phrases(positive,min_count=20)
bigram_neg=Phrases(negative,min_count=20)

for idx in range(len(positive)):
    for token in bigram_pos[positive[idx]]:
        if '_' in token:
            positive[idx].append(token)
            
    for token in bigram_neg[negative[idx]]:
        if '_' in token:
            negative[idx].append(token)
# remove rare and ramdom tokens
dictionary_positive=Dictionary(positive)
dictionary_negative=Dictionary(negative)

dictionary_positive.filter_extremes(no_below=20, no_above=0.5)
dictionary_negative.filter_extremes(no_below=20, no_above=0.5)
# Bag-of-words representation of the documents.
corpus_positive= [dictionary_positive.doc2bow(pos) for pos in positive]
corpus_negative=[dictionary_negative.doc2bow(neg) for neg in negative]
num_topics=10
chunksize=2000
passes=20
iterations=400
eval_every=None
temp = dictionary_positive[0] 
id2word_positive = dictionary_positive.id2token 

model_positive = LdaModel(
    corpus=corpus_positive,
    id2word=id2word_positive,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

outputfile_pos = f'model_positive{num_topics}.gensim'
model_positive.save(outputfile_pos)
temp = dictionary_negative[0] 
id2word_negative = dictionary_negative.id2token 

model_negative = LdaModel(
    corpus=corpus_negative,
    id2word=id2word_negative,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

outputfile_neg = f'model_negative{num_topics}.gensim'
model_negative.save(outputfile_neg)
top_topics_pos = model_positive.top_topics(corpus_positive)
top_topics_neg =  model_negative.top_topics(corpus_negative)
#display the result
pos_display= pyLDAvis.gensim.prepare(model_positive,corpus_positive,dictionary_positive,sort_topics=True)
pyLDAvis.display(pos_display)
#display the result
neg_display= pyLDAvis.gensim.prepare(model_negative,corpus_negative,dictionary_negative,sort_topics=True)
pyLDAvis.display(neg_display)
