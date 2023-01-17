import logging
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import re
import nltk
import pprint
from nltk.corpus import stopwords
from IPython.core.display import display, HTML
from sklearn.feature_extraction.text  import TfidfVectorizer 
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from gensim.models import word2vec
display(HTML("""
<button id='toggleInput'>Hide Input</button> 
<script>
	var $toggleInput = document.querySelector('#toggleInput')
	$toggleInput.addEventListener('click', function() {
		if($toggleInput.innerText === 'Hide Input') {
			$toggleInput.innerText = 'Show Input';
			document.querySelectorAll('.input').forEach($input => $input.style.display = 'none');
		} else {
			$toggleInput.innerText = 'Hide Input';
			document.querySelectorAll('.input').forEach($input => $input.style.display = 'flex');
		}
	});
</script>
"""))
display(HTML("""<h2 id='inputs'>Inputs</h2>"""))

data_file_name = '../input/heckyeh/output.tsv'
percent_training = 0.75
sentiment_cutoff = 7
vocab_size = 2000
n_estimators = 100
display(HTML("""<h2 id='dataCollection'>Data Collection</h2>"""))

data = pd.read_csv(data_file_name, delimiter='\t')
pd.options.display.max_colwidth = 10000000
data.head(10)

print('this is the number of rows: %d , and this is the number of columns: %d' % data.shape)

display(HTML("""<h2 id='featureExtraction'>Feature Extraction</h2>"""))

punktTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def get_words(sentence):
    clean_sentence = re.sub("[^a-z]"," ", sentence.lower())
    return [word for word in clean_sentence.split(' ') if len(word) > 0]

def get_sentences(review):
    clean_review = review.strip()
    return [get_words(sentence) for sentence in punktTokenizer.tokenize(clean_review) if len(sentence) > 0 ]

sentences = []
for review in data["review"]:
    sentences += get_sentences(review)

print("THERE ARE %s SENTENCES" % len(sentences))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
level=logging.INFO)

num_features = 100# Word vector dimensionality                      
min_word_count = 2   # Minimum word count                    
num_workers = 4   # Number of threads to run in parallel
context = 10      # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count,window = context, sample = downsampling)
model.init_sims(replace=True)

model_name = "imdb"
model.save(model_name)
model.doesnt_match('good bad cool awesome stomach'.split(' '))

data['rating'].replace('', np.nan, inplace=True)
data.dropna(subset=['rating'], inplace=True)
print('this is the number of rows: %d , and this is the number of columns: %d' % data.shape)
words_to_remove = stopwords.words("english")

extra_words = ['film','movie', 'one', 'story', 'see', 'ring', 'also']
words_to_remove += extra_words

print(words_to_remove)




interesting_words = data['review'].map(lambda r: [re.sub("[^a-zA-Z]", "", word) for word in r.lower().split() if not word in words_to_remove])
interesting_words = interesting_words.map(lambda words: [w for w in words if not w in words_to_remove])
interesting_words = interesting_words.map(lambda words: [w for w in words if re.sub('[aeiouy]', '', w) != w])

data['review'] = interesting_words.map(lambda words: [w for w in words if len(w)])

print(data['review'])
data = data.assign(sentiment=data['rating'].map(lambda r: 1 if r >= sentiment_cutoff else 0))
data[data['rating'] == 8].head(10)
display(HTML("""<h2 id='trainingDataSelection'>Training Data Selection</h2>"""))

row_count = int(percent_training*float(data.shape[0]))
training_data = data.sample(n=row_count)
training_data.info()
display(HTML("""<h2 id='testingDataSelection'>Testing Data Selection</h2>"""))

training_data_ids = [id for id in training_data.id]
testing_data = data[~data['id'].isin(training_data_ids)]
testing_data.info()
count_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = None, max_features = vocab_size)
all_words = []
for word_list in training_data['review']:
    all_words.append(' '.join(word_list))

count_train_data_features = count_vectorizer.fit_transform(all_words).toarray()
print(count_train_data_features.shape)
tfidf_vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,stop_words = None, max_features = vocab_size)
all_words = []
for word_list in training_data['review']:
    all_words.append(' '.join(word_list))

tfidf_train_data_features = tfidf_vectorizer.fit_transform(all_words).toarray()
print(tfidf_train_data_features.shape)

display(HTML("""<h2 id='modelTraining'>Model Training</h2>"""))

count_vocab = count_vectorizer.get_feature_names()
print(vocab)
print('There are %d number of words in our vocab' % len(count_vocab))

dist = np.sum(count_train_data_features, axis=0)

hist = [];
for tag, count in zip(count_vocab, dist):
    hist.append((count, tag))
    
print("The top ten word list is ")

# IMPORT pprint IN THE IMPORT BLOCK
pprint.pprint(sorted(hist, key= lambda x : x[0], reverse=True)[0:10])

print ("Training the random forest...")

count_forest = RandomForestClassifier(n_estimators=n_estimators) 

count_forest = forest.fit(count_train_data_features, training_data["sentiment"] )
print ("Trained the random forest model")
tfidf_vocab = tfidf_vectorizer.get_feature_names()
print(tfidf_vocab)
print('There are %d number of words in our vocab' % len(tfidf_vocab))

dist = np.sum(tfidf_train_data_features, axis=0)

hist = [];
for tag, count in zip(tfidf_vocab, dist):
    hist.append((count, tag))
    
print("The top ten word list is ")

# IMPORT pprint IN THE IMPORT BLOCK
pprint.pprint(sorted(hist, key= lambda x : x[0], reverse=True)[0:10])

print ("Training the random forest...")

tfidf_forest = RandomForestClassifier(n_estimators=n_estimators) 

tfidf_forest = tfidf_forest.fit(tfidf_train_data_features, training_data["sentiment"] )
print ("Trained the random forest model")
display(HTML("""<h2 id='predictionAnalysis'>Prediction Analysis</h2>"""))

test_words = []
for word_list in testing_data['review']:
    test_words.append(' '.join(word_list))

count_test_data_features = count_vectorizer.transform(test_words).toarray()
tfidf_test_data_features = count_vectorizer.transform(test_words).toarray()

print(count_test_data_features.shape)
count_result = count_forest.predict(count_test_data_features)
print(count_result)
tfidf_result = tfidf_forest.predict(tfidf_test_data_features)
print(tfidf_result)
count_output = pd.DataFrame(data ={"actual": testing_data['sentiment'], 'predicted':count_result})

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0


actual = count_output['actual'].tolist()
predicted = count_output['predicted'].tolist()
eventCount = float(count_output.shape[0]);

for x in range(count_output.shape[0]):
    a = actual[x]
    p = predicted[x]
    if a == 1 and p == 1:
        true_positives += 1
    elif a == 1 and p == 0:
        false_negatives += 1
    elif a == 0 and p == 1:
        false_positives += 1
    elif a == 0 and p == 0:
        true_negatives += 1

display(HTML("""
    <h3>Confusion Matrix</h3>
    <table>
        <tr>
            <th>Confusion Matrix Cell</th><th>Term</th><th>Value</th>
        </tr>
        <tr>
            <th>True Positive</th><td>Sensitivity</td><td>%1.2f</td>
        </tr>
        <tr>
            <th>False Positive</th><td>Fall-Out Rate</td><td>%1.2f</td>
        </tr>
        <tr>
            <th>False Negative</th><td>Miss Rate</td><td>%1.2f</td>
        </tr>
        <tr>
            <th>True Negative</th><td>Specificity</td><td>%1.2f</td>
        </tr>
    </table>
""" % (float(true_positives) / eventCount, float(false_negatives) /eventCount, float(false_positives)/eventCount,float(true_negatives)/eventCount)))
tfidf_output = pd.DataFrame(data ={"actual": testing_data['sentiment'], 'predicted':tfidf_result})

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0


actual = tfidf_output['actual'].tolist()
predicted = tfidf_output['predicted'].tolist()
eventCount = float(tfidf_output.shape[0]);

for x in range(tfidf_output.shape[0]):
    a = actual[x]
    p = predicted[x]
    if a == 1 and p == 1:
        true_positives += 1
    elif a == 1 and p == 0:
        false_negatives += 1
    elif a == 0 and p == 1:
        false_positives += 1
    elif a == 0 and p == 0:
        true_negatives += 1

display(HTML("""
    <h3>Confusion Matrix</h3>
    <table>
        <tr>
            <th>Confusion Matrix Cell</th><th>Term</th><th>Value</th>
        </tr>
        <tr>
            <th>True Positive</th><td>Sensitivity</td><td>%1.2f</td>
        </tr>
        <tr>
            <th>False Positive</th><td>Fall-Out Rate</td><td>%1.2f</td>
        </tr>
        <tr>
            <th>False Negative</th><td>Miss Rate</td><td>%1.2f</td>
        </tr>
        <tr>
            <th>True Negative</th><td>Specificity</td><td>%1.2f</td>
        </tr>
    </table>
""" % (float(true_positives) / eventCount, float(false_negatives) /eventCount, float(false_positives)/eventCount,float(true_negatives)/eventCount)))