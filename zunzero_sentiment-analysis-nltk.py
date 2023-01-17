!pip install contractions
# # Standard library imports
import collections
import re
import warnings

# Third-party imports
import contractions
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import nltk
import numpy as np
import pandas as pd
import sklearn as sk
from tensorflow import keras
import unidecode
# Suppress matplotlib user warnings,
# necessary for newer version of matplotlib
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="matplotlib"
)

# Allows producing visualizations in notebook
%matplotlib inline

# Download required NLTK files 
nltk.download(['averaged_perceptron_tagger', 'punkt', 'wordnet'])
def bins_labels(bins, **kwargs):
    ''' Plot histogram helper function
    
    The code was extracted from Stack Overflow, answer by @Pietro Battiston:
    https://stackoverflow.com/questions/23246125/how-to-center-labels-in-histogram-plot
    
    Parameters
    ----------
    bins : list from start to end by given steps
        description -> The xticks to fit.
        format -> range(start, end, step)
        options -> No apply
    '''
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])
    
def print_text(dataframe, header='text', step=250):
    '''Function to print some tweet samples.
    
    The code prints on the given step a tweet with
    its index and sentiment classification.
    
    This function is for better understanding the most common
    tweet sintaxis and special characters. 
    
    Parameters
    ----------
    dataframe : Pandas Dataframe
        description -> The data text and labels
        format -> headers: ["text": string, "airline_sentiment": string]
        options -> "text": No apply
                   "airline_sentiment": ["positive", "neutral", "negative"]

    header : string
        description -> The object column to print
        format -> No apply
        options -> No apply

    step : int
        description -> The index separation desired to print
        format -> No apply
        options -> [0, len(dataframe) - 1]
    '''
    for index in range(0, len(dataframe), step):
        print(f'Tweet[{index}]: {dataframe[header][index]}')
        print(f'Sentiment: {dataframe["airline_sentiment"][index]}')
        print(f'{"_"*70}\n')

def text_cleaning(input_text):
    ''' Function including all the text clean process.
    
    The code includes all the steps required to clean
    the tweet texts on a final and common format.
    
    The output text is in lowercase.
    
    Parameters
    ----------
    input_text : string
        description -> The input text to clean
        format -> 'string'
        options -> No apply
    
    Returns
    -------
    output_text : string
        description -> The cleaned output text
        format -> 'string'
        options -> No apply
    '''
    input_text = re.sub(u'http\S+|@\S+|#', ' ', input_text)
    input_text = re.sub(u'^(.{140}).*$', '\g<1>', input_text)
    input_text = contractions.fix(input_text)
    input_text = unidecode.unidecode(input_text)
    input_text = re.sub(u'[^a-zA-Z]', ' ', input_text)
    input_text = input_text.lower()
    output_text = ' '.join(input_text.split())

    return output_text

def stem_sentence(input_text):
    ''' Function to stem a given sentence.
    
    The NLTK library is used on this function.
    
    Parameters
    ----------
    input_text : string
        description -> The input text to stem
        format -> 'string'
        options -> Only cleaned text in lowercase
    
    Returns
    -------
    output_text : string
        description -> The stemmed text
        format -> 'string'
        options -> No apply
    '''
    stem = nltk.stem.LancasterStemmer()
    
    words = nltk.word_tokenize(input_text)
    output_text = list(map(stem.stem, words))
    output_text = ' '.join(output_text)
    
    return output_text

# The following functions help to completly lemmatize a given sentence
def nltk_tag_to_wordnet_tag(nltk_tag):
    ''' Function to assign a tag to the detected word.
    
    The NLTK library is used on this function.

    Each word is analized and determines the word
    type as noun, verb, adverb or adjective.
    
    The function is based on the following link resources:
    https://medium.com/@gaurav5430/using-nltk-for-lemmatizing-sentences-c1bfff963258
    https://simonhessner.de/lemmatize-whole-sentences-with-python-and-nltks-wordnetlemmatizer/

    Parameters
    ----------
    nltk_tag : tuple
        description -> The tuple with the tokenized word and its type tag
        format -> nltk.pos_tag object
        options -> Only cleaned text in lowercase
    
    Returns
    -------
    tag : nltk.corpus.wordnet
        description -> The required tag before the lemmatization process.
        format -> nltk.corpus.wordnet
        options -> No apply
    '''
    if nltk_tag.startswith('J'):
        tag = nltk.corpus.wordnet.ADJ
    elif nltk_tag.startswith('V'):
        tag = nltk.corpus.wordnet.VERB
    elif nltk_tag.startswith('N'):
        tag = nltk.corpus.wordnet.NOUN
    elif nltk_tag.startswith('R'):
        tag = nltk.corpus.wordnet.ADV
    elif nltk_tag.startswith('S'):
        tag = nltk.corpus.wordnet.ADJ_SAT
    else:
        tag = None

    return tag

def adverb_to_base(word):
    ''' Lemmatization process on adverbs
    
    The NLTK library is used on this function.

    If the word tokenized tag indicates that the 
    word is an adverb, this function must be used
    to lemmatize the adverb.

    The function is based on the following link resources:
    https://stackoverflow.com/questions/17245123/getting-adjective-from-an-adverb-in-nltk-or-other-nlp-library

    Parameters
    ----------
    word : string
        description -> The tuple with the tokenized word and its type tag
        format -> nltk.pos_tag object
        options -> Only cleaned text in lowercase
    
    Returns
    -------
    tag : nltk.corpus.wordnet corresponding tag value
        description -> The required tag before the lemmatization process.
        format -> nltk.corpus.wordnet
        options -> No apply
    '''
    try:
        synonym = nltk.corpus.wordnet.synsets(word)
        synonym = [lemma for syn_set in synonym \
                   for lemma in syn_set.lemmas()]
        synonym = [pertain.name() for lemma in synonym \
                   for pertain in lemma.pertainyms()]
        base_word = difflib.get_close_matches(word, synonym)[0]
    except:
        base_word = word

    return base_word
    
def lemmatize_word(wordnet_tagged):
    ''' Lemmatization process on a single word
    
    The NLTK library is used on this function.

    A tagged and tokenized word is lemmatize.

    Parameters
    ----------
    wordnet_tagged : nltk.corpus.wordnet object
        description -> The word tag and corresponding word to lemmatize
        format -> nltk.corpus.wordnet corresponding tag value
        options -> no apply
    
    Returns
    -------
    base_word : string
        description -> The word on their base form
        format -> no apply
        options -> no apply
    '''
    lemma = nltk.stem.WordNetLemmatizer()
    if wordnet_tagged[1] == None:
        base_word = wordnet_tagged[0]
    elif wordnet_tagged[1] == 'r':
        base_word = adverb_to_base(wordnet_tagged[0])
    else:
        base_word = lemma.lemmatize(
            wordnet_tagged[0],
            wordnet_tagged[1]
        )
    
    return base_word

def lemmatize_sentence(input_text):
    ''' Lemmatization process in a sentence
    
    The NLTK library is used on this function.

    Parameters
    ----------
    input_text : string
        description -> A complete sentence to lemmatize cleaned and in lower case
        format -> no apply
        options -> no apply
    
    Returns
    -------
    output_text : string
        description -> Lemmatize sentence
        format -> no apply
        options -> no apply
    '''
    #Pass the tokenize sentence and find the POS tag for each token
    words = nltk.word_tokenize(input_text)
    nltk_tagged = nltk.pos_tag(words)

    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(
        lambda tag: (tag[0], nltk_tag_to_wordnet_tag(tag[1])),
        nltk_tagged
    )

    # Lemmatize sentence by tag
    output_text = list(map(lemmatize_word, wordnet_tagged))
    output_text = ' '.join(output_text)

    return output_text
tweets_df = pd.read_csv('../input/twitter-airline-sentiment/Tweets.csv')
tweets_df.head()
tweets_df.info()
tweets_df.describe(include=np.number)
tweets_df.describe(include=np.object)
for attribute in list((tweets_df.describe(include=np.object)).keys()):
    print('')
    print(f'Analyzing the attribute "{attribute}"')
    print(tweets_df[attribute].value_counts())
    print(f'Actual length of dataframe {len(tweets_df)}')
    print('')
    print('*'*50)
# It is easier to keep the desired columns than drop the columns.
tweets_df = tweets_df[['text', 'airline_sentiment']].copy()

print(f'Shape of filtered data, columns: {tweets_df.shape[1]}')
print(f'Shape of filtered data, rows: {tweets_df.shape[0]}')

# Only to display all the text
pd.set_option('display.max_colwidth', None)
tweets_df.head()
# Get the max text length on the dataset
length_texts = list(map(len, tweets_df["text"]))
print(f'The longest tweet has {max(length_texts)} characters')
fig = plt.figure(figsize=(16,5))
ax = fig.add_subplot(111)
bins = range(10, 195, 5)
plt.hist(length_texts, bins=bins, rwidth= 0.9)  # `density=False` would make counts
label_y = plt.ylabel('Frequency')
label_y.set_color('gray')
label_y.set_size(12)
label_x = plt.xlabel('Tweet characters')
label_x.set_color('gray')
label_x.set_size(12)
plt.xticks(list(bins))
bins_labels(bins, fontsize=10, color='gray')
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
plt.axis()
plt.show();
print_text(tweets_df, step=2000)
fig = plt.figure(figsize=(16,5))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
bins = range(0, 4, 1)
plt.hist(sorted(list(tweets_df['airline_sentiment'])), bins=bins, rwidth= 0.9)
plt.ylabel('Frequency')
plt.xlabel('Sentiment')
plt.xticks(list(bins))
bins_labels(bins, fontsize=9)
ax.set_xticklabels(['negative','neutral','positive'])
title = plt.title('Balance on dataset visualization')
title.set_color('gray')
title.set_size(16)
label_y = plt.ylabel('Frequency')
label_y.set_color('gray')
label_y.set_size(12)
label_x = plt.xlabel('Label')
label_x.set_color('gray')
label_x.set_size(14)
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
plt.show();

plt.show();
tweets_df.insert(
    loc=1,
    column='cleaned_text',
    value=tweets_df['text'].map(text_cleaning)
)

tweets_df.head()
tweets_df['airline_sentiment'].replace(
    {'negative': 0,
     'neutral': 1,
     'positive': 2},
    inplace=True
)

tweets_df.head()
tweets_df.insert(
    loc=2,
    column='stem_text',
    value=tweets_df['cleaned_text'].map(stem_sentence)
)

tweets_df.head()
tweets_df.insert(
    loc=3,
    column='lemma_text',
    value=tweets_df['cleaned_text'].map(lemmatize_sentence)
)

tweets_df.head()
text_train, text_test, label_train, label_test = sk.model_selection.train_test_split(
    tweets_df[['cleaned_text', 'stem_text', 'lemma_text']].copy(),
    tweets_df[['airline_sentiment']].copy(),
    train_size=0.7,
    random_state=42 # To allow reproducible results
)

text_test, text_validation, label_test, label_validation = sk.model_selection.train_test_split(
    text_test.copy(),
    label_test.copy(),
    train_size=0.5,
    random_state=42 # To allow reproducible results
)
# Get a list of the number of words per sentence on all the training dataset
length_words = [
    item \
    for header in text_train.keys() \
    for item in list(map(lambda sentence: len(sentence.split()), text_train[header]))
]

print(f'Max number of words per sentence: {max(length_words)}')
print(f'Most frequent number of words per sentence: {max(set(length_words), key = length_words.count)}')
# Plot the distribution of words on the sentences
fig = plt.figure(figsize=(16,5))
ax = fig.add_subplot(111)
bins = range(0, max(length_words) + 2, 1)
plt.hist(length_words, bins=bins, rwidth= 0.9)
title = plt.title('Training dataset')
title.set_color('gray')
title.set_size(16)
label_y = plt.ylabel('Frequency')
label_y.set_color('gray')
label_y.set_size(12)
label_x = plt.xlabel('Number of words in the tweet text')
label_x.set_color('gray')
label_x.set_size(12)
plt.xticks(list(bins))
bins_labels(bins, fontsize=10, color='gray')
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
plt.axis()
plt.show();
maxlen = 30
words_list_per_column = [list(text_train[header]) for header in text_train.keys()]
words_list_per_column = [list(map(lambda sentence: sentence.split(), set_)) for set_ in words_list_per_column]
words_list_per_column = list(map(lambda set_: [word for words in set_ for word in words] , words_list_per_column))
fig = plt.figure(figsize=(25,10))

for index, words in enumerate(words_list_per_column, 1):
    vocabulary = collections.Counter(words)
    vocabulary = dict(vocabulary.most_common(50))
    labels, values = zip(*vocabulary.items())
    indSort = np.argsort(values)[::-1]
    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]
    indexes = np.arange(len(labels))
    bar_width = 0.1

    ax = plt.subplot(1, 3, index)
    plt.barh(indexes, values, align='center')
    plt.yticks(indexes + bar_width, labels)
    title = plt.title(f'Training dataset "{text_train.keys()[index - 1]}"')
    title.set_color('gray')
    title.set_size(16)
    label_y = plt.ylabel('Most common words')
    label_y.set_color('gray')
    label_y.set_size(12)
    label_x = plt.xlabel('Frequency')
    label_x.set_color('gray')
    label_x.set_size(12)
    ax.tick_params(axis='x', colors='gray')
    ax.tick_params(axis='y', colors='gray')
    ax.invert_yaxis()

plt.show();
# Get a dictionary with first and last index appearance by frequency words
max_words = 0
for index, words in enumerate(words_list_per_column):
    vocabulary = collections.Counter(words)
    frequencies = [word_counter[1] for word_counter in vocabulary.most_common()]
    unique_values = list(set(frequencies))
    unique_values.sort()
    index_by_value = {
        value: [frequencies.index(value), len(frequencies) - frequencies[::-1].index(value) - 1] \
        for value in unique_values
    }

    print(f'First and last index by frequency words in "{text_train.keys()[index]}":\n{dict(list(index_by_value.items())[0:10])}\n')
    max_words += index_by_value[1][0]

max_words = int(max_words/3)
print(f'Max number of words for embedding layer: {max_words}')
# Convert labels dataframe to numpy array
label_train = np.array(label_train['airline_sentiment'])
label_validation = np.array(label_validation['airline_sentiment'])
label_test = np.array(label_test['airline_sentiment'])
results = {}
maxlen = 20
for probe_num in range(3):
    # Past to array a selected type text
    header = text_train.keys()[probe_num]
    text_type_train = np.array(text_train[header].copy())
    text_type_validation = np.array(text_validation[header].copy())
    text_type_test = np.array(text_test[header].copy())

    # Word tokenization
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000, lower=False)
    tokenizer.fit_on_texts(text_type_train)

    text_type_train = tokenizer.texts_to_sequences(text_type_train)
    text_type_validation = tokenizer.texts_to_sequences(text_type_validation)
    text_type_test = tokenizer.texts_to_sequences(text_type_test)

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    text_type_train = keras.preprocessing.sequence.pad_sequences(text_type_train, padding='post', maxlen=maxlen)
    text_type_validation = keras.preprocessing.sequence.pad_sequences(text_type_validation, padding='post', maxlen=maxlen)
    text_type_test = keras.preprocessing.sequence.pad_sequences(text_type_test, padding='post', maxlen=maxlen)

    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 50, input_length=maxlen))
    model.add(keras.layers.Conv1D(64, 3, activation='relu'))
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    history = model.fit(
        text_type_train,
        label_train,
        epochs=10,
        batch_size=128,
        use_multiprocessing=True,
        validation_data=(text_type_validation, label_validation),
        verbose=0
    )
    
    results[header] = sum(history.history['val_accuracy'])/len(history.history['val_accuracy'])

print(results)
text_train_static = text_train.copy()
text_validation_static = text_validation.copy()
text_test_static = text_test.copy()
text_train = np.array(text_train_static['stem_text'].copy())
text_validation = np.array(text_validation_static['stem_text'].copy())
text_test = np.array(text_test_static['stem_text'].copy())
max_words=1000
max_len=30

# Word tokenization only in the training set
tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words, lower=False)
tokenizer.fit_on_texts(text_train)

text_train = tokenizer.texts_to_sequences(text_train)
text_validation = tokenizer.texts_to_sequences(text_validation)
text_test = tokenizer.texts_to_sequences(text_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# "maxlen" set to 30 based on numbers of words on histogream per sentence (padding the sentences)
text_train = keras.preprocessing.sequence.pad_sequences(text_train, padding='post', maxlen=maxlen)
text_validation = keras.preprocessing.sequence.pad_sequences(text_validation, padding='post', maxlen=maxlen)
text_test = keras.preprocessing.sequence.pad_sequences(text_test, padding='post', maxlen=maxlen)
model = keras.models.Sequential()
model.add(keras.layers.Embedding(vocab_size, 300, input_length=maxlen))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Conv1D(256, 7, activation='relu', padding='same'))
model.add(keras.layers.GlobalMaxPooling1D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.RMSprop(
        learning_rate=0.0001,
        rho=0.9,
        centered=False
    ),
    metrics=['accuracy']
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    min_delta=0.0005,
    cooldown=0,
    min_lr=1e-6
)
class_weights = sk.utils.class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(label_train),
    y=label_train
)

class_weights = {label : class_weights[label] for label in range(3)}
history = model.fit(
    text_train,
    label_train,
    epochs=50,
    batch_size=16,
    use_multiprocessing=True,
    #class_weight=class_weights,
    validation_data=(text_validation, label_validation),
    callbacks=[early_stopping, reduce_lr]
)
ax = pd.DataFrame(history.history).plot(figsize=(12,6))
plt.grid(True)
plt.gca().set_ylim(0.3, 1.0)
label_y = plt.ylabel('Score')
label_y.set_color('gray')
label_y.set_size(12)
label_x = plt.xlabel('Epoch')
label_x.set_color('gray')
label_x.set_size(12)
ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')
model.evaluate(text_validation, label_validation)
model.evaluate(text_test, label_test)
prediction_test = model.predict(text_test)

# Confusion matrix
evaluation_cm = confusion_matrix(label_test, prediction_test.argmax(axis=1), binary=False)
fig, ax = plot_confusion_matrix(
    conf_mat=evaluation_cm,
    figsize=(6,6),
    class_names=['negative', 'neutral', 'positive'], 
)
label_y = plt.ylabel('true label')
label_y.set_color('gray')
label_y.set_size(12)
label_x = plt.xlabel('predicted label')
label_x.set_color('gray')
label_x.set_size(12)

ax.tick_params(axis='x', colors='gray')
ax.tick_params(axis='y', colors='gray')

plt.show()
print(f'Accuracy: {sk.metrics.accuracy_score(label_test, prediction_test.argmax(axis=1))}')
print(f'Precision: {sk.metrics.precision_score(label_test, prediction_test.argmax(axis=1), average="weighted")}')
print(f'Recall: {sk.metrics.recall_score(label_test, prediction_test.argmax(axis=1), average="weighted")}')
print(f'F1 Score: {sk.metrics.f1_score(label_test, prediction_test.argmax(axis=1), average="weighted")}')