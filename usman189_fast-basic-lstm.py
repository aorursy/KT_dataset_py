import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import textblob



print("Loading data...")

train = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t")

print("Train shape:", train.shape)

test = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t")

print("Test shape:", test.shape)



enc = OneHotEncoder(sparse=False)

enc.fit(train["Sentiment"].values.reshape(-1, 1))

print("Number of classes:", enc.n_values_[0])



print("Class distribution:\n{}".format(train["Sentiment"].value_counts()/train.shape[0]))
from sklearn.feature_extraction.text import CountVectorizer



cv1 = CountVectorizer()

cv1.fit(train["Phrase"])



cv2 = CountVectorizer()

cv2.fit(test["Phrase"])



print("Train Set Vocabulary Size:", len(cv1.vocabulary_))

print("Test Set Vocabulary Size:", len(cv2.vocabulary_))

print("Number of Words that occur in both:", len(set(cv1.vocabulary_.keys()).intersection(set(cv2.vocabulary_.keys()))))
def transform(df):

    df["phrase_count"] = df.groupby("SentenceId")["Phrase"].transform("count")

    df["word_count"] = df["Phrase"].apply(lambda x: len(x.split()))

    df["has_upper"] = df["Phrase"].apply(lambda x: x.lower() != x)

    df["sentence_end"] = df["Phrase"].apply(lambda x: x.endswith("."))

    df["after_comma"] = df["Phrase"].apply(lambda x: x.startswith(","))

    df["sentence_start"] = df["Phrase"].apply(lambda x: "A" <= x[0] <= "Z")

    df["Phrase"] = df["Phrase"].apply(lambda x: x.lower())

    return df



train = transform(train)

test = transform(test)



dense_features = ["phrase_count", "word_count", "has_upper", "after_comma", "sentence_start", "sentence_end"]



train.groupby("Sentiment")[dense_features].mean()
NUM_FOLDS = 2



train["fold_id"] = train["SentenceId"].apply(lambda x: x%NUM_FOLDS)
EMBEDDING_FILE = "../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt"

EMBEDDING_DIM = 200



all_words = set(cv1.vocabulary_.keys()).union(set(cv2.vocabulary_.keys()))



def get_embedding():

    embeddings_index = {}

    f = open(EMBEDDING_FILE)

    for line in f:

        values = line.split()

        word = values[0]

        if len(values) == EMBEDDING_DIM + 1 and word in all_words:

            coefs = np.asarray(values[1:], dtype="float32")

            embeddings_index[word] = coefs

    f.close()

    return embeddings_index



embeddings_index = get_embedding()

print("Number of words that don't exist in GLOVE:", len(all_words - set(embeddings_index)))
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



MAX_SEQUENCE_LENGTH = 60



tokenizer = Tokenizer()

tokenizer.fit_on_texts(np.append(train["Phrase"].values, test["Phrase"].values))

word_index = tokenizer.word_index



nb_words = len(word_index) + 1

embedding_matrix = np.random.rand(nb_words, EMBEDDING_DIM + 2)



for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    sent = textblob.TextBlob(word).sentiment

    if embedding_vector is not None:

        embedding_matrix[i] = np.append(embedding_vector, [sent.polarity, sent.subjectivity])

    else:

        embedding_matrix[i, -2:] = [sent.polarity, sent.subjectivity]

        

seq = pad_sequences(tokenizer.texts_to_sequences(train["Phrase"]), maxlen=MAX_SEQUENCE_LENGTH)

test_seq = pad_sequences(tokenizer.texts_to_sequences(test["Phrase"]), maxlen=MAX_SEQUENCE_LENGTH)
from keras.layers import *

from keras.models import Model

from keras.callbacks import EarlyStopping



def build_model():

    embedding_layer = Embedding(nb_words,

                                EMBEDDING_DIM + 2,

                                weights=[embedding_matrix],

                                input_length=MAX_SEQUENCE_LENGTH,

                                trainable=True)

    dropout = SpatialDropout1D(0.2)

    mask_layer = Masking()

    lstm_layer = LSTM(50)

    

    seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")

    dense_input = Input(shape=(len(dense_features),))

    

    dense_vector = BatchNormalization()(dense_input)

    

    phrase_vector = lstm_layer(mask_layer(dropout(embedding_layer(seq_input))))

    

    feature_vector = concatenate([phrase_vector, dense_vector])

    feature_vector = Dense(50, activation="relu")(feature_vector)

    feature_vector = Dense(20, activation="relu")(feature_vector)

    

    output = Dense(5, activation="softmax")(feature_vector)

    

    model = Model(inputs=[seq_input, dense_input], outputs=output)

    return model
test_preds = np.zeros((test.shape[0], 5))



for i in range(NUM_FOLDS):

    print("FOLD", i+1)

    

    print("Splitting the data into train and validation...")

    train_seq, val_seq = seq[train["fold_id"] != i], seq[train["fold_id"] == i]

    train_dense, val_dense = train[train["fold_id"] != i][dense_features], train[train["fold_id"] == i][dense_features]

    y_train = enc.transform(train[train["fold_id"] != i]["Sentiment"].values.reshape(-1, 1))

    y_val = enc.transform(train[train["fold_id"] == i]["Sentiment"].values.reshape(-1, 1))

    

    print("Building the model...")

    model = build_model()

    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["acc"])

    

    early_stopping = EarlyStopping(monitor="val_acc", patience=10, verbose=1)

    

    print("Training the model...")

    model.fit([train_seq, train_dense], y_train, validation_data=([val_seq, val_dense], y_val),

              epochs=50, batch_size=2048, shuffle=True, callbacks=[early_stopping], verbose=1)

    

    print("Predicting...")

    test_preds += model.predict([test_seq, test[dense_features]], batch_size=1024, verbose=1)

    print()

    

test_preds /= NUM_FOLDS


print("Ratio of test set examples which occur in the train set: {0:.2f}".format(len(set(train["Phrase"]).intersection(set(test["Phrase"])))/test.shape[0]))

test = pd.merge(test, train[["Phrase", "Sentiment"]], on="Phrase", how="left")

print (test.columns)
print("Select the class with the highest probability as prediction...")

test["pred"] = test_preds.argmax(axis=1)



print("Use these predictions for the phrases which don't exist in train set...")

test.loc[test["Sentiment"].isnull(), "Sentiment"] = test.loc[test["Sentiment"].isnull(), "pred"]



print("Make the submission ready...")

test["Sentiment"] = test["Sentiment"].astype(int)

test[["PhraseId", "Sentiment"]].to_csv("submission.csv", index=False)



print (test[["PhraseId", "Sentiment"]])
#submission = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/submission.tsv", sep="\t")

#print (submission)

import os

print(os.listdir("../Output"))



test[["PhraseId", "Sentiment"]].to_csv("submission.csv", index=False)
