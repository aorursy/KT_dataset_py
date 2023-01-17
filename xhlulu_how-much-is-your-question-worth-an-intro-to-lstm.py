import os



import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, GlobalMaxPooling1D, LSTM, Bidirectional, Embedding, Dropout

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences
data_df = pd.read_csv('/kaggle/input/200000-jeopardy-questions/JEOPARDY_CSV.csv')

data_df = data_df[data_df[' Value'] != 'None']



print(data_df.shape)

data_df.head()
data_df['ValueNum'] = data_df[' Value'].apply(

    lambda value: int(value.replace(',', '').replace('$', ''))

)
def binning(value):

    if value < 1000:

        return np.round(value, -2)

    elif value < 10000:

        return np.round(value, -3)

    else:

        return np.round(value, -4)



data_df['ValueBins'] = data_df['ValueNum'].apply(binning)
print("Total number of categories:", data_df[' Value'].unique().shape[0])

print("Number of categories after binning:", data_df['ValueBins'].unique().shape[0])

print("\nBinned Categories:", data_df['ValueBins'].unique())
show_numbers = data_df['Show Number'].unique()

train_shows, test_shows = train_test_split(show_numbers, test_size=0.2, random_state=2019)



train_mask = data_df['Show Number'].isin(train_shows)

test_mask = data_df['Show Number'].isin(test_shows)



train_labels = data_df.loc[train_mask, 'ValueBins']

train_questions = data_df.loc[train_mask, ' Question']

test_labels = data_df.loc[test_mask, 'ValueBins']

test_questions = data_df.loc[test_mask, ' Question']
%%time

bow = CountVectorizer(stop_words='english', max_features=2000)

bow.fit(data_df[' Question'])
X_train = bow.transform(train_questions)

X_test = bow.transform(test_questions)



y_train = train_labels

y_test = test_labels



print("Shape of X_train:", X_train.shape)

print("Shape of X_test:", X_test.shape)

print("Shape of y_train:", y_train.shape)

print("Shape of y_test:", y_test.shape)
%%time

lr = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=200)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)



print(classification_report(y_test, y_pred))
tokenizer = Tokenizer(num_words=50000)

tokenizer.fit_on_texts(data_df[' Question'])



train_sequence = tokenizer.texts_to_sequences(train_questions)

test_sequence = tokenizer.texts_to_sequences(test_questions)



print("Original text:", train_questions[0])

print("Converted sequence:", train_sequence[0])
X_train = pad_sequences(train_sequence, maxlen=50)

X_test = pad_sequences(test_sequence, maxlen=50)



print(X_train.shape)

print(X_test.shape)
le = LabelEncoder()

le.fit(data_df['ValueBins'])



y_train = le.transform(train_labels)

y_test = le.transform(test_labels)



print(y_train.shape)

print(y_test.shape)
num_words = tokenizer.num_words

output_size = len(le.classes_)
model = Sequential([

    Embedding(input_dim=num_words, 

              output_dim=200, 

              mask_zero=True, 

              input_length=50),

    Bidirectional(LSTM(150, return_sequences=True)),

    GlobalMaxPooling1D(),

    Dense(300, activation='relu'),

    Dropout(0.5),

    Dense(output_size, activation='softmax')

    

])



model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.summary()
model.fit(X_train, y_train, epochs=10, batch_size=1024, validation_split=0.1)
y_pred = model.predict(X_test, batch_size=1024).argmax(axis=1)

print(classification_report(y_test, y_pred))