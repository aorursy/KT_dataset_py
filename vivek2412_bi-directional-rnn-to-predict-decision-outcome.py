# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

%matplotlib inline



from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional

from keras.layers.embeddings import Embedding
df = pd.DataFrame()

df = pd.read_csv("../input/industrial-security-clearance-decisions-classified.csv")

df_noblanks = df.loc[(df["digest"].notnull()), ("date", "digest", "keywords", "Favorable decision", "Decision upheld")]

df_noblanks.dropna(subset = ['Favorable decision', 'Decision upheld'], inplace = True)



df_noblanks.head()
#Identify if any rows have unrecognizable date formats; this was a problem with the original data set but has already been cleaned up in this one

bad_rows = pd.to_datetime(df_noblanks["date"], errors = 'coerce').isnull()

print(np.where(bad_rows == True))



#Group the favorable and unfavorable decisions by year

df_noblanks['date'] = pd.to_datetime(df_noblanks['date'])

x = pd.DataFrame(df_noblanks.groupby([df_noblanks['date'].dt.year, 'Favorable decision'])

                 ['Favorable decision'].agg({'count'}).reset_index())

x_pos = x.loc[x['Favorable decision'] == 'Yes']

x_neg = x.loc[x['Favorable decision'] == 'No']

x_pos.loc[:, 'Pctg'] = np.divide(x_pos.loc[:, 'count'], np.add(x_pos.loc[:, 'count'], x_neg.loc[:, 'count']))



fig = plt.figure(figsize = (15,4))

colors = ['limegreen', 'tomato']



#Line chart for annual distribution of decisions - both favorable and unfavorable

ax1 = fig.add_subplot(121)

ax1.set(title = "Annual distribution of decisions", xlabel = "Year", ylabel = "No. of decisions")

ax1.plot(x_pos['date'], x_pos['count'], color = colors[0], linewidth = 2, label = "Favourable decisions")

ax1.plot(x_neg['date'], x_neg['count'], color = colors[1], linewidth = 2, label = "Unfavourable decisions")

ax1.legend()



#Pie chart indicating the proportion of favorable to unfavorable decisions

ax2 = fig.add_subplot(122)

ax2.set(title = "Percentage Unfavourable vs. Favourable decisions")

ax2.pie((sum(x_pos['count']), sum(x_neg['count'])), explode = (0.1, 0), labels = ("Favourable decisions", "Unfavourable decisions"), autopct='%1.1f%%', shadow = True, colors = colors)

ax2.axis('equal')



#Bar chart indicating the percentage of favorable decisions by year

fig2 = plt.figure(figsize = (15,8))

ax = fig2.add_subplot(111)

ax.set(title = "Percentage of favourable decisions", xlabel = "Year", ylabel = "Percentage favourable decisions")

ax.bar(x_pos['date'], x_pos['Pctg'], color = colors[0], linewidth = 3)



rects = ax.patches

labels = x_pos['Pctg']



for rect, label in zip(rects, labels):

    ax.annotate(

            '{:.1%}'.format(label),                      

            (rect.get_x() + rect.get_width() / 2, rect.get_height()),         

            xytext=(0, 5),          

            textcoords="offset points", 

            ha='center',                

            va='bottom')    
#Group the upheld and overturned decisions by year

x = pd.DataFrame(df_noblanks.groupby([df_noblanks['date'].dt.year, 'Decision upheld'])['Decision upheld'].agg({'count'}).reset_index())

x_pos = x.loc[x['Decision upheld'] == 'Yes']

x_neg = x.loc[x['Decision upheld'] == 'No']

x_pos.loc[:, 'Pctg'] = np.divide(x_pos.loc[:, 'count'], np.add(x_pos.loc[:, 'count'], x_neg.loc[:, 'count']))



fig = plt.figure(figsize = (15,4))

colors = ['springgreen', 'salmon']



#Line chart for annual distribution of decisions - both upheld and overturned

ax1 = fig.add_subplot(121)

ax1.set(title = "Annual distribution of decisions", xlabel = "Year", ylabel = "No. of decisions")

ax1.plot(x_pos['date'], x_pos['count'], color = colors[0], linewidth = 2, label = "Decisions upheld")

ax1.plot(x_neg['date'], x_neg['count'], color = colors[1], linewidth = 2, label = "Decision overturned")

ax1.legend()



#Pie chart indicating the proportion of upheld to overturned decisions

ax2 = fig.add_subplot(122)

ax2.set(title = "Percentage Decisions upheld vs. overturned")

ax2.pie((sum(x_pos['count']), sum(x_neg['count'])), explode = (0.1, 0), labels = ("Decisions upheld", "Decisions overturned"), autopct='%1.1f%%', shadow = True, colors = colors)

ax2.axis('equal')



#Bar chart indicating the percentage of upheld decisions by year

fig2 = plt.figure(figsize = (15,8))

ax = fig2.add_subplot(111)

ax.set(title = "Percentage of decisions upheld", xlabel = "Year", ylabel = "Percentage decisions upheld")

ax.bar(x_pos['date'], x_pos['Pctg'], color = colors[0], linewidth = 3)



rects = ax.patches

labels = x_pos['Pctg']



for rect, label in zip(rects, labels):

    ax.annotate(

            '{:.1%}'.format(label),                      

            (rect.get_x() + rect.get_width() / 2, rect.get_height()),         

            xytext=(0, 5),          

            textcoords="offset points", 

            ha='center',                

            va='bottom')       
#Group upheld and favorable decisions

x = pd.DataFrame(df_noblanks.groupby('Favorable decision')['Decision upheld'].agg({'count'}).reset_index())

x.head()



outcome = pd.Series(df_noblanks['Favorable decision'], name='Favorable decision')

upheld = pd.Series(df_noblanks['Decision upheld'], name='Decision upheld')

df_confusion = pd.crosstab(outcome, upheld)



#Cross tabulation (confusion matrix) of decisions upheld and favorable decisions

fig3 = plt.figure(figsize = (15,4))

ax3 = fig3.add_subplot(111)

ax3.set(title = "Cross tabulation of favourable decisions and decisions upheld", 

        xlabel = 'Decision upheld', ylabel = 'Favorable decision', 

        xticks = np.arange(df_confusion.shape[1]), yticks = np.arange(df_confusion.shape[0]),

        xticklabels = df_confusion.index.values, yticklabels = df_confusion.columns.values)

ax3.imshow(df_confusion, interpolation = 'nearest', cmap = plt.cm.Greens)



thresh = df_confusion.values.max() / 2

for i in range(df_confusion.shape[0]):

    for j in range(df_confusion.shape[1]):

        ax3.text(j, i, df_confusion.iloc[i, j],

                ha="center", va="center",

                color="white" if df_confusion.iloc[i, j] > thresh else "black")

plt.show()

        

num_initial_unfavorable_decisions = df_confusion.iloc[0,1] + df_confusion.iloc[1,0]

print("Number of decisions that were initially ruled upon unfavorably = "

      + str(num_initial_unfavorable_decisions))

print("Probability of an unfavorable decision being overturned in case it is appealed = "

     + "{:.2%}".format(df_confusion.iloc[1,0]/num_initial_unfavorable_decisions))



num_initial_favorable_decisions = df_confusion.iloc[0,0] + df_confusion.iloc[1,1]

print("Number of decisions that were initially ruled upon favorably = "

      + str(num_initial_favorable_decisions))

print("Probability of a favorable decision being overturned in case it is appealed = "

     + "{:.2%}".format(df_confusion.iloc[0,0]/num_initial_favorable_decisions))
#X - Transcript of the decision as input

X = df_noblanks.loc[:, ("digest")]



#Y - Predicting the outcome of the decision and whether or not it will be upheld

Y = df_noblanks.loc[:, ("Favorable decision", "Decision upheld")]

#Code the Yes/No values in Favorable and upheld decisions, our predicted values, to 1/0

Y = (Y.replace(to_replace = ['No', 'Yes'], value = [0, 1])).astype(int)



#Split into training and test data

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=42)
tokenizer_object = Tokenizer()

X_train = X_train.flatten()

X_test = X_test.flatten()

total_decisions = np.concatenate((X_train, X_test))

tokenizer_object.fit_on_texts(total_decisions)



#Pad sequences

max_len = max([len(s.split()) for s in total_decisions])



#Define vocab size

vocab_size = len(tokenizer_object.word_index) + 1



#Create sequences

X_train_tokens = tokenizer_object.texts_to_sequences(X_train)

X_test_tokens = tokenizer_object.texts_to_sequences(X_test)



X_train_pad = pad_sequences(X_train_tokens, maxlen = max_len, padding = 'post')

X_test_pad = pad_sequences(X_test_tokens, maxlen = max_len, padding = 'post')





#Bi-directional RNN with embeddings being trained

EMBEDDING_DIM = 50

model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length = max_len))

model.add(Bidirectional(GRU(units = 32, dropout = 0.15, recurrent_dropout = 0.15)))

model.add(Dense(2, activation = 'sigmoid'))



model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X_train_pad, Y_train, batch_size = 128, epochs = 10, validation_data = (X_test_pad, Y_test), verbose = 2)



#Plotting  costs by iteration

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()