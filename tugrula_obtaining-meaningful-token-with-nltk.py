# Import Pandas

import pandas as pd

# Import Counter

from collections import Counter

# Import WordNetLemmatizer

from nltk.stem import WordNetLemmatizer

# Import word_tokenize

from nltk.tokenize import word_tokenize

# Import stopwords

from nltk.corpus import stopwords

# Import pyplot

import matplotlib.pyplot as plt

# Import string

import string





# Import randint just for test

from random import randint



# Main image size

plt.rcParams["figure.figsize"] = (18, 9)
df_text = pd.read_csv(r'../input/training_text', sep='\|\|', engine='python', skiprows=1, names=['ID', 'Text']).set_index('ID')
# Tokenize the article: tokens

tokens = word_tokenize(str(df_text.iloc[randint(0, len(df_text.index))].values))

# Take the 15 most common tokens

only_token_all=sorted(Counter(tokens).most_common(15), key=lambda w: w[1], reverse=True)
fig1, ax = plt.subplots()

ax.bar(range(len(only_token_all)), [t[1] for t in only_token_all]  , align="center")

ax.set_xticks(range(len(only_token_all)))

ax.set_xticklabels([t[0] for t in only_token_all])

plt.xlabel('Tokens')

plt.ylabel('Number of Usage')

plt.title(r'$\mathrm{Common\ Tokens\ in\ TEXT:}\ Applied\ Tokenize$')

plt.grid(True)

plt.show()
# Convert the tokens into lowercase: lower_tokens

lower_tokens = [t.lower() for t in tokens]

# and delete punctuation

lower_tokens = [''.join(c for c in s if c not in string.punctuation) for s in lower_tokens]

lower_tokens = [s for s in lower_tokens if s]

# Retain alphanumeric: alpha_only

alpha_only = [t for t in lower_tokens if not t.isdigit()]

# Again take the 15 most common tokens

alpha_only_all=sorted(Counter(alpha_only).most_common(15), key=lambda w: w[1], reverse=True)
fig2, ax = plt.subplots()

ax.bar(range(len(alpha_only_all)), [t[1] for t in alpha_only_all]  , align="center")

ax.set_xticks(range(len(alpha_only_all)))

ax.set_xticklabels([t[0] for t in alpha_only_all])

plt.xlabel('Tokens')

plt.ylabel('Number of Usage')

plt.title(r'$\mathrm{Common\ Tokens\ in\ TEXT:}\ Applied\ Tokenize&Alpha$')

plt.grid(True)

plt.show()
# Remove all stop words: no_stops

stop = set(stopwords.words('english'))

no_stops = [t for t in alpha_only if t not in stop]

# Again take the 15 most common tokens

no_stops_all=sorted(Counter(no_stops).most_common(15), key=lambda w: w[1], reverse=True)
fig3, ax = plt.subplots()

ax.bar(range(len(no_stops_all)), [t[1] for t in no_stops_all]  , align="center")

ax.set_xticks(range(len(no_stops_all)))

ax.set_xticklabels([t[0] for t in no_stops_all])

plt.xlabel('Tokens')

plt.ylabel('Number of Usage')

plt.title(r'$\mathrm{Common\ Tokens\ in\ TEXT:}\ Applied\ Tokenize&Alpha&Stop$')

plt.grid(True)

plt.show()
# Instantiate the WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

# Lemmatize all tokens into a new list: lemmatized

lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

# Append the 15 most common tokens for lemmatizer

lemmatized_all=sorted(Counter(lemmatized).most_common(15), key=lambda w: w[1], reverse=True)
fig4, ax = plt.subplots()

ax.bar(range(len(lemmatized_all)), [t[1] for t in lemmatized_all]  , align="center")

ax.set_xticks(range(len(lemmatized_all)))

ax.set_xticklabels([t[0] for t in lemmatized_all])

plt.xlabel('Tokens')

plt.ylabel('Number of Usage')

plt.title(r'$\mathrm{Common\ Tokens\ in\ TEXT:}\ Applied\ Tokenize&Alpha&Stop&Lemmatized$')

plt.grid(True)

plt.show()
# Remove all stop words with updated data

stop.update(['study', 'table', 'method', 'conclusion', 'case', 'data', 'syndrome', 'analyze', 'author', 'show', 'control', 'expression', 'supplementary', 'result', 'figure','fig', 'level', 'deletion', 'mm', 'state', 'effect', 'stability', 'activity','change','structure', 'line', 'loss', 'expression', 'et', 'al'])

no_stops_updated = [t for t in lemmatized if t not in stop]

# Append the 15 most common tokens for no_stop

no_stops_updated_all = sorted(Counter(no_stops_updated).most_common(15), key=lambda w: w[1], reverse=True)
fig5, ax = plt.subplots()

ax.bar(range(len(no_stops_updated_all)), [t[1] for t in no_stops_updated_all]  , align="center")

ax.set_xticks(range(len(no_stops_updated_all)))

ax.set_xticklabels([t[0] for t in no_stops_updated_all])

plt.xlabel('Tokens')

plt.ylabel('Number of Usage')

plt.title(r'$\mathrm{Common\ Tokens\ in\ TEXT:}\ Applied\ Tokenize&Alpha&Updated-Stop&Lemmatized$')

plt.grid(True)

plt.show()