# Import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm
import seaborn as sns
# Import training and test data from Quora questions dataset
train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')

# Import list of bad words (banned by Google)
bad_words = pd.read_csv('../input/bad-words/bad_words.csv', header=None)
# Sample of bad words - some are actually not swear words as such
bad_words.sample(5)
# Returns a list (possibly empty) of bad words in text
def detect_badwords(text):
    # tokenize the text
    tokens = word_tokenize(text)
    bad_found = []
    for word in tokens:
        for bad_word in bad_words[0]:
            if bad_word == word:
                bad_found.append(word)
    return bad_found
# Simple examples
s1 = 'This politician speaks nothing but bullshit, what a twat'
s2 = 'The sun shines brightly today'
[detect_badwords(s) for s in [s1,s2]]
# Register `pandas.progress_apply' with `tqdm`
tqdm.pandas()
# Run detect_badwords over all entries in the training data
temp_size = 10000
train_bad_words = train['question_text'].progress_apply(detect_badwords)
# Add as a column to the DataFrame
train['bad_words'] = train_bad_words
# Collect all entries that contain bad words
bools_bw_nonempty = [train['bad_words'].iloc[i] != [] for i in tqdm(range(train.shape[0]))]
df_bw = train[bools_bw_nonempty]

# Sincere set
df_bw_sincere = df_bw[df_bw['target'] == 0]
# Insincere set
df_bw_insincere = df_bw[df_bw['target'] ==1 ]

df_bw_sincere.sample(5)
df_bw_insincere.sample(5)
# Total number of entries that use at least one bad word
bw_total_sincere = len(df_bw_sincere)
bw_total_insincere = len(df_bw_insincere)

print('Total number of sincere posts that contain at least one bad word:', bw_total_sincere)
print('Total number of insincere posts that contain at least one bad word:', bw_total_insincere)

# Total number of sincere and insincere posts
num_sincere = len(train[ train['target'] == 0])
num_insincere = len(train[ train['target'] == 1])
# Proportions
bw_ratio_sincere = bw_total_sincere/num_sincere
bw_ratio_insincere = bw_total_insincere/num_insincere

# Plot
ratio_data = pd.DataFrame(
    {'Flag': ['Sincere', 'Insincere'],
     'Proportion': [bw_ratio_sincere, bw_ratio_insincere]}
)
ratio_data
# Plot of proportion of questions that contain bad words
sns.barplot(x='Flag', 
            y='Proportion',
            data=ratio_data).set_title('Proportion of entries with at least one bad word');
print('''From the training data, insincere posts are about
{:.1f} times more likely to contain bad words'''.format(bw_ratio_insincere/bw_ratio_sincere))
# Define a function to flatten a list of lists
def flatten(l):
    x = []
    for sublist in l:
        for element in sublist:
            x.append(element)
    return x

# import collections library which has functions to compute frequency of elements
import collections
# Function to take in a DataFrame of Quora entries and output a DataFrame of swear words and frequency count
def count_frequency(df):
    # put all the used bad words into a list
    bw_list = flatten(df['bad_words'].tolist())
    # count the frequency of each bad word
    counter = collections.Counter(bw_list)
    # re-order in terms of frequency
    counter = counter.most_common()
    # put into a DataFrame
    df_out = pd.DataFrame(counter, columns = ['Word', 'Frequency'])
    # output
    return df_out

freq_sincere = count_frequency(df_bw_sincere)
freq_insincere = count_frequency(df_bw_insincere)
freq_sincere.head()
# Add a column for frequency normalised by the total number of questions in the set
num_sincere = len(train[ train['target'] == 0])
num_insincere = len(train[ train['target'] == 1])
print('Number of sincere questions:', num_sincere)
print('Number of insincere questions:', num_insincere)

freq_sincere['Normalised Frequency'] = freq_sincere['Frequency']/num_sincere
freq_insincere['Normalised Frequency'] = freq_insincere['Frequency']/num_insincere
freq_insincere.head()
# Bar plot: 10 most frequent bad words in sincere questions
sns.barplot(x='Word',
            y='Normalised Frequency',
            data=freq_sincere.iloc[:10]
           ).set_title('Most frequent \'bad\' words in Quora sincere questions');
# Bar plot: 10 most frequent bad words in insincere questions
sns.barplot(x='Word',
            y='Normalised Frequency',
            data=freq_insincere.iloc[:10]
           ).set_title('Most frequent \'bad\' words in Quora insincere questions');
# Create DataFrame with bad-word frequencies in sincere and insincere sets
temp = freq_insincere.set_index('Word')['Normalised Frequency'].rename('Frequency in insincere posts')
temp2 = freq_sincere.set_index('Word')['Normalised Frequency'].rename('Frequency in sincere posts')
df_scat = pd.concat([temp,temp2], axis=1).fillna(0)
df_scat.head()
np.arange(1,10,1)
# Scatter plot and line y=x
ax = sns.scatterplot(x='Frequency in sincere posts', 
                y='Frequency in insincere posts', 
                data=df_scat)
ax.set_xlim(left=0, right=0.002)
ax.set_ylim(bottom=0, top=0.02)
ax.set_title('''Scatter plot of bad words against their 
normalised frequencies in the Quora posts''');
sns.lineplot(np.linspace(0,0.002,100), 2*np.linspace(0,0.002,100),
            color='coral')
# Extract words below the orange line
bools = df_scat['Frequency in insincere posts'] < df_scat['Frequency in sincere posts']
bw_remove = df_scat[bools].index.values.tolist()
# Sample of these words
len(bw_remove)
# Remove any instances in train of bad words in bw_remove
train_dropbw = train
for i in tqdm(range(len(train))):
    entry = train_dropbw['bad_words'].iloc[i]
    if entry != []:    
        entry = [x for x in entry if x not in bw_remove]
        train_dropbw['bad_words'].iloc[i] = entry
               
train_dropbw.head()
# Collect all entries that contain bad words
bools_bw_nonempty = [train_dropbw['bad_words'].iloc[i] != [] for i in tqdm(range(train.shape[0]))]
df_bw = train_dropbw[bools_bw_nonempty]

# Sincere set
df_bw_sincere = df_bw[df_bw['target'] == 0]
# Insincere set
df_bw_insincere = df_bw[df_bw['target'] ==1 ]
bw_total_sincere = len(df_bw_sincere)
bw_total_insincere = len(df_bw_insincere)
# Proportions
bw_ratio_sincere = bw_total_sincere/num_sincere
bw_ratio_insincere = bw_total_insincere/num_insincere

# Plot
ratio_data = pd.DataFrame(
    {'Flag': ['Sincere', 'Insincere'],
     'Proportion': [bw_ratio_sincere, bw_ratio_insincere]}
)
ratio_data
# Plot of proportion of questions that contain bad words
sns.barplot(x='Flag', 
            y='Proportion',
            data=ratio_data).set_title('Proportion of entries with at least one bad word');
print('''From the training data, insincere posts are about
{:.1f} times more likely to contain bad words'''.format(bw_ratio_insincere/bw_ratio_sincere))
