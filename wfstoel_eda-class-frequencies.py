import numpy as np

import pandas as pd
DATA_DIR = '/kaggle/input/bengaliai-cv19/'



path_train_data = DATA_DIR + 'train.csv'
df_train = pd.read_csv(path_train_data)

df_train.drop('grapheme', inplace=True, axis=1)

df_train.sample(5)
import seaborn as sns

import matplotlib.pyplot as plt



sns.set()

sns.set_palette(sns.dark_palette('purple'))



bar_palette = lambda n: sns.hls_palette(n, l=.4)
def plot_frequency(data):

    if isinstance(data, str):

        counts = df_train[data].value_counts()

        name = data

        

    else:

        counts = df_train.groupby(data).size().sort_values(ascending=False)

        name = '-'.join(data)



    counts /= len(df_train)

    n = len(counts)

    x = np.arange(n)

    y = counts.cumsum()

    

    plt.figure(figsize=(20, 5))

    if n <= 100:

        plt.subplot(1, 2, 1)

    sns.barplot(x=counts.index, y=counts, order=counts.index, palette=bar_palette(n))

    plt.title('Frequency of each %s class [%d]' % (name, n))

    plt.ylabel('Frequency')

    

    if n > 40:

        plt.xticks([])

    else:

        plt.xlabel('Class label')

    

    if n > 100:

        plt.show()

        plt.figure(figsize=(20, 5))

       

    else:

        plt.subplot(1, 2, 2)

        

    plt.fill_between(x, y, step='post', alpha=0.4)

    plt.step(x, y, where='post')

    plt.ylim(0, 1.05)

    plt.title('Cumulative frequency of each %s class' % name)

    plt.xlabel('Number of classes')

    plt.ylabel('Cumulative frequency')

    plt.show()
plot_frequency('consonant_diacritic')
plot_frequency('vowel_diacritic')
plot_frequency('grapheme_root')
plot_frequency(['consonant_diacritic', 'vowel_diacritic'])
plot_frequency(['consonant_diacritic', 'grapheme_root'])
plot_frequency(['vowel_diacritic', 'grapheme_root'])
plot_frequency(['consonant_diacritic', 'vowel_diacritic', 'grapheme_root'])