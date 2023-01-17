import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from collections import Counter



def count_words_fast(text):

    text = text.lower()

    skips = [".", ",", ";", ":", "'", '"', "\n", "!", "?", "(", ")"]

    for ch in skips:

        text = text.replace(ch, "")

    word_counts = Counter(text.split(" "))

    return word_counts



def word_stats(word_counts):

    num_unique = len(word_counts)

    counts = word_counts.values()

    return (num_unique, counts)
hamlets = pd.read_csv("/kaggle/input/hamlet-english-german-portuguese/hamlets.csv",index_col=0)             

hamlets
language, text = hamlets.iloc[0]



counted_text = count_words_fast(text)



data = pd.DataFrame({

    "word": list(counted_text.keys()),

    "count": list(counted_text.values())

})

data.head(10)
data["length"] = data["word"].apply(len)



data.loc[data["count"] > 10,  "frequency"] = "frequent"

data.loc[data["count"] <= 10, "frequency"] = "infrequent"

data.loc[data["count"] == 1,  "frequency"] = "unique"



data.groupby('frequency').count()
sub_data = pd.DataFrame({

    "language": language,

    "frequency": ["frequent","infrequent","unique"],

    "mean_word_length": data.groupby(by = "frequency")["length"].mean(),

    "num_words": data.groupby(by = "frequency").size()

})



sub_data
def summarize_text(language, text):

    counted_text = count_words_fast(text)



    data = pd.DataFrame({

        "word": list(counted_text.keys()),

        "count": list(counted_text.values())

    })

    

    data.loc[data["count"] > 10,  "frequency"] = "frequent"

    data.loc[data["count"] <= 10, "frequency"] = "infrequent"

    data.loc[data["count"] == 1,  "frequency"] = "unique"

    

    data["length"] = data["word"].apply(len)

    

    sub_data = pd.DataFrame({

        "language": language,

        "frequency": ["frequent","infrequent","unique"],

        "mean_word_length": data.groupby(by = "frequency")["length"].mean(),

        "num_words": data.groupby(by = "frequency").size()

    })

    

    return(sub_data)

    

grouped_data = pd.DataFrame(columns = ["language", "frequency", "mean_word_length", "num_words"])



for i in range(hamlets.shape[0]):

    language, text = hamlets.iloc[i]

    sub_data = summarize_text(language, text)

    grouped_data = grouped_data.append(sub_data)

    

grouped_data
colors = {"Portuguese": "green", "English": "blue", "German": "red"}

markers = {"frequent": "o","infrequent": "s", "unique": "^"}

import matplotlib.pyplot as plt

for i in range(grouped_data.shape[0]):

    row = grouped_data.iloc[i]

    plt.plot(row.mean_word_length, row.num_words,

        marker=markers[row.frequency],

        color = colors[row.language],

        markersize = 10

    )



color_legend = []

marker_legend = []

for color in colors:

    color_legend.append(

        plt.plot([], [],

        color=colors[color],

        marker="o",

        label = color, markersize = 10, linestyle="None")

    )

for marker in markers:

    marker_legend.append(

        plt.plot([], [],

        color="k",

        marker=markers[marker],

        label = marker, markersize = 10, linestyle="None")

    )

plt.legend(numpoints=1, loc = "upper left")



plt.xlabel("Mean Word Length")

plt.ylabel("Number of Words")

plt.show();