# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import graphviz

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import misc

from sklearn import tree

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, export_graphviz







palette = sns.color_palette("Set2")

sns.set_palette(palette)

sns.set_style("white")

%matplotlib inline
data = pd.read_csv("../input/data.csv")

data.head()
sum(data.target)
train, test = train_test_split(data, test_size = 0.3)

print("Training samples:{}; Test samples{}". format(len(train), len(test)))
sum(train.target)

p_tempo = data[data['target'] == 1]['tempo']

n_tempo = data[data['target'] == 0]['tempo']

p_dance = data[data['target'] == 1]['danceability']

n_dance = data[data['target'] == 0]['danceability']

p_duration = data[data['target'] == 1]['duration_ms']

n_duration = data[data['target'] == 0]['duration_ms']

p_loudness = data[data['target'] == 1]['loudness']

n_loudness = data[data['target'] == 0]['loudness']

p_speechiness = data[data['target'] == 1]['speechiness']

n_speechiness = data[data['target'] == 0]['speechiness']

p_valence = data[data['target'] == 1]['valence']

n_valence = data[data['target'] == 0]['valence']

p_energy = data[data['target'] == 1]['energy']

n_energy = data[data['target'] == 0]['energy']

p_acousticness = data[data['target'] == 1]['acousticness']

n_acousticness = data[data['target'] == 0]['acousticness']

p_key = data[data['target'] == 1]['key']

n_key = data[data['target'] == 0]['key']

p_instrumentalness = data[data['target'] == 1]['instrumentalness']

n_instrumentalness = data[data['target'] == 0]['instrumentalness']







fig = plt.figure(figsize=(12, 10))

plt.title("Song Tempo Like/Dislike Distribution")

p_tempo.hist(alpha=0.7, bins=30, label='positive')

n_tempo.hist(alpha=0.7, bins=30, label='negative')

plt.legend(loc="upper right")
fig2 = plt.figure(figsize=(20, 20))



# Danceability

pic3 = fig2.add_subplot(331)

pic3.set_xlabel('Danceability')

pic3.set_ylabel('Count')

pic3.set_title("Danceability Like Distribution")

p_dance.hist(alpha=0.5, bins=30)

pic4 = fig2.add_subplot(331)

n_dance.hist(alpha=0.5, bins=30)





# Duration

pic5 = fig2.add_subplot(332)

p_duration.hist(alpha=0.5, bins=30)

pic5.set_xlabel('Duration (ms)')

pic5.set_ylabel('Count')

pic5.set_title("Duration Like Distribution")

pic6 = fig2.add_subplot(332)

n_duration.hist(alpha=0.5, bins=30)





# Loudness

pic7 = fig2.add_subplot(333)

p_loudness.hist(alpha=0.5, bins=30)

pic7.set_xlabel('Loudness')

pic7.set_ylabel('Count')

pic7.set_title("Loudness Like Distribution")



pic8 = fig2.add_subplot(333)

n_loudness.hist(alpha=0.5, bins=30)



# Speechiness

pic9 = fig2.add_subplot(334)

p_speechiness.hist(alpha=0.5, bins=30)

pic9.set_xlabel('Speechiness')

pic9.set_ylabel('Count')

pic9.set_title("Song Speechiness Like Distribution")



pic10 = fig2.add_subplot(334)

n_speechiness.hist(alpha=0.5, bins=30)



# Valence

pic11 = fig2.add_subplot(335)

p_valence.hist(alpha=0.5, bins=30)

pic11.set_xlabel('Valence')

pic11.set_ylabel('Count')

pic11.set_title("Song Valence Like Distribution")



pic12 = fig2.add_subplot(335)

n_valence.hist(alpha=0.5, bins=30)



# Energy

pic13 = fig2.add_subplot(336)

p_energy.hist(alpha=0.5, bins=30)

pic13.set_xlabel('Energy')

pic13.set_ylabel('Count')

pic13.set_title("Song Energy Like Distribution")



pic14 = fig2.add_subplot(336)

n_energy.hist(alpha=0.5, bins=30)



# Key

pic15 = fig2.add_subplot(337)

p_key.hist(alpha=0.5, bins=30)

pic15.set_xlabel('Key')

pic15.set_ylabel('Count')

pic15.set_title("Song Key Like Distribution")



pic15 = fig2.add_subplot(337)

n_key.hist(alpha=0.5, bins=30)



# Acousticness

pic16 = fig2.add_subplot(338)

p_acousticness.hist(alpha=0.5, bins=30)

pic16.set_xlabel('Acousticness')

pic16.set_ylabel('Count')

pic16.set_title("Song Acousticness Like Distribution")



pic16 = fig2.add_subplot(338)

n_acousticness.hist(alpha=0.5, bins=30)



# Instrumentalness

pic17 = fig2.add_subplot(339)

p_instrumentalness.hist(alpha=0.5, bins=30)

pic17.set_xlabel('Instrumentalness')

pic17.set_ylabel('Count')

pic17.set_title("Song Instrumentalness Like Distribution")



pic17 = fig2.add_subplot(339)

n_instrumentalness.hist(alpha=0.5, bins=30)
c = tree.DecisionTreeClassifier(min_samples_leaf = 50, 

                                random_state =10)

features = ["valence", "energy", "danceability", "speechiness", "acousticness", "instrumentalness", "loudness","duration_ms","liveness","tempo","time_signature","mode","key"]



X_train = train[features]

y_train = train["target"]



X_test = test[features]

y_test = test["target"]



dt = c.fit(X_train, y_train)

y_prediction = c.predict(X_test)
score = accuracy_score(y_test, y_prediction)*100

rounded_score = round(score, 1)

print("Decision Tree Classifier Accuracy: {}". format(rounded_score))

y_tpre = c.predict(X_train)

score = accuracy_score(y_train, y_tpre)*100

rounded_score = round(score, 1)

print("Decision Tree Classifier Accuracy: {}". format(rounded_score))
hidden_neuron_nums = list(range(15,100))

#[2,3,4,5,6...9, 10, 20, 30, ... 90, 100, 125, 150, 175]

total_performance_records = []

for hn in hidden_neuron_nums:

    c_ = tree.DecisionTreeClassifier(min_samples_leaf=hn, random_state=10)

    perf_records_ = []

    for i in range(20):

         c_.fit(X_train, y_train)

         tst_p_ = c_.predict(X_test)

         performance = np.sum(tst_p_ == y_test) / float(tst_p_.size)

         perf_records_.append(performance)

    total_performance_records.append(np.mean(perf_records_))

    print ("Evaluate hidden layer {} done, accuracy {:.3f}".format(

        hn, total_performance_records[-1]))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 120)

clf.fit(X_train, y_train)
forest_y_pred = clf.predict(X_test)

score = accuracy_score(y_test, forest_y_pred) * 100

rounded_score = round(score, 1)

print("Random Forest (n_est: 100) Accuracy: {}%".format(rounded_score))