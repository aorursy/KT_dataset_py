# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import ast
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/movies_metadata.csv")

data = data[data['adult']=='False']
data = data[~data['overview'].isna()]


data.columns
# which rows actually have genres and not a blank list
has_genres_mask = data['genres'] != '[]'

genres = data['genres'][has_genres_mask]
genres.head()
def to_labels(genres_list):
    genres_list = ast.literal_eval(genres_list)
    return [g['name'] for g in genres_list]
genres_strings = genres.apply(to_labels)
genres_strings.head()
from sklearn.preprocessing import MultiLabelBinarizer
labeler = MultiLabelBinarizer()
labeler.fit(genres_strings)
labeler.classes_

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifierCV

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt
X = data['overview'][has_genres_mask]
y = labeler.transform(genres_strings)

X.shape, y.shape
pd.DataFrame(y, columns=labeler.classes_).corr()
plt.imshow(pd.DataFrame(y).corr())
top = sorted(list(zip(y.sum(axis=0), labeler.classes_)))[::-1]
top
top_genres =sorted([t[1] for t in top][1:10])
top_genres
top_labeler = MultiLabelBinarizer(classes=top_genres)
top_labeler.fit(genres_strings)
top_labeler.transform([['this is a' ,'Drama']])
y = top_labeler.transform(genres_strings)
len(y.sum(axis=1)!=0), sum(y.sum(axis=1)!=0)
no_labels_mask = y.sum(axis=1)==0
sum(no_labels_mask),len(no_labels_mask)
X_train, X_test, y_train, y_test = train_test_split(
    X[~no_labels_mask], y[~no_labels_mask], test_size=0.33, random_state=42)
counter = TfidfVectorizer(stop_words='english', lowercase=True, max_features=1000)

pipe = Pipeline(
    [
        ('text_transform', counter),
        ('predictor', MLPClassifier(warm_start=True, max_iter=5, hidden_layer_sizes=(10)))
#         ('predictor', RandomForestClassifier(class_weight='balanced'))
    ])
for i in range(200):
    pipe.fit(X_train, y_train)
    print('epoc {0}, train {1:.3f}, test {2:.3f}'.format(i, 
                                                         pipe.score(X_train, y_train),
                                                         pipe.score(X_test, y_test)))
    
    
# pipe.fit(X_train, y_train)
pipe.score(X_train, y_train)
pipe.score(X_test, y_test)
# get genures of Jumaji
def get_geures(text):
    return top_labeler.classes_[pipe.predict([text]).ravel().astype('bool')]

get_geures('Four high school kids discover an old video game console and are drawn into the game jungle setting, literally becoming the adult avatars they chose. What they discover is that you just play Jumanji - you must survive it. To beat the game and return to the real world, they\'ll have to go on the most dangerous adventure of their lives, discover what Alan Parrish left 20 years ago, and change the way they think about themselves - or they\'ll be stuck in the game forever.')
y.sum(axis=0)
get_geures('Ford Brody (Aaron Taylor-Johnson), a Navy bomb expert, has just reunited with his family in San Francisco when he is forced to go to Japan to help his estranged father, Joe (Bryan Cranston). Soon, both men are swept up in an escalating crisis when Godzilla, King of the Monsters, arises from the sea to combat malevolent adversaries that threaten the survival of humanity. The creatures leave colossal destruction in their wake, as they make their way toward their final battleground: San Francisco.')
get_geures('Captain John Miller (Tom Hanks) takes his men behind enemy lines to find Private James Ryan, whose three brothers have been killed in combat. Surrounded by the brutal realties of war, while searching for Ryan, each man embarks upon a personal journey and discovers their own strength to triumph over an uncertain future with honor, decency and courage.')
get_geures('Shaun (Simon Pegg) is a 30-something loser with a dull, easy existence. When he\'s not working at the electronics store, he lives with his slovenly best friend, Ed (Nick Frost), in a small flat on the outskirts of London. The only unpredictable element in his life is his girlfriend, Liz (Kate Ashfield), who wishes desperately for Shaun to grow up and be a man. When the town is inexplicably overrun with zombies, Shaun must rise to the occasion and protect both Liz and his mother (Penelope Wilton).')
from sklearn.externals import joblib
joblib.dump(pipe, 'movie_geure_predictor.joblib')
print(os.listdir("../working/"))
