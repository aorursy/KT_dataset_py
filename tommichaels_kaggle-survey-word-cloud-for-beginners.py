from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as image 

import pandas as pd

import re

from wordcloud import WordCloud, STOPWORDS

from IPython.display import Image as im
kaggle = pd.read_csv('../input/kaggle-survey-2017/freeformResponses.csv')

kaggle.head()
titles = kaggle.CurrentJobTitleFreeForm.tolist()

titles[0:10]
clean_titles = [t for t in titles if t != "nan"]

clean_titles[0:10]
type(clean_titles[0])
clean_title_str = [str(t) for t in titles]

type(clean_title_str[0])
im(filename="drake.jpg")
complete_title_str = [t for t in clean_title_str if t != "nan"]
complete_title_str[10:20]
zstring = ' '.join(str(e) for e in complete_title_str)

print(zstring[750:900])
non_alphabet = re.compile('[^A-Za-z ]+')

zstring = re.sub(non_alphabet, ' ', zstring)

print(zstring[90:250])
zkeys = zstring.split()

zkeys[0:10]
zkeys = [w for w in zkeys if len(w) > 3]

zkeys = [w.lower() for w in zkeys]

zkeys = ' '.join(zkeys)
zkeys[0:50]
pi_mask = np.array(Image.open('../input/pisymbol/pisymbol.jpeg'))
im(filename="pisymbol.jpeg")
wc = WordCloud(background_color="white", max_words=3000, mask=pi_mask,

               stopwords=STOPWORDS)



wc.generate(zkeys)
wc.to_file("kaggle.png")
im(filename="kaggle.png")