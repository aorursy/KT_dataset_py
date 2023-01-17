# Python program to generate WordCloud 

  

# importing all necessery modules 

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 

import pandas as pd 

import nltk 
# Open function to open the file "pm_narendrs_modi_unspeech_jul20.txt"  



nm_un_speech = open("/kaggle/input/pm_narendrs_modi_unspeech_jul20.txt")

speech1 = nm_un_speech.read()



dl_np_speech = open("/kaggle/input/hh_dalai_lama_nobel_speech.txt")

speech2 = dl_np_speech.read()
# Generate a word cloud object and plot it on the x and y axis

wordcloud = WordCloud().generate(speech1)

 

plt.imshow(wordcloud)

 

#Turn off the axis. Otherwise you will see a bunch of extra numbers around the word cloud

plt.axis("off")

 

#Show the word cloud

plt.show()
# Generate a word cloud object and plot it on the x and y axis

wordcloud = WordCloud().generate(speech2)

 

plt.imshow(wordcloud)

 

#Turn off the axis. Otherwise you will see a bunch of extra numbers around the word cloud

plt.axis("off")

 

#Show the word cloud

plt.show()