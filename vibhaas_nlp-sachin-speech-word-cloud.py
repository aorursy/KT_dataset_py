# Python program to generate WordCloud 

  

# importing all necessery modules 

from wordcloud import WordCloud, STOPWORDS 

import matplotlib.pyplot as plt 

import pandas as pd 

import nltk 
# Open function to open the file "Sachin-Speech.txt"  



speech = open("/kaggle/input/Sachin-Speech.txt")
speech1 = speech.read()
speech1
# Generate a word cloud object and plot it on the x and y axis

wordcloud = WordCloud().generate(speech1)

 

plt.imshow(wordcloud)

 

#Turn off the axis. Otherwise you will see a bunch of extra numbers around the word cloud

plt.axis("off")

 

#Show the word cloud

plt.show()