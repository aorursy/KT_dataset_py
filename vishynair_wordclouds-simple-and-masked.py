#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from PIL import Image
#reading the dataset(Just pulling in one of the dataset present)
df = pd.read_csv("/kaggle/input/nyt-comments/CommentsMarch2018.csv")
#Dimensions of the dataset
df.shape
#Head of the dataframe
df.head(3)
df = df[['commentBody']]
df.head()
#Set of StopWords
stopwords = set(STOPWORDS)
print(stopwords)
def Tokenizer(text):
    """Simple Tokenizer function which also converts complete corpus into lowercase. Input the text"""
    full_string = ''
    for sentence in text:
        sent = str(sentence)
        token = sent.split()
        for i in range(len(token)):
            token[i] = token[i].lower()
        full_string += " ".join(token)
    return full_string
full_string = Tokenizer(df['commentBody'])
full_string[0:50]
#Creating the wordcloud; Setting the width,height,background color,font size for the wordcloud
wordcloud = WordCloud(width = 500,height=500,background_color='black',stopwords=stopwords,min_font_size=10).generate(full_string)
#Plotting the WordCloud
plt.figure(figsize=(7,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#Importing a comment image and converting into a numpy array. This will be your masked data.
mask = np.array(Image.open("../input/mask-images/comment_image.png"))
#Creating the wordcloud; Setting the width,height,background color,font size for the wordcloud and passing mask
wordcloud_masked = WordCloud(width = 500,height=500,background_color='white',stopwords=stopwords,min_font_size=10,mask=mask).generate(full_string)
#Plotting the WordCloud
plt.figure(figsize=(7,7))
plt.imshow(wordcloud_masked)
plt.axis("off")
plt.show()
#Importing a coloured cloud image and converting into a numpy array. This will be your masked data.
mask_cloud = np.array(Image.open("../input/mask-images/coloured_cloud.png"))
#ImageColorGenerator from wordcloud can be used in a way to recolor your wordcloud later 
image_colors = ImageColorGenerator(mask_cloud)
#Creating the wordcloud; Setting the width,height,background color,font size for the wordcloud and passing mask
wordcloud_masked = WordCloud(width = 500,height=500,background_color='grey',stopwords=stopwords,min_font_size=10,mask=mask_cloud).generate(full_string)
#Plotting the WordCloud with recoloring.
#The words in the side in white color is becuase of the white color captured from the image background
plt.figure(figsize=(7,7))
plt.imshow(wordcloud_masked.recolor(color_func=image_colors))
plt.axis("off")
plt.show()
#Importing the circle image and converting into a numpy array. This will be your masked data.
mask_circle = np.array(Image.open("../input/mask-images/word_cloud_circle.PNG"))
#Creating the wordcloud; Setting the width,height,background color,font size for the wordcloud and passing mask
#Here the contour_width is set greater than 0 so that whenever the value is greater than zero and mask is not None,draws contour
wordcloud_masked = WordCloud(width = 500,height=500,background_color='grey',stopwords=stopwords,min_font_size=10,
                             contour_width=3, contour_color='steelblue',mask=mask_circle).generate(full_string)
#Plotting the WordCloud
plt.figure(figsize=(7,7))
plt.imshow(wordcloud_masked)
plt.axis("off")
plt.show()