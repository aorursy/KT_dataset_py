# set directory

import os

os.chdir('../input/customized-wordcloud/')
# name the word cloud data source as variable text

text = open('text.txt', 'r',encoding= 'UTF-8-sig').read()

text[0:500]
import jieba

text = ' '.join(jieba.cut(text))

text[0:500]
from PIL import Image

import numpy as np

icon_path = 'icon.png'

icon = Image.open(icon_path)

mask = Image.new("RGB", icon.size, (255,255,255))

mask.paste(icon,icon)

mask = np.array(mask)
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return tuple([255,255,255]) # RGB code of white color
from wordcloud import ImageColorGenerator

color_func = ImageColorGenerator(mask)
import random

from palettable.colorbrewer.sequential import YlGnBu_9 # choose the color set you like

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return tuple(YlGnBu_9.colors[random.randint(0,8)]) # we got 9 colors, so we generate random number from 0 to 8
# indicate the font path that can display Chinese

font_path = 'SNsanafonGyou.ttf'
from wordcloud import WordCloud

import matplotlib.pyplot as plt

wc = WordCloud(font_path=font_path, background_color="black", max_words=2000, mask=mask, max_font_size=300, random_state=1)

wc.generate_from_text(text)

wc.recolor(color_func=color_func, random_state=2)



# save as png

#output_path = 'wordcloud.png'

#wc.to_file(output_path)



# display the word cloud

plt.rcParams["figure.figsize"] = (25,25)

plt.imshow(wc)

plt.axis("off")

plt.show()