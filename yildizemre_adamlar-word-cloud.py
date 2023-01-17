import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import warnings 
warnings.filterwarnings('ignore')

adamlar = pd.read_csv('../input/adamlar1/adamlar.csv')
adamlar['Unnamed: 1']
def plot_wordcloud(text, mask=None, max_words=1000, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 80,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
d = '../input/aharfi/'
c=[]
for i in range(1,30):
    b=a[i]
    a[i]=b.replace('_x000D_\n',' ')
    a[i]=b.replace('reklamı gizle / hide ads',' ')
    a[i]=b.replace('x2',' ')
    a[i]=b.replace('x3',' ')   
    
    c.append(a[i])

e ='../input/masks/'
adamlar_text = str(c)
adamlar_mask = np.array(Image.open(e + 'user.png'))
plot_wordcloud(adamlar_text, adamlar_mask, max_words=1000, max_font_size=50, 
               title = 'Adamlar Word Cloud', title_size=50)
editorsPick_comments_text = str(a)
star_mask = np.array(Image.open(e + 'star.png'))
plot_wordcloud(comments_text, star_mask, max_words=8000, max_font_size=120, figure_size=(16,14),
               title = 'ADAMLAR', image_color=True)
