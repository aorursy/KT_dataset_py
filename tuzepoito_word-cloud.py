import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
f = open('../input/pararomance', "rt")

text = f.readlines()

df = pd.DataFrame({'novel_titles':text})

print(df.head(5))
plt.subplots(figsize=(12,12))

wordcloud = WordCloud(

                          background_color='white',

                          width=1024,

                          height=768

                         ).generate(" ".join(df['novel_titles']))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
frequencies = pd.Series(wordcloud.words_)

frequencies.nlargest(20)