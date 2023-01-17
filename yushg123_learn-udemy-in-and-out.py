# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')
data
#The data is not consistent for this column, so I am converting it to a True and false. Also, there is a udemy course link in this is paid column. Must be a bug, but I checked and the 

#course is paid, so I will be replacing it with True.

data['is_paid'] = data['is_paid'].replace('FALSE', 'False')

data['is_paid'] = data['is_paid'].replace('TRUE', 'True')

data['is_paid'] = data['is_paid'].replace('https://www.udemy.com/learnguitartoworship/', 'True')



sns.set(style="darkgrid")

ax = sns.countplot(x="is_paid", data=data)
data['price'] = data['price'].replace('Free', '0')



#There is one course that has True as its price. I found the quote (Rs. 12480). I will assume that the prices are in dollars, so I will replace it with $178. Also, ignore the next two lines.

#True_sample = data[data['price'] == 'TRUE'].index

#data = data.drop(True_sample)

data['price'] = data['price'].replace('TRUE', '30')



data['price'] = data['price'].astype(int)





plt.hist(data['price'])
sns.boxplot(data['num_subscribers'])
plt.hist(data['subject'])
data['level'] = data['level'].replace('52', 'Beginner Level')

plt.hist(data['level'])
plt.figure(figsize=(20, 6))

ax = sns.violinplot(x="num_subscribers", y="subject", hue="is_paid", data=data, palette="muted")
!pip install bubbly

!pip install chart_studio


from bubbly.bubbly import bubbleplot 

from plotly.offline import iplot

import chart_studio.plotly as py







figure = bubbleplot(dataset=data, x_column='price', y_column='num_subscribers', bubble_column='num_lectures', size_column='num_lectures', height=350)



iplot(figure)
titles = data['course_title'].str.cat(sep=' ')

from wordcloud import WordCloud, ImageColorGenerator



wordcloud = WordCloud(max_words=200, colormap='Set3', background_color='black').generate(titles)



plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()

data['subject'].unique()
sub_data = data[data['subject'] == 'Business Finance']

titles = sub_data['course_title'].str.cat(sep=' ')





wordcloud = WordCloud(max_words=200, colormap='Set3', background_color='black').generate(titles)



plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()

sub_data = data[data['subject'] == 'Graphic Design']

titles = sub_data['course_title'].str.cat(sep=' ')





wordcloud = WordCloud(max_words=200, colormap='Set3', background_color='black').generate(titles)



plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
sub_data = data[data['subject'] == 'Musical Instruments']

titles = sub_data['course_title'].str.cat(sep=' ')





wordcloud = WordCloud(max_words=200, colormap='Set3', background_color='black').generate(titles)



plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()

sub_data = data[data['subject'] == 'Web Development']

titles = sub_data['course_title'].str.cat(sep=' ')





wordcloud = WordCloud(max_words=200, colormap='Set3', background_color='black').generate(titles)



plt.figure(figsize=(15,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
