# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed
df = pd.read_csv('../input/OnlineNewsPopularityReduced.csv', sep=',')
(df.info())

# Информация о датасете(тип каждого признака, есть ли в данных пропуски)
df.head()

# Просмотр первых n строк таблицы (по умолчанию n=5)
df.describe().T

#Описания по параметрам(транспонировано для удобого просмотра)
df.groupby(['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 

            'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']).size()

#Группируем по дням недели
df.loc[df['weekday_is_monday'] == 1, 'Publications by weekdays'] = 'Monday'

df.loc[df['weekday_is_tuesday'] == 1, 'Publications by weekdays'] = 'Tuesday'

df.loc[df['weekday_is_wednesday'] == 1, 'Publications by weekdays'] = 'Wednesday'

df.loc[df['weekday_is_thursday'] == 1, 'Publications by weekdays'] = 'Thursday'

df.loc[df['weekday_is_friday'] == 1, 'Publications by weekdays'] = 'Friday'

df.loc[df['weekday_is_saturday'] == 1, 'Publications by weekdays'] = 'Saturday'

df.loc[df['weekday_is_sunday'] == 1, 'Publications by weekdays'] = 'Sunday'

#Создадим общую переменную для дней недели, в которой собственно и будут хранится дни(так будет красивее и нагляднее)
df.groupby(['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 

            'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']).size().plot(kind='bar') 

plt.show()

#Вариант визуализации без единого для дней недели столбца('Publications by weekdays'), он рабочий, но просто не такой приятный
df.groupby(['Publications by weekdays']).size()

#Отображаем дни недели и количество опубликованных в этот день статей
df.groupby(['Publications by weekdays']).size().plot(kind='bar') 

plt.show()

#Сделаем наглядную визуализацию данных показанных выше
df['n_tokens_title'].hist(bins = 20);
from scipy.stats import spearmanr



cor_title_shares = spearmanr(df['n_tokens_title'], df['shares'])

print('Correlation between number of words in the title and share = ', cor_title_shares[0], 'p-value =', cor_title_shares[1])

#Используем коэффициент корреляции спирмена для определения связи
cor_img_shares = spearmanr(df['num_imgs'], df['shares'])

print('Correlation between number of images in the title and share =', cor_img_shares[0], 'p-value =', cor_img_shares[1])

cor_video_shares = spearmanr(df['num_videos'], df['shares'])

print('Correlation between number of videos in the title and share =', cor_video_shares[0], 'p-value =', cor_video_shares[1])

#Снова таки найдём корреляцию по спирману отдельно как для количества картинок, так и для количества видеороликов
df.groupby('is_weekend')['shares'].mean().plot(kind='barh') 

plt.xlabel('Share') 

plt.ylabel('Is a weekend') 

plt.show();

#Обьеденияем показатели ('выходной ли' с 'популярностью') и визуализируем.
cor_content_shares = spearmanr(df['n_tokens_content'], df['shares'])

print('Correlation between number of words in the content and share = ', cor_content_shares[0], 'p-value =', cor_content_shares[1])
cor_hrefs_shares = spearmanr(df['num_hrefs'], df['shares'])

print('Correlation between number of links in the title and share =', cor_hrefs_shares[0], 'p-value =', cor_hrefs_shares[1])

cor_self_shares = spearmanr(df['num_self_hrefs'], df['shares'])

print('Correlation between number of  links to other articles published by Mashable in the title and share =', cor_self_shares[0], 'p-value =', cor_self_shares[1])
numeric = ['num_keywords', 'n_unique_tokens', 'n_tokens_title', 'n_tokens_content', 'num_imgs', 'num_videos', 'num_hrefs', 'shares']

sns.heatmap(df[numeric].corr(method='spearman'), annot = True);
cor_keywords_shares = spearmanr(df['num_keywords'], df['shares'])

print('Correlation between number of keywords in the metadata and share =', cor_keywords_shares[0], 'p-value =', cor_keywords_shares[1])