# importing required packages

import os

import numpy as np

import random as rnd

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from PIL import Image
os.listdir('../input/pokemon-images-and-types')
os.listdir('../input/pokemon-images-and-types/images/images')[:20]
len(os.listdir('../input/pokemon-images-and-types/images/images'))
pokemons = pd.read_csv('../input/pokemon-images-and-types/pokemon.csv')

pokemons.head(10)
pokemons.nunique()
def createType(row):

    if row['Type2']=='None':

        return row['Type1']

    return '-'.join([row['Type1'], row['Type2'] ])
pokemons['Type2'].fillna('None', inplace=True)

pokemons['Type'] = pokemons.apply(lambda row: createType(row), axis=1)

pokemons.head(10)
pokemons.nunique()
labels = ['One type pokemons', 'Two types pokemons']

sizes = [pokemons['Type2'].value_counts()['None'], 

         pokemons['Type2'].count() - pokemons['Type2'].value_counts()['None']]

colors = ['lightskyblue', 'lightcoral']



patches, texts, _ = plt.pie(sizes, colors=colors, startangle=90, autopct='%1.1f%%')

plt.legend(patches, labels, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()
def createBarChart(data, name=''):

    colors = {'Water': 'blue', 'Normal': 'orange', 'Grass': 'green', 'Bug': 'pink', 'Fire': 'red',

              'Psychic': 'purple', 'Rock': 'gray', 'Electric': 'yellow', 'Poison': 'lightgreen', 'Ground': 'brown',

              'Dark':  'darkblue', 'Fighting': 'crimson', 'Dragon': 'salmon', 'Ghost': 'orchid', 

              'Steel': 'silver', 'Ice': 'lightblue', 'Fairy': 'darkgreen', 'Flying': 'orangered', 'None': 'black'}

    labels = [name for name in data.keys()]

    values = [data[name] for name in data.keys()]

    bar_colors = [colors[t.split('-')[0]] for t in labels]

    

    plt.bar(labels, values, color=bar_colors)

    plt.xticks(rotation = 90)

    plt.ylabel('Counts')

    plt.title(name)

    

    plt.tight_layout()
plt.figure(2, figsize=(13, 6), edgecolor = 'k')



plt.subplot(121)

createBarChart(pokemons['Type1'].value_counts(), name='First type of pokemons')



plt.subplot(122)

createBarChart(pokemons['Type2'].value_counts().drop(['None']), name='Second type of pokemons')



plt.show()
plt.figure(18, figsize=(18, 36))



for i, key in enumerate(pokemons['Type1'].value_counts().keys()):

    subtypes = pokemons.loc[pokemons['Type1']==key]['Type'].value_counts()

    plt.subplot(6, 3, i + 1)

    createBarChart(subtypes, name='{} pokemon\'s subtypes distribution'.format(key))



plt.tight_layout()

plt.show()
counts = pokemons['Type'].value_counts()

pokemons['Counts'] = [counts[x] for x in pokemons['Type']]

data = pd.pivot_table(data=pokemons, index='Type1', columns='Type2', values='Counts')



sns.set(rc={'figure.figsize':(8,14)})

sns.heatmap(data, cmap='coolwarm', annot=True, cbar=False, square=True, linewidths=.5)
fig = plt.figure(16, figsize=(18, 18))



for i, pic in enumerate(rnd.sample(os.listdir('../input/pokemon-images-and-types/images/images'), 16)):

    a = fig.add_subplot(4, 4, i + 1)

    img = plt.imshow(mpimg.imread('../input/pokemon-images-and-types/images/images/{}'.format(pic)))

    a.set_title(pic)

    plt.grid(None)



plt.show()
img = mpimg.imread('../input/pokemon-images-and-types/images/images/psyduck.png')

print(img.shape)
img = mpimg.imread('../input/pokemon-images-and-types/images/images/lurantis.jpg')

print(img.shape)
images = []



fill_color = (255,255,255)



for img in os.listdir('../input/pokemon-images-and-types/images/images'):

    im = Image.open('../input/pokemon-images-and-types/images/images/{}'.format(img))

    if img.split('.')[1] == 'png':

        im = im.convert("RGBA")

        if im.mode in ('RGBA', 'LA'):

            bg = Image.new(im.mode[:-1], im.size, fill_color)

            bg.paste(im, im.split()[-1]) # omit transparency

            im = bg 

    images.append(np.array(im))
fig = plt.figure(16, figsize=(18, 18))



for i, pic in enumerate(rnd.sample(images, 16)):

    a = fig.add_subplot(4, 4, i + 1)

    img = plt.imshow(Image.fromarray(pic))

    plt.grid(None)



plt.show()