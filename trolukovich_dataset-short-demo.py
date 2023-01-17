import pandas as pd

import numpy as np

import requests

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os



from PIL import Image

from io import BytesIO



%config InlineBackend.figure_format = 'svg'

pd.set_option('display.max_columns', 100)

warnings.filterwarnings('ignore')
# There are 22 csv files in this dataset, each for unique category of items

os.listdir('../input/world-of-warcraft-items-dataset')
# Number of items in each csv file:

for file in os.listdir('../input/world-of-warcraft-items-dataset'):

    df = pd.read_csv(f'../input/world-of-warcraft-items-dataset/{file}')

    print(f'{file}: {df.shape[0]} items')
# Typical WOW items look like this:

urls = ['https://i.imgur.com/H4OXGV3.png', 'https://www.raiditem.com/upload/itemico/201704091108461.jpg']



fig = plt.figure(figsize = (13, 9))

for i, url in enumerate(urls):

    r = requests.get(url)

    plt.subplot(f'12{i+1}')

    plt.imshow(Image.open(BytesIO(r.content)))

    plt.axis('off')
df = pd.read_csv('../input/world-of-warcraft-items-dataset/two hand.csv')



# First 5 rows

df.head()
# Last 5 rows

df.tail()
# Number of 2handed weapons by level

# We can see peaks on last levels for each expansion pack - 60, 70, 80, 85 and so on

counts = df['reqlevel'].value_counts()

fig = plt.figure(figsize = (11, 5))

sns.lineplot(x = counts.index.values, y = counts.values).set_title('Number of 2handed weapons by level')
# Let's look at mean damage of 2handed weapon by level:

df['avg_dmg'] = (df['dmgmin1'] + df['dmgmax1']) / 2

mean_dmg_per_level = df[['avg_dmg', 'reqlevel']].dropna().groupby('reqlevel', as_index = False).mean()



fig = plt.figure(figsize = (11, 5))

sns.lineplot(x = 'reqlevel', y = 'avg_dmg', data = mean_dmg_per_level).set_title('Mean damage of 2handed weapon by level')



# We can see a huge peak at level 110 an drop on 110-120 level because of new damage system in BfA expansion
lvl85 = df.loc[df['reqlevel'] == 85]

print(lvl85.shape)
fig = plt.figure(figsize = (10, 6))

sns.lineplot(x = lvl85.index, y = lvl85['mledmgmax'], label = 'max')

sns.lineplot(x = lvl85.index, y = lvl85['mledmgmin'], label = 'min')

sns.lineplot(x = lvl85.index, y = lvl85['dps'], label = 'dps')

plt.ylabel('')

plt.legend()
cols = ['name_enus', 'mledmgmax', 'mledmgmin', 'dps', 'mlespeed', 'str', 'sta', 'critstrkrtng', 'mastrtng', 'hastertng', 'socket1']

lvl85[cols].sort_values(by = 'dps', ascending = False).head(6)
sulfuras = 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/609cb9b9-7ff9-432d-a57a-0152e0902e82/d5k1igv-4e890a9b-e2e8-4614-8fbd-78673be30928.jpg/v1/fill/w_1024,h_512,q_75,strp/sulfuras__the_extinguished_hand_by_soki_art_d5k1igv-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NTEyIiwicGF0aCI6IlwvZlwvNjA5Y2I5YjktN2ZmOS00MzJkLWE1N2EtMDE1MmUwOTAyZTgyXC9kNWsxaWd2LTRlODkwYTliLWUyZTgtNDYxNC04ZmJkLTc4NjczYmUzMDkyOC5qcGciLCJ3aWR0aCI6Ijw9MTAyNCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.GAeNtxf43dvgjclk-4hBMzc7iPsxPougzFgpkx1QBsg'

r = requests.get(sulfuras)    

plt.imshow(Image.open(BytesIO(r.content)))

plt.axis('off')

plt.show()
shalug = 'https://wow.zamimg.com/uploads/screenshots/normal/298322-shalugdoom-the-axe-of-unmaking.jpg'

r = requests.get(shalug)    

plt.imshow(Image.open(BytesIO(r.content)))

plt.axis('off')

plt.show()
skull = 'https://www.speed4game.com/upload/image/two-handed%20axe/Skullstealer%20Greataxe2.jpg'

r = requests.get(skull)    

plt.imshow(Image.open(BytesIO(r.content)))

plt.axis('off')

plt.show()