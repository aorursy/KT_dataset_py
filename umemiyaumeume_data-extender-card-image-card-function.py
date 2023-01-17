# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pprint import pprint as pp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sys

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# please set your email adress.

ADRESS = ''



ROBOT_NAME = 'Kaggle yu-gi-oh Crawler ({})'

CRAWRING_INTERVAL = 0.1

SAVE_EXT = '.jpg'
cards_data_csv = os.path.join('../input', 'card_data.csv')

df = pd.read_csv(cards_data_csv, index_col=0)

cols = df.columns.values

card_names = df.index.values

card_type_dir = {}

for col in cols:

    if col == 'ATK' or col == 'DEF':

        continue

    if not col in card_type_dir:

        card_type_dir[col] = []

    for status in df[col]:

        if not status in card_type_dir[col] and not pd.isnull(status):

            card_type_dir[col].append(status)





pp(cols)

pp(card_type_dir)

for param in card_type_dir.keys():

    print('{} has {} elements.'.format(param, str(len(card_type_dir[param]))))
aim_target = {

    'Type': [

        'Normal Monster',

        'Effect Monster'

    ]

}
target_card_names = []

if len(aim_target) != 0:

    for param in aim_target.keys():

        for element in aim_target[param]:

            target_card_names.append(df[(df[param] == element)].index.values)

    target_card_names = list(set([flat for inner in target_card_names for flat in inner]))

else:

    target_card_names = card_names

print('crawling card num :{}'.format(len(target_card_names)))



del card_names
from bs4 import BeautifulSoup

from urllib.request import Request, urlopen

from urllib.parse import quote

from mimetypes import guess_extension

from time import sleep

from tqdm import tqdm_notebook



os.makedirs(os.path.join('../extend_input', 'card_images'), exist_ok=True)

# print(os.listdir('..'))

# print(quote('たのしい ピクニック'))

url = 'https://www.bing.com/images/search?q='

img_link = []

header = {

    'User-Agent': ROBOT_NAME.format(ADRESS) +' Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36' 

}

for cname in tqdm_notebook(target_card_names):

    url_with_query = url + quote(cname) + "&FORM=HDRSC2"

    req = Request(url_with_query, headers=header)

    try:

        with urlopen(req, timeout=3) as p:

            content = p.read()

            mime = p.getheader('Content-Type')

            if not mime or not content:

                continue

    except:

        sys.stderr.write('Error in fetching :'+ format(url_with_query) + "\n")

        continue

    soup = BeautifulSoup(content, 'html.parser')

    img_elem = soup.find_all('div', attrs={'class': 'hoff'})[0].find('img' , attrs={'class': 'mimg'}).get('src').split('&')[0]

#     ext = guess_extension(mime.split(';')[0])

#     if ext in ('.jpeg', '.jpg', '.png', '.jpe'):

    ext = SAVE_EXT

    try:

        with open(os.path.join('../extend_input', 'card_images', cname + ext), 'wb') as f:

            f.write(urlopen(img_elem).read())

    except:

        sys.stderr.write('Error in Saving :'+ format(cname) + "\n")

        continue

        sys.exit(0)

    print('/saved. {}/'.format(cname + ext), end='')

    sleep(CRAWRING_INTERVAL)
# url = 'https://www.bing.com/images/search?q='+quote('Gravekeepers Watcher')+"&FORM=HDRSC2"

# req = Request(url, headers=header)

# try:

#     with urlopen(req, timeout=3) as p:

#         content = p.read()

#         mime = p.getheader('Content-Type')

#         if not mime or not content:

#             pass

# except:

#     sys.stderr.write('Error in fetching :'+ format(url) + "\n")

# soup = BeautifulSoup(content, 'lxml')

# img_elem = soup.find_all('div', attrs={'class': ['img_cont', 'hoff']})[0].find('img' , attrs={'class': 'mimg'})

# print(img_elem.get('src').split('&')[0])