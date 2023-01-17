!git clone https://github.com/lucasdavid/wikiart.git

!python3 wikiart/wikiart.py --datadir ./wikiart-saved/ fetch --only artists
import json



with open('./wikiart-saved/meta/artists.json', 'r') as artists_file:

    parsed_json = json.load(artists_file)

    

print(json.dumps(parsed_json, indent=4, sort_keys=True))
entry = '''[{

        "artistName": "Claude Monet",

        "birthDay": "/Date(-4074969600000)/",

        "birthDayAsString": "November 14, 1840",

        "contentId": 211667,

        "deathDay": "/Date(-1359331200000)/",

        "deathDayAsString": "December 5, 1926",

        "dictonaries": [

            1221,

            316

        ],

        "image": "https://uploads0.wikiart.org/00115/images/claude-monet/440px-claude-monet-1899-nadar-crop.jpg!Portrait.jpg",

        "lastNameFirst": "Monet Claude",

        "url": "claude-monet",

        "wikipediaUrl": "http://en.wikipedia.org/wiki/Claude_Monet"

    }]'''



with open('./wikiart-saved/meta/artists.json', 'w') as artists_file:

    artists_file.write(entry)

    



# Result check

with open('./wikiart-saved/meta/artists.json', 'r') as artists_file:

    parsed_json = json.load(artists_file)

    

print(json.dumps(parsed_json, indent=4, sort_keys=True))
!python3 wikiart/wikiart.py --datadir ./wikiart-saved/ fetch
!cd wikiart-saved/images/claude-monet && find . -mindepth 2 -type f -print -exec mv {} . \;

!cd wikiart-saved/images/claude-monet && find -mindepth 1 -maxdepth 1 -type d -exec rm -r {} \;
!ls wikiart-saved/images/claude-monet/
import cv2

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
def reescale(image, size=256):

    width = size

    height = size

    

    dimensions = (width, height)

    

    image = cv2.resize(image, dsize=dimensions, interpolation = cv2.INTER_AREA)

    

    return image





def fix_rgb(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    return image





# This function exists as main preprocessing pipeline,

# in case more steps are to be added

def preprocess(image):

    image = reescale(image, 256)

    

    image = fix_rgb(image)

    

    return image

    
image = cv2.imread('wikiart-saved/images/claude-monet/212986.jpg')



plt.imshow(image)
image = preprocess(image)



plt.imshow(image)
from os import listdir

from os.path import isfile, join
files = [f for f in listdir('wikiart-saved/images/claude-monet') if isfile(join('wikiart-saved/images/claude-monet', f))]



images = []



for file in files:

    file_path = join('wikiart-saved/images/claude-monet', file)

    

    images.append(preprocess(cv2.imread(file_path)))

    
print(len(images), images[0].shape)
def plot_picture_set(images, start_idx=0, columns=5, rows=6, figsize=(16,16)):

    fig=plt.figure(figsize=figsize)



    for i in range(1, columns*rows+1):

        fig.add_subplot(rows, columns, i)

        plt.imshow(images[i + start_idx])

    plt.show()
columns = 5

rows = 6



for i in range(0, len(images) - (len(images) % (columns*rows)), columns*rows):

    plot_picture_set(images, i, columns, rows)