# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

"""

Created on Sat Dec  7 22:02:45 2019



@author: viren

"""

import time

import datetime

import pandas as pd

import numpy as np

import requests

from bs4 import BeautifulSoup

import re



# For timing

s = time.time()

print("\n\n\t\t\t\t****Webscrapying Begins****\n")

# Getting the content from given link

r = requests.get("https://height-weight-chart.com")

c = r.content #raw text

soup = BeautifulSoup(c,"html.parser")  #html format text



# We can use this and get exact images data,

# but we miss out the html link of each image.

x = soup.find_all("img",{"class":"thumb"})



# This consists of the html link and the height weight info

all = soup.find_all("a")

# Since we know the total images (383), 

# we can select it from the "all" content   386 = (4+383)-1

content = all[3:386]



# Getting the seperate parameters from content

image_link = content[0]['href']

info = content[0].find("img")

text = info.get('title')

src = info.get('src')



# Creating lists for dataframe

title = []

ht_wt = []

file = []

a = len(content)

for i in range(0, a):

    image_link = content[i]['href']

    title = str(title) + ", " + str(image_link)

    

    info = content[i].find("img")

    text = info.get('title')

    text1 = info.get('src')

    ht_wt = str(ht_wt) + ", " + str(text)

    file = str(file) + ", " + str(text1)

    

title = title.split(",")

file = file.split(",")

f = [x.replace('s/', '') for x in file]

a = [x.replace('_s', '') for x in f]

file = a

ht_wt = ht_wt.split(",")



path = "https://height-weight-chart.com/"

link = []

link = [path.strip() + x.strip() for x in title]





# Downloading the images from the homepage



##This script downloads low quality images

#print("\n\n****Downloading begins****\n\n")



# response = requests.get(path)

# soup = BeautifulSoup(response.text, 'html.parser')

# img_tags = soup.find_all("img",{"class":"thumb"})

# urls = [img['src'] for img in img_tags]



# for url in urls:

#     filename = re.search(r'/([\w_-]+[.](jpg))$', url)

#     with open(filename.group(1), 'wb') as f:

#         if 'http' not in url:

#             # sometimes an image source can be relative 

#             # if it is provide the base url which also happens 

#             # to be the site variable atm. 

#             url = '{}{}'.format(path, url)

#         response = requests.get(url)

#         f.write(response.content)



# print("\n\n****All images are downloaded****\n\n")









#Downloading the images from individual pages of the subject



##This script downloads high quality images

c = time.time()

print("\nTime Elapsed: ",datetime.timedelta(seconds=c-s))

print("\n\n\t\t****Collecting metadata for downloading images and .csv file****\n")



k=0

a = len(link)

img_tags=[]

htwt = []

nt_img = []

nt_htwt = []

print("\n\t\t\t\t****This will take some time****")

for i in range(1,a):

    response = requests.get(link[i])

    soup = BeautifulSoup(response.text, 'html.parser')

    all_tags = soup.find_all("img")

    tag = file[i]

    flag = len(img_tags)

    for j in range(len(all_tags)):

        all = all_tags[j].get('src')

        pattern = "l/" + tag[0:5].strip() ##selecting only first elements "height" as it is similar

        result = str(pattern) in str(all)

        if result == True:

            img_tags.append(all)

            htwt.append(ht_wt[i])

            check = len(img_tags)

            break;

    if flag == check:                  ## Finding the links of the image tag, that are not appended

        nt_img.append(link[i])

        nt_htwt.append(ht_wt[i])

    print("%s /" %i,(a-1))





c = time.time()

print("\nTime Elapsed: ",datetime.timedelta(seconds=c-s))

print("\n\nNumber of images found:",len(img_tags))

print("\nNumber of images remaining:",(a-1)-len(img_tags))

print("\nImages not found: \n",nt_img)



## Appending the remaining images to found images "tags"

for i in range(len(nt_img)):

    response = requests.get(nt_img[i])

    soup = BeautifulSoup(response.text, 'html.parser')

    tags = soup.find_all("img",{"class":"largepic"})

    tag = tags[0]

    nt_all = tag['src']

    img_tags.append(nt_all)

    htwt.append(nt_htwt[i])



print("\n\nTotal images found:",len(img_tags))

print("\n\t\t\t\t****All images are found****\n")    

c = time.time()

print("\nTime Elapsed: ",datetime.timedelta(seconds=c-s))

print("\n\t\t\t\t****Downloading begins****\n\n\nDownloaded images are being stored in the same path as the 'scraper.py' file\n\n")

for url in img_tags:

    filename = re.search(r'/([\w_-]+[.](jpg))$', url)

    with open(filename.group(1), 'wb') as f:

        if 'http' not in url:

            # sometimes an image source can be relative 

            # if it is provide the base url which also happens 

            # to be the site variable atm. 

            url = '{}{}'.format(path, url)

        response = requests.get(url)

        f.write(response.content)

        print(url)



c = time.time()

print("\n\n\t\t\t\t\t****All images are downloaded****\n\n")

c = time.time()

print("\nTime Elapsed: ",datetime.timedelta(seconds=c-s))





# Creating DataFrame

img_name = []

for x in img_tags:

    x = x.replace('l/','')

    img_name.append(x)



df = pd.DataFrame()

df["Image_link"] = link[1:]

df["Filename"] = img_name

df["Height & Weight"] = htwt

df = pd.DataFrame(np.sort(df.values, axis=0), index=df.index, columns=df.columns)

df.to_csv("Output_data.csv")



print("\t\t\t\t****Output CSV file is created****\n")

print("Final execution Time: ",datetime.timedelta(seconds=c-s))