# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import re

import time

from datetime import datetime

import matplotlib.dates as mdates

import matplotlib.ticker as ticker

from urllib.request import urlopen

from bs4 import BeautifulSoup

import requests
no_pages = 2



def get_data(pageNo):  

#     headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept-Encoding":"gzip, deflate", "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "DNT":"1","Connection":"close", "Upgrade-Insecure-Requests":"1"}

# +str(pageNo)+'?ie=UTF8&pg='+str(pageNo), headers=headers)#, proxies=proxies

    r = requests.get('https://www.cdc.gov/coronavirus/2019-ncov/faq.html')

    content = r.content

    soup = BeautifulSoup(content)

    print(soup)



    alls = []

    for d in soup.findAll('span', attrs={'role':'heading'}):

        print(d)

    

    for d in soup.findAll('div', attrs={'class':'d-print-block collapse show'}):

        print(d)

#         name = d.find('span', attrs={'class':'zg-text-center-align'})

#         n = name.find_all('img', alt=True)

#         #print(n[0]['alt'])

#         author = d.find('a', attrs={'class':'a-size-small a-link-child'})

#         rating = d.find('span', attrs={'class':'a-icon-alt'})

#         users_rated = d.find('a', attrs={'class':'a-size-small a-link-normal'})

#         price = d.find('span', attrs={'class':'p13n-sc-price'})



#         all1=[]



#         if name is not None:

#             #print(n[0]['alt'])

#             all1.append(n[0]['alt'])

#         else:

#             all1.append("unknown-product")



#         if author is not None:

#             #print(author.text)

#             all1.append(author.text)

#         elif author is None:

#             author = d.find('span', attrs={'class':'a-size-small a-color-base'})

#             if author is not None:

#                 all1.append(author.text)

#             else:    

#                 all1.append('0')



#         if rating is not None:

#             #print(rating.text)

#             all1.append(rating.text)

#         else:

#             all1.append('-1')



#         if users_rated is not None:

#             #print(price.text)

#             all1.append(users_rated.text)

#         else:

#             all1.append('0')     



#         if price is not None:

#             #print(price.text)

#             all1.append(price.text)

#         else:

#             all1.append('0')

#         alls.append(all1)    

#     return alls