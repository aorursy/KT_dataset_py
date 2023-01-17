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
import requests

from bs4 import BeautifulSoup

import pandas as pd



prefix = "https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/"

webpage_response = requests.get('https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/shellter.html')



webpage = webpage_response.content

soup = BeautifulSoup(webpage, "html.parser")



turtle_links = soup.find_all("a")

links = []

#go through all of the a tags and get the links associated with them"

for a in turtle_links:

  links.append(prefix+a["href"])

    

#Define turtle_data:

turtle_data = {}



#follow each link:

for link in links:

  webpage = requests.get(link)

  turtle = BeautifulSoup(webpage.content, "html.parser")

  turtle_name = turtle.select(".name")[0].get_text()

  

  stats = turtle.find("ul")

  stats_text = stats.get_text("|")

  turtle_data[turtle_name] = stats_text.split("|")



turtle_df = pd.DataFrame.from_dict(turtle_data, orient='index')



turtle_df.head()

  