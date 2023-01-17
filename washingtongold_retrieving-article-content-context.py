import pandas as pd

data = pd.read_csv('/kaggle/input/epic-question-answering-consumer-covid-articles/epic_qa_consumer.csv')

data.head()
import numpy as np

data['content'] = np.nan
import urllib.request

import bs4



def get_content(url):

    try:

        webpage=str(urllib.request.urlopen(url).read())

        soup = bs4.BeautifulSoup(webpage)

        text = soup.get_text().replace('\\t', '').replace('\\r','').replace('\\n','')

        return text

    except:

        print("Unsuccessful: {}".format(url))

        return np.nan
from tqdm import tqdm



def is_str(v):

    return type(v) is str

for index in tqdm(data[data['url'].apply(lambda x:1 if is_str(x) else 0) == 1].index):

    data.loc[index,'content'] = get_content(data.loc[index,'url'])
data
data.to_csv('epic_qa_consumer_w_content.csv')