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
# reading csv
df = pd.read_csv("../input/newyork-room-rentalads/room-rental-ads.csv")
df.head()
df.describe()
df.shape
df.dtypes
df.isna().sum()
df.dropna(how = "any", inplace=True)
df
df["Vague/Not"].value_counts()
# renaming columns
df.rename(columns={'Description': 'desc', 'Vague/Not': 'classify'},inplace=True)

df
count = list(df['classify'].value_counts())
count
# visualisation
import matplotlib.pyplot as plt  
x=['vague','Not vague']
y = count
fig = plt.figure(figsize = (15, 5)) 
  
plt.bar(x, y, color ='pink',width = 0.5) 
  
plt.xlabel("Vague or Not vague") 
plt.ylabel("No. of ads in each group") 

plt.title("Classification") 
plt.show() 
df.dtypes
df.classify = df.classify.astype("int").astype("category")
df
import re
import spacy
nlp = spacy.load('en')

def normalize(msg):
    
    msg = re.sub('[^A-Za-z]+', ' ', msg) #remove special character and intergers
    doc = nlp(msg)
    res=[]
    for token in doc:
        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 2): #word filteration
            pass
        else:
            res.append(token.lemma_.lower())
    return res

df["desc"] = df["desc"].apply(normalize)
df.head(10)
# words count
from collections import Counter
words_collection = Counter([item for subtext in df['desc'] for item in subtext])
most_common = pd.DataFrame(words_collection.most_common(20))
most_common.columns = ['most_common_word','count']
most_common
import numpy as np 
import plotly 
import plotly.graph_objects as go 
import plotly.offline as pyo 
from plotly.offline import init_notebook_mode 
  
init_notebook_mode(connected=True) 
  
# generating 150 random integers 
# from 1 to 50 
x = list(most_common['most_common_word'])
  

y = list(most_common['count'])
  
# plotting scatter plot 
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers', marker=dict( 
        color=np.random.randn(10), 
        colorscale='Viridis',  
        showscale=True
    ) )) 
  
fig.show() 
import plotly.express as px 
  
fig = px.sunburst(most_common, path=['most_common_word'],values='count',color ='count')
fig.show()