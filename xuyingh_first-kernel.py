# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objs as go
import plotly.tools as tls
from nltk.corpus import stopwords
import string
import time
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
punctuation = string.punctuation
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
resource = pd.read_csv('../input/resources.csv')
trian_resource = pd.merge(train,resource,how='left',on='id')
test_resource = pd.merge(test,resource,on='id',how='left')
train.describe()
train.describe(include=['object'])
temp = train['school_state'].value_counts()
temp_index = temp.index
temp_values = temp.values / len(train)*100
df = pd.DataFrame({'state Name':temp_index,'precent of disturbion':temp_values})
figure = plt.figure(figsize=(16,6))
ax = sns.barplot(x='state Name',y='precent of disturbion',data=df,errwidth=4)
plt.xlabel('state Name',fontsize=15)
plt.ylabel('precent of disturbion %',fontsize=15)
plt.title('Distribution of school states in %',fontsize=15,y=1.05)
plt.show()
temp = train['project_grade_category'].value_counts()
temp_index = temp.index
temp_values = temp.values / len(train)*100
df = pd.DataFrame({'school grade levels':temp_index,'precent of disturbion':temp_values})
figure = plt.figure(figsize=(16,6))
ax = sns.barplot(x='school grade levels',y='precent of disturbion',data=df,errwidth=4,color='blue')
plt.xlabel('school grade levels',fontsize=15)
plt.ylabel('precent of disturbion %',fontsize=15)
plt.title('Distribution of project_subject_categories in %',fontsize=15,y=1.05)
plt.show()
import re


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()# lowercase text  
    text = REPLACE_BY_SPACE_RE.sub(' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text    
    text = BAD_SYMBOLS_RE.sub('',text)# delete symbols which are in BAD_SYMBOLS_RE from text    
    temp = [s.strip() for s in text.split() if s not in STOPWORDS]# delete stopwords from text
    new_text = ''
    for i in temp:
        new_text +=i+' '
    text = new_text
    return text.strip()
temp_data = train.dropna(subset=['project_resource_summary'])
# converting into lowercase
temp_data['project_resource_summary'] = temp_data['project_resource_summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))
temp_data['project_resource_summary'] = temp_data['project_resource_summary'].map(text_prepare)


from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(temp_data['project_resource_summary'].values))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Top resources needed for the project", fontsize=35)
plt.axis("off")
plt.show() 

df_accept= train.loc[train['project_is_approved']==1,'teacher_prefix'].value_counts()
df_reject = train.loc[train['project_is_approved']==0,'teacher_prefix'].value_counts()
df_da = pd.DataFrame({'accept':df_accept.values,'reject':df_reject.values},index=df_accept.index)
df_da.plot(kind='bar',title='distribution of ',figsize=(10,5),use_index=True,fontsize=14,rot='-45',stacked=True)
full_data = [train,test]
for dataset in full_data:
    dataset['project_submitted_datetime'] = pd.to_datetime(dataset['project_submitted_datetime'])
    dataset['year'] = dataset['project_submitted_datetime'].dt.year
    dataset['month'] = dataset['project_submitted_datetime'].dt.month
    dataset['day_month'] = dataset['project_submitted_datetime'].dt.day
    dataset['weekday'] = dataset['project_submitted_datetime'].dt.weekday
    dataset['hour'] = dataset['project_submitted_datetime'].dt.hour
    dataset['minute'] = dataset['project_submitted_datetime'].dt.minute
train.head(10)
    
resource.head()
for dataset in full_data:
    dataset['teacher_prefix'] =  dataset['teacher_prefix'].replace(['Rare'],'Teacher')

for dataset in full_data:
    dataset['teacher_prefix'] =  dataset['teacher_prefix'].replace(['Teacher','Dr.',np.nan],'Rare')
from sklearn.preprocessing import LabelEncoder
features = [
    'teacher_prefix',
    'school_state',
    'project_grade_category',
    'project_subject_categories',
    'project_subject_subcategories']
df_full_data = [train,test]
for dataset in df_full_data:
    for feature in features:
        le = LabelEncoder()
        le.fit(dataset[feature])
        dataset[feature] = le.transform(dataset[feature])
train.head()
            
full