# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/ks-projects-201801.csv')
data.info() #this code gives some basic information of datas
data.columns 
data.head() #Let's check some samples and take a position what can do this. 
data_new = data[['main_category','currency','goal','state','country','backers','usd_goal_real']]
data_new.head() #new selected datas.
            
data_new.main_category.value_counts()
datacategory_index = data.main_category.value_counts().index
datacategory_name =  data.main_category.value_counts().values
data_category = pd.DataFrame({'name':datacategory_index,'value':datacategory_name})
#data.main_category.value_counts().index
#data.main_category.value_counts().values

 
plt.figure(figsize=(15,10))
sns.barplot(x=datacategory_index,y=datacategory_name)
plt.xticks(rotation=-40) #gives angle
plt.xlabel("Category Names",size=15,color="cyan")
plt.ylabel("Number of Projects",size=15, color="magenta")
plt.title("Number of applications by categories",size=18,fontweight="bold")
plt.show()
new_index = data_category['value'].sort_values(ascending=True).index.values
sorted_data = data_category.reindex(new_index)
sorted_data_tail = sorted_data.tail(3)
sorted_data_head = sorted_data.head(3)
conc_data_col = pd.concat([sorted_data_head,sorted_data_tail],axis =0,ignore_index =False)
conc_data_col #we concatenation each 3 samples tail and head data 
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['name'],y=sorted_data['value'])
plt.xticks(rotation=-40) #gives angle
plt.xlabel("Category Names",size=15,color="cyan")
plt.ylabel("Number of Projects",size=15, color="magenta")
plt.title("Number of applications by categories",size=18,fontweight="bold")
plt.show()
plt.figure(figsize=(15,10))
ax= sns.barplot(x=datacategory_index, y=datacategory_name,palette =sns.cubehelix_palette(datacategory_index.size))
                #ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Category Names',size=15, color='g')
plt.ylabel('Number of Projects',size=15, color="g")
plt.title('Number of applications by categories',size=18, fontweight="bold")
plt.xticks(rotation=-40)
plt.show()
data_new.country.value_counts()
#You saw 'N,0"' in country.We have to eliminate this undefined character set.
data_new.country.value_counts().index
elected_data = data_new[data_new.country != 'N,0"']
elected_data.country.value_counts().index

country_list = list(elected_data['country'].unique())
#country_list
a = 0
top={} #it's dictionary.
successful = {}
failed = {}
canceled = {}
live = {}
suspend = {}

#sumofallproject
#sumofsuccessful

for c in country_list:
    
    x = elected_data[elected_data['country']==c]
    sumofproject = sum(x.state.value_counts()) #all projects
    top[a]=sumofproject
    successful[a] = x.state.value_counts().successful
    failed[a] = x.state.value_counts().failed
    canceled[a] = x.state.value_counts().canceled
    live[a] = x.state.value_counts().live
    
    a = a+1
    
#df = pd.DataFrame({'col':L})

#Let's sort all data which most appy on kickstarter project and another sorting which mean most successful 

#new_index = data_category['value'].sort_values(ascending=True).index.values
#sorted_data = data_category.reindex(new_index)

#top.keys().values()
#datai
#print(country_list)
frame_data = pd.DataFrame({'countrycode':country_list})
#frame_data['countrycode'] = pd.DataFrame.from_dict(country_list,orient='index')
frame_data['top'] = pd.DataFrame.from_dict(top,orient='index')
frame_data['successful'] = pd.DataFrame.from_dict(successful,orient='index')
frame_data['failed'] = pd.DataFrame.from_dict(failed,orient='index')
frame_data['canceled'] = pd.DataFrame.from_dict(canceled,orient='index')
frame_data['live'] = pd.DataFrame.from_dict(live,orient='index')

#k = list(countries)
#k.astype(object)
#datai = pd.DataFrame[{'country':"deneme1",'isim':"362"}]
#sumofallproject = frame_data[]
   
#datai = pd.DataFrame[{'country':top.keys(),'total':top.values()}]
#frame_data.drop(['0'],axis=1,inplace=True)
#df.drop(['B', 'C'], axis=1)
frame_data
# frame_data.value_counts()


new_index = frame_data['top'].sort_values(ascending=False).index.values
sorted_data2 = frame_data.reindex(new_index)

f,ax = plt.subplots(figsize=(20,10))
sns.barplot(x=sorted_data2['top'], y=sorted_data2['countrycode'])
plt.show()
sorted_data2
f,ax = plt.subplots(figsize =(20,10))
sns.pointplot(x='countrycode',y='failed',data=sorted_data2,color='red',alpha=0.8)
sns.pointplot(x='countrycode',y='successful',data=sorted_data2,color='lime',alpha=0.8)
#sns.pointplot(x='countrycode',y='live',data=sorted_data,color='blue',alpha=0.8)

plt.text(16.3,60000,'failed projects',color='red',fontsize = 18,style = 'italic')
plt.text(16,40000,'successful projects',color='lime',fontsize = 17,style = 'italic')
#plt.text(40,0.65,'live projects',color='blue',fontsize = 18,style = 'italic')

plt.xlabel('Country Code',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Successful Projects  vs  Failed Projects',fontsize = 20,color='blue',fontweight='bold')
plt.grid()
sorted_data2['failed']  = sorted_data2['failed'] / max(sorted_data2['failed'] ) 
sorted_data2['successful'] =  sorted_data2['successful'] / max(sorted_data2['successful'])
jplt = sns.jointplot(sorted_data2.failed, sorted_data2.successful, kind="kde", size=7)

plt.show()
sns.lmplot(x="successful", y="live", data=sorted_data2)
plt.show()
sorted_data.head()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=sorted_data, palette=pal, inner="points")
plt.show()
sorted_data2.corr()
                  
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(sorted_data2.corr(), annot=True, linewidths=1,linecolor="green", fmt= '.1f',ax=ax)
plt.show()