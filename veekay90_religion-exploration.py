# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#lets load the datasets



global_df= pd.read_csv('../input/global.csv')

regional_df= pd.read_csv('../input/regional.csv')



global_df.head()

#reducing the dataset 



cols= ['year','christianity_all','islam_all','hinduism_all','buddhism_all','noreligion_all','syncretism_all','christianity_percent','hinduism_percent','islam_percent','buddhism_percent','syncretism_percent','noreligion_percent']

global_df= global_df[cols]



fig= plt.figure(figsize= (10,8), facecolor= '#eeeeee')



for col in global_df.columns:

    if '_all' in col:

        ax= plt.plot(global_df.index, global_df[col],label= col, linewidth= 6)

        plt.legend(loc=2)

        plt.gca().legend_.remove()

        plt.title('Number of adherents over the years')

        plt.legend(bbox_to_anchor= (-.2,-.5,1.4,.5), mode= 'expand', loc=5,ncol=4,fontsize= 14)

        plt.xticks(size=15)



plt.figure(figsize= (8,8))

vals=[]

label= []

for col in global_df.columns:

    if '_percent' in col:

        vals.append(global_df[global_df.year==1945][col].values)

        label.append(col)

plt.pie(vals, labels= label, startangle= 0, autopct= '%.2f')

plt.title('Religions in 1945')

plt.show()
plt.figure(figsize= (8,8))

vals=[]

label= []

for col in global_df.columns:

    if '_percent' in col:

        vals.append(global_df[global_df.year==2010][col].values)

        label.append(col)

plt.pie(vals, labels= label,autopct= "%.2f", startangle= 0)

plt.title('Religions in 2010')

plt.show()

#exploring the regional dataframe



regional_df.head()
group= regional_df.groupby(['year','region'])



def area_plots(group):

    #christianity over the years

    group['christianity_all'].agg('sum').unstack().plot(kind='area',colormap = 'viridis', figsize= (6,6),grid= False, stacked= True)

    plt.title("Christianity", fontsize= 15)

    

    #hinduism 

    colormap= plt.cm.Reds

    group['hinduism_all'].agg('sum').unstack().plot(kind='area', colormap= colormap,figsize= (6,6), grid = False, stacked= True)

    plt.title('Hinduism')

    

    #islam

    colormap= plt.cm.Blues

    group['islam_all'].agg('sum').unstack().plot(kind='area', colormap= colormap,figsize= (6,6), grid = False, stacked= True)

    plt.title('ISLAM')

    

    #buddhism

    colormap= plt.cm.Greens

    group['buddhism_all'].agg('sum').unstack().plot(kind='area', colormap= colormap,figsize= (6,6), grid = False, stacked= True)

    plt.title('Buddhism')

    

    #syncretism

    colormap= plt.cm.inferno

    group['syncretism_all'].agg('sum').unstack().plot(kind='area', colormap= colormap,figsize= (6,6), grid = False, stacked= True)

    plt.title('Syncretism')

  

    #Atheism

    

    group['noreligion_all'].agg('sum').unstack().plot(kind='area', colormap= 'cubehelix',figsize= (6,6), grid = False, stacked= True)

    plt.title('Athiest')

    

    #population

    group['population'].agg('sum').unstack().plot(kind='area',figsize= (6,6), grid= False, stacked= True)

    plt.title('Population over the years')
area_plots(group)
plt.figure(figsize= (6,4))

sns.barplot(x= regional_df.year, y= regional_df[regional_df.region=='Europe']['population'],palette= 'Purples')

plt.title('European Population')



plt.figure(figsize= (6,4))

sns.barplot(x= regional_df.year, y= regional_df[regional_df.region=='Africa']['population'],palette= 'Reds')

plt.title('African Population')



plt.figure(figsize= (6,4))

sns.barplot(x= regional_df.year, y= regional_df[regional_df.region=='Mideast']['population'],palette= 'PuBuGn')

plt.title('Middle East Population')



plt.figure(figsize= (6,4))

sns.barplot(x= regional_df.year, y= regional_df[regional_df.region=='West. Hem']['population'],palette= 'Blues')

plt.title('Western Hemisphere Population')



plt.figure(figsize= (6,4))

sns.barplot(x= regional_df.year, y= regional_df[regional_df.region=='Asia']['population'],palette= 'Greens')

plt.title('Asian Population')
