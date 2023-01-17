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
_2015 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')

_2016 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')

_2017 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')

_2018 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')

_2019 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')

len(_2015.columns)

len(_2016.columns)
len(_2017.columns)
len(_2018.columns)
len(_2019.columns)
_2015.columns
_2016.columns
_2017.columns
_2018.columns
_2019.columns
import numpy as np

import pandas as pd
# Index(['Country', 'Region', 'Happiness Score',

#        'Health', 'Freedom','Corruption', 'Generosity'],

#       dtype='object')



_2015 = pd.read_csv('/kaggle/input/world-happiness/2015.csv')



_2015.drop('Happiness Rank', axis = 1, inplace = True) 

_2015.drop('Standard Error', axis = 1, inplace = True) 

_2015.drop('Economy (GDP per Capita)', axis = 1, inplace = True) 

_2015.drop('Family', axis = 1, inplace = True) 

_2015.drop('Dystopia Residual', axis = 1, inplace = True) 



_2015.columns = ['Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']

_2015
# Index(['Country', 'Region', 'Happiness Score',

#        'Health', 'Freedom','Corruption', 'Generosity'],

#       dtype='object')



_2016 = pd.read_csv('/kaggle/input/world-happiness/2016.csv')



_2016.drop('Happiness Rank', axis = 1, inplace = True) 

_2016.drop('Lower Confidence Interval', axis = 1, inplace = True) 

_2016.drop('Upper Confidence Interval', axis = 1, inplace = True) 

_2016.drop('Economy (GDP per Capita)', axis = 1, inplace = True) 

_2016.drop('Family', axis = 1, inplace = True) 

_2016.drop('Dystopia Residual', axis = 1, inplace = True) 



_2016.columns = ['Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']



_2016
# Index(['Country', 'Region', 'Happiness Score',

#        'Health', 'Freedom','Corruption', 'Generosity'],

#       dtype='object')



_2017 = pd.read_csv('/kaggle/input/world-happiness/2017.csv')



_2017.drop('Happiness.Rank', axis = 1, inplace = True) 

_2017.drop('Whisker.high', axis = 1, inplace = True) 

_2017.drop('Whisker.low', axis = 1, inplace = True) 

_2017.drop('Economy..GDP.per.Capita.', axis = 1, inplace = True) 

_2017.drop('Family', axis = 1, inplace = True) 

_2017.drop('Dystopia.Residual', axis = 1, inplace = True) 



#Region sutunu olmadigi icin onu eklememiz lazim

_2017['Region'] = '-'



#sutun adlarinin yerlerini degistirmemiz lazim

_2017 = _2017[['Country','Region', 'Happiness.Score', 'Health..Life.Expectancy.', 'Freedom','Trust..Government.Corruption.', 'Generosity']]



#sutun isimlerini degistirdik

_2017.columns = ['Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']



_2017
# Index(['Country', 'Region', 'Happiness Score',

#        'Health', 'Freedom','Corruption', 'Generosity'],

#       dtype='object')



_2018 = pd.read_csv('/kaggle/input/world-happiness/2018.csv')



_2018.drop('Overall rank', axis = 1, inplace = True) 

_2018.drop('Social support', axis = 1, inplace = True) 



_2018.drop('GDP per capita', axis = 1, inplace = True) 



# #Region sutunu olmadigi icin onu eklememiz lazim

_2018['Region'] = '-'



# #sutun adlarinin yerlerini degistirmemiz lazim

_2018 = _2018[['Country or region','Region', 'Score', 'Healthy life expectancy', 'Freedom to make life choices','Perceptions of corruption', 'Generosity']]



# #sutun isimlerini degistirdik

_2018.columns = ['Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']



_2018
# Index(['Country', 'Region', 'Happiness Score',

#        'Health', 'Freedom','Corruption', 'Generosity'],

#       dtype='object')



_2019 = pd.read_csv('/kaggle/input/world-happiness/2019.csv')



_2019.drop('Overall rank', axis = 1, inplace = True) 

_2019.drop('GDP per capita', axis = 1, inplace = True) 

_2019.drop('Social support', axis = 1, inplace = True) 



# #Region sutunu olmadigi icin onu eklememiz lazim

_2019['Region'] = '-'



# #sutun adlarinin yerlerini degistirmemiz lazim

_2019= _2019[['Country or region','Region', 'Score', 'Healthy life expectancy', 'Freedom to make life choices','Perceptions of corruption', 'Generosity']]



# #sutun isimlerini degistirdik

_2019.columns = ['Country','Region','Happiness Score','Health','Freedom','Corruption','Generosity']





_2019
_newdf = pd.concat([_2015,_2016,_2017,_2018,_2019], axis = 0,ignore_index=True) 

_newdf 




# _newdf2 = _newdf.groupby('Country').sum()



# _newdf2['Region'] = '-'



# _newdf2 = _newdf2[['Region','Happiness Score','Health','Freedom','Corruption','Generosity']]

# _newdf2



aggregation_functions = {'Happiness Score': 'sum', 'Health': 'sum','Freedom': 'sum','Corruption':'sum','Generosity':'sum', 'Region': 'first'}

_newdf2 = _newdf.groupby(_newdf['Country']).aggregate(aggregation_functions)



_newdf2.sort_values(by=['Happiness Score'],ascending=False) 







for i in _newdf2.Country:

    print(i)

    

        
