import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import sqlite3
import os
import re
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from gensim import corpora, models, similarities
output_notebook()
%matplotlib inline

import os
print(os.listdir("../input"))
conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
DataStats = pd.read_sql(
                       """
                        SELECT
                            Series.[Number of Indicies]
                            ,Country.[Number of Countries]
                        FROM (
                            SELECT 1 [idx]
                                ,count(*) [Number of Indicies]
                                ,NULL [Number of Countries]
                            FROM   Series  
                        ) Series
                        INNER JOIN (
                            SELECT 1 [idx]
                                ,NULL [Number of Indicies]
                                ,count(*) [Number of Countries]
                            FROM Country
                        ) Country
                            on Series.idx=Country.idx
                       """, con=conn)
print(DataStats)
metricsPerYear = pd.read_sql(
                       """
                        SELECT CountryName
                            ,count(IndicatorCode) [metricsPerYear]
                        FROM   Indicators
                        GROUP BY CountryCode, Year                  
                       """, con=conn)
metricsPerYear.hist(column='metricsPerYear', bins=50)
metricsPerCountry = pd.read_sql(
                       """
                        SELECT CountryName
                            ,count(distinct(IndicatorCode)) [metricsPerCountry]
                        FROM   Indicators
                        GROUP BY CountryCode                  
                        ORDER BY [metricsPerCountry] desc
                       """, con=conn)
metricsPerCountry.hist(column='metricsPerCountry', bins=50)
print(metricsPerCountry.head(20))
print(metricsPerCountry.tail(20))
PlayAround = pd.read_sql(
                       """
                       
                        SELECT Ind.CountryName
                            ,Ind.Year
                            ,Ind.Value
                            ,Ser.IndicatorName
                        FROM   Indicators Ind
                        INNER JOIN Series Ser
                            on Ser.SeriesCode=Ind.IndicatorCode
                        WHERE Ind.IndicatorCode = 'SM.POP.NETM'
                        and Ind.CountryCode = 'NAC'
                        
                       """, con=conn)
print(PlayAround.head(5))
plt.plot(PlayAround['Value'])

IndicatorSQLResults = pd.read_sql(
                       """
                        SELECT IndicatorName
                            ,IndicatorCode
                        FROM   Indicators
                       """, con=conn)
Indicator_array =  IndicatorSQLResults[['IndicatorName','IndicatorCode']].drop_duplicates().values

modified_indicators = []
unique_indicator_codes = []
for ele in Indicator_array:
    indicator = ele[0]
    indicator_code = ele[1].strip()
    if indicator_code not in unique_indicator_codes:
        # delete , ( ) from the IndicatorNames
        new_indicator = re.sub('[,()]',"",indicator).lower()
        # replace - with "to" and make all words into lower case
        new_indicator = re.sub('-'," to ",new_indicator).lower()
        modified_indicators.append([new_indicator,indicator_code])
        unique_indicator_codes.append(indicator_code)

Indicators = pd.DataFrame(modified_indicators,columns=['IndicatorName','IndicatorCode'])
Indicators = Indicators.drop_duplicates()

key_word_dict = {}
key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']
key_word_dict['Food'] = ['food','grain','nutrition','calories']
key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']
key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']
key_word_dict['Economy'] = ['income','gdp','gni','deficit','budget','market','stock','bond','infrastructure']
key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']
key_word_dict['Education'] = ['education','literacy']
key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']
key_word_dict['Rural'] = ['rural','village']
key_word_dict['Urban'] = ['urban','city']
feature = 'Food'
for indicator_ele in Indicators.values:
    for ele in key_word_dict[feature]:
        word_list = indicator_ele[0].split()
        if ele in word_list or ele+'s' in word_list:
            print(indicator_ele)
            break
# set a list of stop words to remove from the corpus, can always add to this list manually
stoplist = set('for a of the and to in [ on from per'.split())

#break down each line to a comma delimited list of words
texts = [[word for word in str(document).lower().split() if word not in stoplist] 
         for document in Indicators.values]

#generate a word frequency count
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

#filters out words that only show up once in the set of corpuses
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

# pretty-printer
from pprint import pprint  
pprint(texts[:5])


import operator
sorted_X = sorted(frequency.items(), key=operator.itemgetter(1),reverse=True)
sorted_X[:20]
dictionary = corpora.Dictionary(texts)
print(dictionary)
#use this to check the id of specific words in the dictionary
dictionary.token2id['working']
corpus = [dictionary.doc2bow(text) for text in texts]
pprint(corpus[:10])
pprint(Indicators.values[:10])
from gensim.test.utils import common_dictionary
from gensim.models import LsiModel

model = LsiModel(corpus,id2word=dictionary)
vectorized_corpus = model[corpus]
#key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']
#key_word_dict['Food'] = ['food','grain','nutrition','calories']
#key_word_dict['Trade'] = ['trade','import','export','good','shipping','shipment']
#key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']
#key_word_dict['Economy'] = ['income','gdp','gni','deficit','budget','market','stock','bond','infrastructure']
#key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']
#key_word_dict['Education'] = ['education','literacy']
#key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']
#key_word_dict['Rural'] = ['rural','village']
#key_word_dict['Urban'] = ['urban','city']

doc = "Trade"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi=model[vec_bow]
index = similarities.MatrixSimilarity(model[corpus])
sims = index[vec_lsi]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
pprint(sims[:10])
ct=0
for val in sims:
    if val[1] > 0.1:
        pprint(Indicators.values[val[0]][0])
        ct=ct+1
        if ct > 30:
            break