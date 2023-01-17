import pandas as pd

import numpy as np

import json

import math

import uuid

pd.set_option('display.max_columns', None)

import json

import os

from collections import Counter

from pandas import DataFrame

#!pip install wordcloud

#!pip install textblob



import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline  

import nltk

from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer

from wordcloud import WordCloud, STOPWORDS

from textblob import TextBlob    

stop = stopwords.words('english')



import pandas as pd

import numpy as np

from numpy import inf

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler



import xgboost as xgb



scaler = MinMaxScaler(feature_range=(0,1))



#!pip install eli5

import eli5
train_df = pd.read_csv('../input/tmdb-clean-dataset/train.csv')

test_df = pd.read_csv('../input/tmdb-clean-dataset/test.csv')



# Merging datasets to ensure things like encoding are created on a joint dataset

merged_df = train_df.append(test_df,ignore_index=True)
"""

Helper functions to explode json data into columns for the train and test datasets

"""



import uuid



def get_count_houses(data_column,col_name):

    

    ls = []

    data = [extract_meta_2(str(row),col_name) for row in data_column]

    data_2 = [house for production_houses in data for house in production_houses]

    house_count = Counter(data_2)

    

    return house_count





def extract_single_meta(json_data,col_name):

    if (json_data == 'nan'):

        return 'x'

    

    # Some text cleaning

    json_data = json_data.replace("'",'"')

    json_data = json_data.replace('None','"None"')

    json_data = json_data.replace('s n"','s n ')

    json_data = json_data.replace('ed"','ed ')

    json_data = json_data.replace('ld"','ld ')

    json_data = json_data.replace('n"s','ns' )

    json_data = json_data.replace('o"s','os' )

    json_data = json_data.replace('"s','s' )



    data = json.loads(json_data)

    meta = [d[col_name] for d in data]

    return meta[0]





def extract_meta(json_data,col_name):

    if (json_data == 'nan'):

        return 'x' 

    

    json_data = json_data.replace("'",'"')

    str_json_data = str(json_data)

    str_json_data = str_json_data.replace("l\"A","l")

    data = None

    data= json.loads(str_json_data)

    

    meta = [d[col_name] for d in data]

    

    return meta



def extract_meta_2(json_data,col_name):

    if (json_data == 'nan'):

        return 'x' 

    

    json_data = json_data.replace(u'\xa0',u' ')

    json_data = json_data.replace("{'",'{"')

    json_data = json_data.replace("'}",'"}')

    json_data = json_data.replace("':",'":')

    json_data = json_data.replace(": '",': "')

    json_data = json_data.replace("',",'",')

    json_data = json_data.replace(", '",', "')

    json_data = json_data.replace("'",'')

    

    str_json_data = str(json_data)

    data= None

    data= json.loads(str_json_data)

        

    meta = [d[col_name] for d in data]

    

    return meta

    





def extract_metadata(data_column, col_name, col_type = "blank"):

   

    # Extract all meta-genres

    all_meta = set()

    counter = Counter()

        

    if (col_type != 'production_companies' and col_type != 'keyword'):

        distinct_meta = [extract_meta(str(row),col_name) for row in data_column]

    else:

        counter = get_count_houses(data_column,col_name)

        counter = {k:counter[k] for k in counter if counter[k] > 5}

    

        distinct_meta = [extract_meta_2(str(row),col_name) for row in data_column]

        flattened_meta = [data for row in distinct_meta for data in row]

        

        distinct_meta = []

        for k in counter:

            if k in flattened_meta:

                distinct_meta.append(k)

    

    for data in distinct_meta:

        if (col_type != 'production_companies' and col_type != 'keyword'):

            all_meta.update(set(data))

        else:

            all_meta.update(set([data]))

        

    all_meta_dict = dict()

    i = 0

    for item in all_meta:

        all_meta_dict[item] = i

        i += 1

    all_meta_list = list(all_meta)

    

    return all_meta, all_meta_dict, all_meta_list





def add_meta_to_dataframe(all_meta_dict,all_meta_list, _df, data_column, col_name, col_type='blank'):



    if (col_type != 'production_companies' and col_type != 'keyword'):

        distinct_meta = [extract_meta(str(row),col_name) for row in data_column]

    else:

        distinct_meta = [extract_meta_2(str(row),col_name) for row in data_column]

        

    one_hot_np = np.empty([len(_df),len(all_meta_dict)])

    for i in range(0, len(_df)):

        one_hot_meta = np.zeros(len(all_meta_dict))

        for j in range(0,len(distinct_meta[i])):

            if (distinct_meta[i][j] in all_meta_dict):

                one_hot_meta[all_meta_dict[distinct_meta[i][j]]] = 1

        one_hot_np[i] = one_hot_meta

        

    one_hot_df = pd.DataFrame(one_hot_np)

    one_hot_df.columns = all_meta_list

    

    if (col_name == 'iso_639_1'):

        #one_hot_df.drop('id',axis=1,inplace=True)

        one_hot_df = one_hot_df['en'] 

    else:

        one_hot_df['sum' + col_name] = one_hot_df.sum(axis=1,skipna=False)

    

    # Final train df

    _df_new = pd.concat([_df.reset_index(drop=True),one_hot_df],axis=1)

    

    return _df_new





# Genres        

genres = merged_df['genres']

genres.fillna('nan',inplace=True)

all_meta, all_meta_dict, all_meta_list = extract_metadata(genres,"name")

genres = train_df['genres']

genres.fillna('nan',inplace=True)

train_df_new = add_meta_to_dataframe(all_meta_dict, all_meta_list, train_df, genres,"name")

genres_test = test_df['genres']

genres_test.fillna('nan',inplace=True)

test_df_new = add_meta_to_dataframe(all_meta_dict,all_meta_list,test_df,genres_test,"name")



# Spoken languages

spoken_languages = merged_df['spoken_languages']

spoken_languages.fillna('nan',inplace=True)

all_meta, all_meta_dict, all_meta_list = extract_metadata(spoken_languages,"iso_639_1")

spoken_languages = train_df['spoken_languages']

spoken_languages.fillna('nan',inplace=True)



train_df_new = add_meta_to_dataframe(all_meta_dict, all_meta_list, train_df_new, spoken_languages,"iso_639_1")





spoken_languages_test = test_df['spoken_languages']

spoken_languages_test.fillna('nan',inplace=True)

test_df_new = add_meta_to_dataframe(all_meta_dict,all_meta_list,test_df_new,spoken_languages_test,"iso_639_1")





# production countries

production_countries = merged_df['production_countries']

production_countries.fillna('nan',inplace=True)

all_meta, all_meta_dict, all_meta_list = extract_metadata(production_countries,"iso_3166_1")

production_countries = train_df['production_countries']

production_countries.fillna('nan',inplace=True)

train_df_new = add_meta_to_dataframe(all_meta_dict, all_meta_list, train_df_new, production_countries,"iso_3166_1")





production_countries_test = test_df['production_countries']

production_countries_test.fillna('nan',inplace=True)

test_df_new = add_meta_to_dataframe(all_meta_dict,all_meta_list,test_df_new,production_countries_test,"iso_3166_1")



"""

# Keywords

keywords = merged_df['Keywords']

keywords.fillna('nan',inplace=True)

all_meta, all_meta_dict, all_meta_list = extract_metadata(keywords,"name","keyword")

keywords = train_df['Keywords']

keywords.fillna('nan',inplace=True)

train_df_new = add_meta_to_dataframe(all_meta_dict, all_meta_list, train_df_new, keywords,"name","keyword")





keywords_test = test_df['Keywords']

keywords_test.fillna('nan',inplace=True)

test_df_new = add_meta_to_dataframe(all_meta_dict,all_meta_list,test_df_new,keywords_test,"name","keyword")

"""





# Belongs to collection

belongs_to_collection = train_df['belongs_to_collection']

belongs_to_collection.fillna('nan',inplace=True)

result = [extract_single_meta(str(row),'name') for row in belongs_to_collection]

result = pd.DataFrame(result)

result.columns = ['collection']

train_df_new = pd.concat([train_df_new.reset_index(drop=True),result],axis=1)





belongs_to_collection_test = test_df['belongs_to_collection']

belongs_to_collection_test.fillna('nan',inplace=True)

result = [extract_single_meta(str(row),'name') for row in belongs_to_collection_test]

result = pd.DataFrame(result)

result.columns = ['collection']

test_df_new = pd.concat([test_df_new.reset_index(drop=True),result],axis=1)



# production companies

production_companies = merged_df['production_companies']

production_companies.fillna('nan',inplace=True)

all_meta, all_meta_dict, all_meta_list = extract_metadata(production_companies,"name","production_companies")

production_companies = train_df['production_companies']

production_companies.fillna('nan',inplace=True)

train_df_new = add_meta_to_dataframe(all_meta_dict, all_meta_list, train_df_new, production_companies,"name","production_companies")





production_companies_test = test_df['production_companies']

production_companies_test.fillna('nan',inplace=True)

test_df_new = add_meta_to_dataframe(all_meta_dict,all_meta_list,test_df_new,production_companies_test,"name","production_companies")

"""

Extracts cast and crew information into train and test flat files csv 





"""

def extract_single_meta(json_data,col_name):

    if (json_data == 'nan'):

        return 'x'

    

    json_data = json_data.replace("'",'"')

    json_data = json_data.replace('None','"None"')

    json_data = json_data.replace('s n"','s n ')

    json_data = json_data.replace('ed"','ed ')

    json_data = json_data.replace('ld"','ld ')

    json_data = json_data.replace('n"s','ns' )

    json_data = json_data.replace('o"s','os' )

    json_data = json_data.replace('"s','s' )



    data = json.loads(json_data)

    meta = [d for d in data]

    return meta[0]





"""

Train dataset

"""



# Cast

imdb_id = train_df['imdb_id']

cast = train_df['cast']

main_ls = []

x = 0

i = 0

for item in cast:

    i+= 1

    ls = []

    

    if str(item) == '[]':

        main_ls.append(['-1','nan','nan','-1','-1','nan','10000','nan'])

        x += 1

        continue

    

    item = str(item)

    item = item.replace("', '","'; '")

    item = item.replace(", '","; '")

    item= item.replace("{","")

    item= item.replace("},",";")

    item= item.replace("[","")

    item= item.replace("]","")

    item= item.replace("'","")

    item = item.replace('Jimmy;,','Jimmy')

    item = item.replace('Elektra Quartet;','Elektra Quartet')

    

    item_ls = item.split(';')

    i = 0

    ls = []

    for it in item_ls:

        i += 1

        

        it = it.replace('[','')

        it = it.replace(']','')

        it_ls = it.split(':')

        ls.append(str(it_ls[1]).strip())

    

        if i % 8 == 0:

            #print (imdb_id[x])

            ls.append(imdb_id[x])

            main_ls.append(ls)

            ls = []

    x += 1

            

cast_df = DataFrame.from_records(main_ls)

cast_df.columns = ['cast_id','character','credit_id','gender','id','name','order','profile_path','imdb_id']



        

"""

Test dataset

"""

# Cast

imdb_id = test_df['imdb_id']

cast = test_df['cast']

main_ls = []

x = 0

i = 0

for item in cast:

    i+= 1

    ls = []

    

    if str(item) == '[]':

        main_ls.append(['-1','nan','nan','-1','-1','nan','10000','nan'])

        x += 1

        continue

    

    item = str(item)

    item = item.replace("', '","'; '")

    item = item.replace(", '","; '")

    item= item.replace("{","")

    item= item.replace("},",";")

    item= item.replace("[","")

    item= item.replace("]","")

    item= item.replace("'","")

    

    item = item.replace('Jimmy;,','Jimmy')

    item = item.replace('Elektra Quartet;','Elektra Quartet')

    

    item_ls = item.split(';')

    i = 0

    ls = []

    for it in item_ls:

        i += 1

        

        it = it.replace('[','')

        it = it.replace(']','')

        it_ls = it.split(':')

        ls.append(str(it_ls[1]).strip())

        

        if i % 8 == 0:

            #print (imdb_id[x])

            ls.append(imdb_id[x])

            main_ls.append(ls)

            ls = []

    x += 1



cast_df_test = DataFrame.from_records(main_ls)

cast_df_test.columns = ['cast_id','character','credit_id','gender','id','name','order','profile_path','imdb_id']



    
"""

Extract (explode) crew information for train and test datasets

"""





"""

Train data

"""

imdb_id = train_df['imdb_id']

cast = train_df['crew']

main_ls = []

x = 0

i = 0

for item in cast:

    i+= 1

    ls = []

    #print (item)

    

    if str(item) == '[]':

        main_ls.append(['nan','nan','nan','nan','nan','nan','nan'])

        x += 1

        continue

    

    item = str(item)

    item = item.replace("', '","'; '")

    item = item.replace(", '","; '")

    item= item.replace("{","")

    item= item.replace("},",";")

    item= item.replace("[","")

    item= item.replace("]","")

    item= item.replace("'","")

    

    #item = item.replace(",","")

    item = item.replace('Jimmy;,','Jimmy')

    item = item.replace('Elektra Quartet;','Elektra Quartet')

    

    item_ls = item.split(';')

    i = 0

    ls = []

    for it in item_ls:

        i += 1

        

        it = it.replace('[','')

        it = it.replace(']','')

        it_ls = it.split(':')

        ls.append(str(it_ls[1]).strip())

    

        if i % 7 == 0:

            #print (imdb_id[x])

            ls.append(imdb_id[x])

            main_ls.append(ls)

            ls = []

    x += 1



crew_df = DataFrame.from_records(main_ls)

crew_df.columns = ['credit_id','department','gender','id','job','name','profile_path','imdb_id']





"""

Test data

"""

imdb_id = test_df['imdb_id']

cast = test_df['crew']



main_ls = []

x = 0

i = 0

for item in cast:

    i+= 1

    ls = []

    #print (item)

    

    if str(item) == '[]':

        main_ls.append(['nan','nan','nan','nan','nan','nan','nan'])

        x += 1

        continue

    

    item = str(item)

    item = item.replace("', '","'; '")

    item = item.replace(", '","; '")

    item= item.replace("{","")

    item= item.replace("},",";")

    item= item.replace("[","")

    item= item.replace("]","")

    item= item.replace("'","")

    item = item.replace('Jimmy;,','Jimmy')

    item = item.replace('Elektra Quartet;','Elektra Quartet')

    

    item_ls = item.split(';')

    i = 0

    ls = []

    for it in item_ls:

        i += 1

        

        it = it.replace('[','')

        it = it.replace(']','')

        

        it_ls = it.split(':')

        ls.append(str(it_ls[1]).strip())

        

        if i % 7 == 0:

            #print (imdb_id[x])

            ls.append(imdb_id[x])

            main_ls.append(ls)

            ls = []

    x += 1



crew_df_test = DataFrame.from_records(main_ls)

crew_df_test.columns = ['credit_id','department','gender','id','job','name','profile_path','imdb_id']
"""

Merge cast features:

1) Count by movie

2) Count of gender by movie

3) Count of gender by cast/crew attributes & movie

4) count of top actors in a movie as defined by iMDB

5) gender of top 3 actors in the movie defined by order 

"""



"""

Train

"""



# Cast counts by movie

cast_counts = cast_df[['cast_id','imdb_id']].groupby('imdb_id').agg('count')

cast_counts.columns = ['cast_counts']

cast_counts.reset_index(inplace=True)





df_train= train_df_new.merge(cast_counts, how='left',left_on='imdb_id',right_on='imdb_id')







# Cast gender counts by movie

cast_gender_counts = cast_df[['cast_id', 'gender','imdb_id']].groupby(['imdb_id','gender'],as_index=False).agg('count')

cast_gender_counts = cast_gender_counts.pivot(index='imdb_id',columns='gender',values='cast_id')

cast_gender_counts.reset_index(inplace=True)

cast_gender_counts.fillna(0,inplace=True) # reasonable assumption

cast_gender_counts.columns=['imdb_id','gender_cast_count_0','gender_cast_count_1','gender_cast_count_2']

df_train = df_train.merge(cast_gender_counts,how='left',left_on='imdb_id',right_on='imdb_id')





# Get gender of top 3 actors

cast_gender_top_3 = cast_df[['order','gender','imdb_id']]

cast_gender_top_3['int_order'] = cast_gender_top_3.order.astype(int)

cast_gender_top_3 = cast_gender_top_3.loc[cast_gender_top_3['int_order'] <= 2,]

cast_gender_top_3.drop('order',inplace=True,axis=1)

cast_gender_top_3 = cast_gender_top_3.pivot_table(index='imdb_id',columns='int_order',values='gender',aggfunc='first')

cast_gender_top_3.reset_index(inplace=True)

cast_gender_top_3.columns=['imdb_id','cast_1_gender','cast_2_gender','cast_3_gender']

df_train = df_train.merge(cast_gender_top_3,how='left',left_on='imdb_id',right_on='imdb_id')





# Count of top actors per movie as defined by IMDB

top_actors = pd.read_csv('../input/top-1000-tmdb-actors/Top 1000 Actors and Actresses.csv',encoding='iso-8859-1')

top_actors_name = top_actors[['Name']]

cast_df = cast_df.merge(top_actors_name,how='left',left_on='name',right_on='Name')



print (cast_df.head(10))



cast_top_actors = cast_df[['imdb_id','Name']]

cast_top_actors = cast_top_actors.dropna()



cast_top_actors_2= cast_top_actors.groupby("imdb_id").count()

cast_top_actors_2.reset_index(inplace=True)

cast_top_actors_2.columns=['imdb_id','count_top_actors']

df_train = df_train.merge(cast_top_actors_2,how='left',left_on='imdb_id',right_on='imdb_id')



"""

Test

"""



# Cast counts by movie

cast_counts = cast_df_test[['cast_id','imdb_id']].groupby('imdb_id').agg('count')

cast_counts.columns = ['cast_counts']

cast_counts.reset_index(inplace=True)

df_test= test_df_new.merge(cast_counts, how='left',left_on='imdb_id',right_on='imdb_id')



# Cast gender counts by movie

cast_gender_counts = cast_df_test[['cast_id', 'gender','imdb_id']].groupby(['imdb_id','gender'],as_index=False).agg('count')

cast_gender_counts = cast_gender_counts.pivot(index='imdb_id',columns='gender',values='cast_id')

cast_gender_counts.reset_index(inplace=True)

cast_gender_counts.fillna(0,inplace=True) # reasonable assumption

cast_gender_counts.columns=['imdb_id','gender_cast_count_0','gender_cast_count_1','gender_cast_count_2']

df_test = df_test.merge(cast_gender_counts,how='left',left_on='imdb_id',right_on='imdb_id')





# Get gender of top 3 actors

cast_gender_top_3 = cast_df_test[['order','gender','imdb_id']]

cast_gender_top_3['int_order'] = cast_gender_top_3.order.astype(int)

cast_gender_top_3 = cast_gender_top_3.loc[cast_gender_top_3['int_order'] <= 2,]

cast_gender_top_3.drop('order',inplace=True,axis=1)

cast_gender_top_3 = cast_gender_top_3.pivot_table(index='imdb_id',columns='int_order',values='gender',aggfunc='first')

cast_gender_top_3.reset_index(inplace=True)

cast_gender_top_3.columns=['imdb_id','cast_1_gender','cast_2_gender','cast_3_gender']

df_test = df_test.merge(cast_gender_top_3,how='left',left_on='imdb_id',right_on='imdb_id')





# Count of top actors per movie as defined by IMDB

top_actors = pd.read_csv('../input/top-1000-tmdb-actors/Top 1000 Actors and Actresses.csv',encoding='iso-8859-1')

top_actors_name = top_actors[['Name']]

cast_df_test = cast_df_test.merge(top_actors_name,how='left',left_on='name',right_on='Name')



cast_top_actors = cast_df_test[['imdb_id','Name']]

cast_top_actors = cast_top_actors.dropna()



cast_top_actors_2= cast_top_actors.groupby("imdb_id").count()

cast_top_actors_2.reset_index(inplace=True)

cast_top_actors_2.columns=['imdb_id','count_top_actors']

df_test = df_test.merge(cast_top_actors_2,how='left',left_on='imdb_id',right_on='imdb_id')
"""

Merge crew features:

1) Count by movie

2) Count of gender by movie

3) Count of gender by cast/crew attributes & movie

4) count of top actors in a movie as defined by iMDB

5) gender of top 3 actors in the movie defined by order 

"""



"""

Train

"""



# Crew counts by movie

cast_counts = crew_df[['credit_id','imdb_id']].groupby('imdb_id').agg('count')

cast_counts.columns = ['crew_counts']

cast_counts.reset_index(inplace=True)

df_train= df_train.merge(cast_counts, how='left',left_on='imdb_id',right_on='imdb_id')

#df_train.drop('credit_id',axis=1,inplace=True)



# Crew gender counts by movie

cast_gender_counts = crew_df[['credit_id', 'gender','imdb_id']].groupby(['imdb_id','gender'],as_index=False).agg('count')

cast_gender_counts = cast_gender_counts.pivot(index='imdb_id',columns='gender',values='credit_id')

cast_gender_counts.reset_index(inplace=True)

cast_gender_counts.fillna(0,inplace=True) # reasonable assumption

cast_gender_counts.columns=['imdb_id','gender_crew_count_0','gender_crew_count_1','gender_crew_count_2']

df_train = df_train.merge(cast_gender_counts,how='left',left_on='imdb_id',right_on='imdb_id')



# Crew gender counts by movie and dept

cast_gender_counts = crew_df[['credit_id', 'gender','imdb_id','department']].groupby(['imdb_id','gender','department'],as_index=False).agg('count')

cast_gender_counts = cast_gender_counts.pivot_table(index='imdb_id',columns=['gender','department'],values='credit_id')

cast_gender_counts.reset_index(inplace=True)

cast_gender_counts.fillna(0,inplace=True) # reasonable assumption

df_train = df_train.merge(cast_gender_counts,how='left',left_on='imdb_id',right_on='imdb_id')



# Crew job counts by movie

crew_job_counts = crew_df[['credit_id', 'job','imdb_id']].groupby(['imdb_id','job'],as_index=False).agg('count')

crew_job_counts = crew_job_counts.pivot(index='imdb_id',columns='job',values='credit_id')

crew_job_counts.reset_index(inplace=True)

crew_job_counts.fillna(0,inplace=True) # reasonable assumption

df_train = df_train.merge(crew_job_counts,how='left',left_on='imdb_id',right_on='imdb_id')



# Crew department counts by movie

crew_dept_counts = crew_df[['credit_id', 'department','imdb_id']].groupby(['imdb_id','department'],as_index=False).agg('count')

crew_dept_counts = crew_dept_counts.pivot(index='imdb_id',columns='department',values='credit_id')

crew_dept_counts.reset_index(inplace=True)

crew_dept_counts.fillna(0,inplace=True) # reasonable assumption

df_train = df_train.merge(crew_dept_counts,how='left',left_on='imdb_id',right_on='imdb_id')



"""

Test

"""





# Crew counts by movie

cast_counts = crew_df_test[['credit_id','imdb_id']].groupby('imdb_id').agg('count')

cast_counts.columns = ['crew_counts']

cast_counts.reset_index(inplace=True)

df_test= df_test.merge(cast_counts, how='left',left_on='imdb_id',right_on='imdb_id')

#df_test.drop('credit_id',axis=1,inplace=True)



# Crew gender counts by movie

cast_gender_counts = crew_df_test[['credit_id', 'gender','imdb_id']].groupby(['imdb_id','gender'],as_index=False).agg('count')

cast_gender_counts = cast_gender_counts.pivot(index='imdb_id',columns='gender',values='credit_id')

cast_gender_counts.reset_index(inplace=True)

cast_gender_counts.fillna(0,inplace=True) # reasonable assumption

cast_gender_counts.columns=['imdb_id','gender_crew_count_0','gender_crew_count_1','gender_crew_count_2']

df_test = df_test.merge(cast_gender_counts,how='left',left_on='imdb_id',right_on='imdb_id')



# Crew gender counts by movie and dept

cast_gender_counts = crew_df_test[['credit_id', 'gender','imdb_id','department']].groupby(['imdb_id','gender','department'],as_index=False).agg('count')

cast_gender_counts = cast_gender_counts.pivot_table(index='imdb_id',columns=['gender','department'],values='credit_id')

cast_gender_counts.reset_index(inplace=True)

cast_gender_counts.fillna(0,inplace=True) # reasonable assumption

df_test = df_test.merge(cast_gender_counts,how='left',left_on='imdb_id',right_on='imdb_id')



# Crew job counts by movie

crew_job_counts = crew_df_test[['credit_id', 'job','imdb_id']].groupby(['imdb_id','job'],as_index=False).agg('count')

crew_job_counts = crew_job_counts.pivot(index='imdb_id',columns='job',values='credit_id')

crew_job_counts.reset_index(inplace=True)

crew_job_counts.fillna(0,inplace=True) # reasonable assumption

df_test = df_test.merge(crew_job_counts,how='left',left_on='imdb_id',right_on='imdb_id')



# Crew department counts by movie

crew_dept_counts = crew_df_test[['credit_id', 'department','imdb_id']].groupby(['imdb_id','department'],as_index=False).agg('count')

crew_dept_counts = crew_dept_counts.pivot(index='imdb_id',columns='department',values='credit_id')

crew_dept_counts.reset_index(inplace=True)

crew_dept_counts.fillna(0,inplace=True) # reasonable assumption

df_test = df_test.merge(crew_dept_counts,how='left',left_on='imdb_id',right_on='imdb_id')



"""

Get top characters and their actors that appeared at least 5 times in the dataset

"""



character = cast_df['character']

cast = cast_df['name']



character.fillna('x',inplace=True)

cast.fillna('x',inplace=True)

character_counts = Counter(character)

cast_counts = Counter(cast)



counter_char = {x: character_counts[x] for x in character_counts if character_counts[x] >= 40}

counter_cast = {x: cast_counts[x] for x in cast_counts if cast_counts[x] >= 40}



characters = [x for x in counter_char]

casts = [x for x in counter_cast]



cast_top = cast_df[cast_df['name'].isin(casts)]

cast_top_counts = cast_top[['imdb_id','name','cast_id']].groupby(['imdb_id','name']).agg('count')

cast_top_counts.reset_index(inplace=True)





char_top = cast_df[cast_df['character'].isin(characters)]

char_top_counts = char_top[['imdb_id','character','cast_id']].groupby(['imdb_id','character']).agg('count')

char_top_counts.reset_index(inplace=True)





cast_top_test = cast_df_test[cast_df_test['name'].isin(casts)]

cast_top_test_counts = cast_top_test[['imdb_id','name','cast_id']].groupby(['imdb_id','name']).agg('count')

cast_top_test_counts.reset_index(inplace=True)





char_top_test = cast_df_test[cast_df_test['character'].isin(characters)]

char_top_test_counts = char_top_test[['imdb_id','character','cast_id']].groupby(['imdb_id','character']).agg('count')

char_top_test_counts.reset_index(inplace=True)





cast_top_counts = cast_top_counts.pivot(index='imdb_id',columns='name',values='cast_id')

cast_top_counts.fillna(0,inplace=True)



cast_top_counts.reset_index(inplace=True)

cast_top_test_counts = cast_top_test_counts.pivot(index='imdb_id',columns='name',values='cast_id')

cast_top_test_counts.fillna(0,inplace=True)



cast_top_test_counts.reset_index(inplace=True)



char_top_counts = char_top_counts.pivot(index='imdb_id',columns='character',values='cast_id')

char_top_counts.fillna(0,inplace=True)



char_top_counts.reset_index(inplace=True)



char_top_test_counts = char_top_test_counts.pivot(index='imdb_id',columns='character',values='cast_id')

char_top_test_counts.fillna(0,inplace=True)



char_top_test_counts.reset_index(inplace=True)



df_train = df_train.merge(char_top_counts,how='left',left_on='imdb_id',right_on='imdb_id')

df_train = df_train.merge(cast_top_counts,how='left',left_on='imdb_id',right_on='imdb_id')

df_train.fillna(0,inplace=True)





df_test = df_test.merge(char_top_test_counts,how='left',left_on='imdb_id',right_on='imdb_id')

df_test = df_test.merge(cast_top_test_counts,how='left',left_on='imdb_id',right_on='imdb_id')

df_test.fillna(0,inplace=True)
"""

Get crew that appeared at least 5 times in the dataset

"""



#character = crew_df['job']

cast = crew_df['name']



#character.fillna('x',inplace=True)

cast.fillna('x',inplace=True)

#character_counts = Counter(character)

cast_counts = Counter(cast)



#counter_char = {x: character_counts[x] for x in character_counts if character_counts[x] >= 15}

counter_cast = {x: cast_counts[x] for x in cast_counts if cast_counts[x] >= 40}



#characters = [x for x in counter_char]

casts = [x for x in counter_cast]



cast_top = crew_df[crew_df['name'].isin(casts)]

cast_top_counts = cast_top[['imdb_id','name','credit_id']].groupby(['imdb_id','name']).agg('count')

cast_top_counts.reset_index(inplace=True)



#char_top = crew_df[crew_df['job'].isin(characters)]

#char_top_counts = char_top[['imdb_id','job','credit_id']].groupby(['imdb_id','job']).agg('count')

#char_top_counts.reset_index(inplace=True)



cast_top_test = crew_df_test[crew_df_test['name'].isin(casts)]

cast_top_test_counts = cast_top_test[['imdb_id','name','credit_id']].groupby(['imdb_id','name']).agg('count')

cast_top_test_counts.reset_index(inplace=True)



#char_top_test = crew_df_test[crew_df_test['job'].isin(characters)]

#char_top_test_counts = char_top_test[['imdb_id','job','credit_id']].groupby(['imdb_id','job']).agg('count')

#char_top_test_counts.reset_index(inplace=True)



cast_top_counts = cast_top_counts.pivot(index='imdb_id',columns='name',values='credit_id')

cast_top_counts.fillna(0,inplace=True)

cast_top_counts.reset_index(inplace=True)

cast_top_test_counts = cast_top_test_counts.pivot(index='imdb_id',columns='name',values='credit_id')

cast_top_test_counts.fillna(0,inplace=True)

cast_top_test_counts.reset_index(inplace=True)





#char_top_counts = char_top_counts.pivot(index='imdb_id',columns='job',values='credit_id')

#char_top_counts.fillna(0,inplace=True)

#char_top_counts.reset_index(inplace=True)

#char_top_test_counts = char_top_test_counts.pivot(index='imdb_id',columns='job',values='credit_id')

#char_top_test_counts.fillna(0,inplace=True)

#char_top_test_counts.reset_index(inplace=True)



#df_train = df_train.merge(char_top_counts,how='left',left_on='imdb_id',right_on='imdb_id')

df_train = df_train.merge(cast_top_counts,how='left',left_on='imdb_id',right_on='imdb_id')

df_train.fillna(0,inplace=True)





#df_test = df_test.merge(char_top_test_counts,how='left',left_on='imdb_id',right_on='imdb_id')

df_test = df_test.merge(cast_top_test_counts,how='left',left_on='imdb_id',right_on='imdb_id')

df_test.fillna(0,inplace=True)
df_train['overview'].fillna('',inplace=True)

df_test['overview'].fillna('',inplace=True)



df_train['len_original_title'] = df_train.original_title.apply(str).apply(len)

df_test['len_original_title'] = df_test.original_title.apply(len)



df_train['len_overview'] = df_train.overview.apply(str).apply(len)

df_test['len_overview'] = df_test.overview.apply(str).apply(len)



df_train['len_tagline'] = df_train.tagline.apply(str).apply(len)

df_test['len_tagline'] = df_test.tagline.apply(str).apply(len)



df_train['len_title'] = df_train.title.apply(str).apply(len)

df_test['len_title'] = df_test.title.apply(str).apply(len)



df_train['num_words_title'] = df_train.title.apply(lambda x: len(str(x).split(' ')))

df_test['num_words_title'] = df_test.title.apply(lambda x: len(str(x).split(' ')))



df_train['num_words_original_title'] = df_train.original_title.apply(lambda x: len(str(x).split(' ')))

df_test['num_words_original_title'] = df_test.original_title.apply(lambda x: len(str(x).split(' ')))



df_train['num_words_tagline'] = df_train.tagline.apply(lambda x: len(str(x).split(' ')))

df_test['num_words_tagline'] = df_test.tagline.apply(lambda x: len(str(x).split(' ')))



df_train['num_words_overview'] = df_train.overview.apply(lambda x: len(str(x).split(' ')))

df_test['num_words_overview'] = df_test.overview.apply(lambda x: len(str(x).split(' ')))
def senti(x):

    return TextBlob(x).sentiment



overview_train = df_train['overview']

overview_test = df_test['overview']

tagline_train = df_train['tagline']

tagline_test = df_test['tagline']



overview_train = overview_train.apply(lambda x: " ".join(str(x).lower() for x in str(x).split()))

overview_train = overview_train.str.replace('[^\w\s]','')

overview_test = overview_test.apply(lambda x: " ".join(str(x).lower() for x in str(x).split()))

overview_test = overview_test.str.replace('[^\w\s]','')



tagline_train = tagline_train.apply(lambda x: " ".join(str(x).lower() for x in str(x).split()))

tagline_train = tagline_train.str.replace('[^\w\s]','')

tagline_test = tagline_test.apply(lambda x: " ".join(str(x).lower() for x in str(x).split()))

tagline_test = tagline_test.str.replace('[^\w\s]','')





overview_train = overview_train.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))

overview_test = overview_test.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))

tagline_train = tagline_train.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))

tagline_test = tagline_test.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))





st = PorterStemmer()

overview_train = overview_train.apply(lambda x: " ".join([st.stem(word) for word in str(x).split()]))

overview_test = overview_test.apply(lambda x: " ".join([st.stem(word) for word in str(x).split()]))

tagline_train = tagline_train.apply(lambda x: " ".join([st.stem(word) for word in str(x).split()]))

tagline_test = tagline_test.apply(lambda x: " ".join([st.stem(word) for word in str(x).split()]))





overview_train_senti = overview_train.apply(senti)

overview_test_senti = overview_test.apply(senti)



tagline_train_senti = tagline_train.apply(senti)

tagline_test_senti = tagline_test.apply(senti)



df_train['overview_senti_1'] = overview_train_senti.apply(lambda x: x[0])

df_train['overview_senti_2'] = overview_train_senti.apply(lambda x: x[1])

df_train['tagline_senti_1'] = tagline_train_senti.apply(lambda x: x[0])

df_train['tagline_senti_2'] = tagline_train_senti.apply(lambda x: x[1])



df_test['overview_senti_1'] = overview_test_senti.apply(lambda x: x[0])

df_test['overview_senti_2'] = overview_test_senti.apply(lambda x: x[1])

df_test['tagline_senti_1'] = tagline_test_senti.apply(lambda x: x[0])

df_test['tagline_senti_2'] = tagline_test_senti.apply(lambda x: x[1])
# creating features based on dates

def process_date(df):

    date_parts = ["year", "weekday", "month", 'weekofyear', 'day', 'quarter']

    for part in date_parts:

        part_col = 'release_date' + "_" + part

        df[part_col] = getattr(df['release_date_2'].dt, part).astype(int)

    

    return df





def fix_date(x):

    """

    Fixes dates which are in 20xx

    """

    

    

    year = x.split('/')[2]

    

    if len(year) == 2:

        if int(year) <= 19:

            return x[:-2] + '20' + year

        else:

            return x[:-2] + '19' + year

    else:

        return x

    

    

df_train['release_date_2'] = df_train['release_date'].apply(lambda x: fix_date(x))

df_test['release_date_2'] = df_test['release_date'].apply(lambda x: fix_date(x) if x != 0 else np.NaN)

df_train['release_date_2'] = pd.to_datetime(df_train['release_date'])

df_test['release_date_2'] = pd.to_datetime(df_test['release_date'])





df_train = process_date(df_train)

df_test = process_date(df_test)



df_train.drop('release_date_2',axis=1,inplace=True)

df_test.drop('release_date_2',axis=1,inplace=True)
train_sub = df_train[['id','collection','release_date_year']]

train_sub = train_sub[train_sub['collection'] != 'x']







train_sub['movie_rank'] = train_sub.groupby('collection')['release_date_year'].rank(ascending=True)



train_sub.drop('collection',inplace=True,axis=1)

train_sub.drop('release_date_year',inplace=True,axis=1)





df_train = df_train.merge(train_sub,how='left',left_on='id',right_on='id')





test_sub = df_test[['id','collection','release_date_year']]

test_sub = test_sub[test_sub['collection'] != 'x']



test_sub['movie_rank'] = test_sub.groupby('collection')['release_date_year'].rank(ascending=True)

test_sub.drop('collection',inplace=True,axis=1)

test_sub.drop('release_date_year',inplace=True,axis=1)

df_test = df_test.merge(test_sub,how='left',left_on='id',right_on='id')



df_train.loc[df_train.collection == 'x','movie_rank'] = np.NaN 

df_test.loc[df_test.collection == 'x','movie_rank'] = np.NaN
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

le2 = LabelEncoder()

le3 = LabelEncoder()



merged_collection = pd.concat([df_train.collection,df_test.collection])

merged_language = merged_df['original_language']



le.fit(merged_collection)

le2.fit(merged_language)



train_collection = df_train['collection']

test_collection = df_test['collection']



train_language = df_train['original_language']

test_language = df_test['original_language']



train_collection_labels = le.transform(train_collection)

test_collection_labels = le.transform(test_collection)



train_language_labels = le2.transform(train_language)

test_language_labels = le2.transform(test_language)



df_train['collection_encoded'] = train_collection_labels

df_test['collection_encoded'] = test_collection_labels

df_train['original_language_encoded'] = train_language_labels

df_test['original_language_encoded'] = test_language_labels
import collections, re

from nltk.corpus import stopwords



stop_english = stopwords.words('english')

ls_all_words = []

ls_all_words_tagline = []

count_words_param = 30





"""

Train

"""



#dict_corpus_overview = {}

corpus_overview = df_train['overview']

corpus_overview_list = corpus_overview.tolist()



corpus_overview = [str(re.sub(' +',' ',str(txt))) for txt in corpus_overview_list]

corpus_overview = [str(re.sub('[^A-Za-z0-9\' ]+','',str(txt))) for txt in corpus_overview]

corpus_overview = [str(txt).lower() for txt in corpus_overview]

bagofwords = [collections.Counter(re.findall(r'\w+',txt)) for txt in corpus_overview]

sumbags = sum(bagofwords,collections.Counter())

sumbags_filter =  {x : sumbags[x] for x in sumbags if sumbags[x] >= count_words_param} # words that appeared at least x times in corpus





for ele in sumbags_filter:

    if (ele not in stop_english and ele not in ls_all_words):

        ls_all_words.append(ele)



list_index = list(range(len(df_train)))

df = pd.DataFrame(0, index=list_index, columns=ls_all_words)



i = 0

for item in bagofwords:

    for ele in item:

        if (ele in ls_all_words):

            df.set_value(i,ele,item[ele])

    i += 1



    

df_train = pd.concat([df_train.reset_index(drop=True),df],axis=1)





corpus_overview = df_train['tagline']

corpus_overview_list = corpus_overview.tolist()



corpus_overview = [str(re.sub(' +',' ',str(txt))) for txt in corpus_overview_list]

corpus_overview = [str(re.sub('[^A-Za-z0-9\' ]+','',str(txt))) for txt in corpus_overview]

corpus_overview = [str(txt).lower() for txt in corpus_overview]

bagofwords = [collections.Counter(re.findall(r'\w+',txt)) for txt in corpus_overview]

sumbags = sum(bagofwords,collections.Counter())

sumbags_filter =  {x : sumbags[x] for x in sumbags if sumbags[x] >= count_words_param} # words that appeared at least x times in corpus



for ele in sumbags_filter:

    if (ele not in stop_english and ele not in ls_all_words_tagline):

        ls_all_words_tagline.append(ele)



list_index = list(range(len(df_train)))

df = pd.DataFrame(0, index=list_index, columns=ls_all_words_tagline)



i = 0

for item in bagofwords:

    for ele in item:

        if (ele in ls_all_words_tagline):

            df.set_value(i,ele,item[ele])

    i += 1



    

df_train = pd.concat([df_train.reset_index(drop=True),df],axis=1)





"""

Test

""" 



corpus_overview_test = df_test['overview']

corpus_overview_test_list = corpus_overview_test.tolist()



corpus_overview_test = [str(re.sub(' +',' ',str(txt))) for txt in corpus_overview_test_list]

corpus_overview_test = [str(re.sub('[^A-Za-z0-9\' ]+','',str(txt))) for txt in corpus_overview_test]

corpus_overview_test = [str(txt).lower() for txt in corpus_overview_test]

bagofwords = [collections.Counter(re.findall(r'\w+',txt)) for txt in corpus_overview_test]

sumbags = sum(bagofwords,collections.Counter())

sumbags_filter =  {x : sumbags[x] for x in sumbags if sumbags[x] >= count_words_param} # words that appeared at least x times in corpus



#for ele in sumbags_filter:

#    if (ele not in stop_english and ele not in ls_all_words):

#        ls_all_words.append(ele)



list_index = list(range(len(df_test)))

df = pd.DataFrame(0, index=list_index, columns=ls_all_words) # reuse training words only



i = 0

for item in bagofwords:

    for ele in item:

        if (ele in ls_all_words):

            df.set_value(i,ele,item[ele])

    i += 1





df_test = pd.concat([df_test.reset_index(drop=True),df],axis=1)



corpus_overview_test = df_test['tagline']

corpus_overview_test_list = corpus_overview_test.tolist()



corpus_overview_test = [str(re.sub(' +',' ',str(txt))) for txt in corpus_overview_test_list]

corpus_overview_test = [str(re.sub('[^A-Za-z0-9\' ]+','',str(txt))) for txt in corpus_overview_test]

corpus_overview_test = [str(txt).lower() for txt in corpus_overview_test]

bagofwords = [collections.Counter(re.findall(r'\w+',txt)) for txt in corpus_overview_test]

sumbags = sum(bagofwords,collections.Counter())

sumbags_filter =  {x : sumbags[x] for x in sumbags if sumbags[x] >= count_words_param} # words that appeared at least x times in corpus



#for ele in sumbags_filter:

#    if (ele not in stop_english and ele not in ls_all_words):

#        ls_all_words.append(ele)



list_index = list(range(len(df_test)))

df = pd.DataFrame(0, index=list_index, columns=ls_all_words_tagline) # reuse training words only



i = 0

for item in bagofwords:

    for ele in item:

        if (ele in ls_all_words_tagline):

            df.set_value(i,ele,item[ele])

    i += 1





df_test = pd.concat([df_test.reset_index(drop=True),df],axis=1)
df_test.loc[df_test['id'] == 3033,'budget'] = 250 

df_test.loc[df_test['id'] == 3051,'budget'] = 50

df_test.loc[df_test['id'] == 3084,'budget'] = 337

df_test.loc[df_test['id'] == 3224,'budget'] = 4  

df_test.loc[df_test['id'] == 3594,'budget'] = 25  

df_test.loc[df_test['id'] == 3619,'budget'] = 500  

df_test.loc[df_test['id'] == 3831,'budget'] = 3  

df_test.loc[df_test['id'] == 3935,'budget'] = 500  

df_test.loc[df_test['id'] == 4049,'budget'] = 995946 

df_test.loc[df_test['id'] == 4424,'budget'] = 3  

df_test.loc[df_test['id'] == 4460,'budget'] = 8  

df_test.loc[df_test['id'] == 4555,'budget'] = 1200000 

df_test.loc[df_test['id'] == 4624,'budget'] = 30 

df_test.loc[df_test['id'] == 4645,'budget'] = 500 

df_test.loc[df_test['id'] == 4709,'budget'] = 450 

df_test.loc[df_test['id'] == 4839,'budget'] = 7

df_test.loc[df_test['id'] == 3125,'budget'] = 25 

df_test.loc[df_test['id'] == 3142,'budget'] = 1

df_test.loc[df_test['id'] == 3201,'budget'] = 450

df_test.loc[df_test['id'] == 3222,'budget'] = 6

df_test.loc[df_test['id'] == 3545,'budget'] = 38

df_test.loc[df_test['id'] == 3670,'budget'] = 18

df_test.loc[df_test['id'] == 3792,'budget'] = 19

df_test.loc[df_test['id'] == 3881,'budget'] = 7

df_test.loc[df_test['id'] == 3969,'budget'] = 400

df_test.loc[df_test['id'] == 4196,'budget'] = 6

df_test.loc[df_test['id'] == 4221,'budget'] = 11

df_test.loc[df_test['id'] == 4222,'budget'] = 500

df_test.loc[df_test['id'] == 4285,'budget'] = 11

df_test.loc[df_test['id'] == 4319,'budget'] = 1

df_test.loc[df_test['id'] == 4639,'budget'] = 10

df_test.loc[df_test['id'] == 4719,'budget'] = 45

df_test.loc[df_test['id'] == 4822,'budget'] = 22

df_test.loc[df_test['id'] == 4829,'budget'] = 20

df_test.loc[df_test['id'] == 4969,'budget'] = 20

df_test.loc[df_test['id'] == 5021,'budget'] = 40 

df_test.loc[df_test['id'] == 5035,'budget'] = 1 

df_test.loc[df_test['id'] == 5063,'budget'] = 14 

df_test.loc[df_test['id'] == 5119,'budget'] = 2 

df_test.loc[df_test['id'] == 5214,'budget'] = 30 

df_test.loc[df_test['id'] == 5221,'budget'] = 50 

df_test.loc[df_test['id'] == 4903,'budget'] = 15

df_test.loc[df_test['id'] == 4983,'budget'] = 3

df_test.loc[df_test['id'] == 5102,'budget'] = 28

df_test.loc[df_test['id'] == 5217,'budget'] = 75

df_test.loc[df_test['id'] == 5224,'budget'] = 3 

df_test.loc[df_test['id'] == 5469,'budget'] = 20 

df_test.loc[df_test['id'] == 5840,'budget'] = 1 

df_test.loc[df_test['id'] == 5960,'budget'] = 30

df_test.loc[df_test['id'] == 6506,'budget'] = 11 

df_test.loc[df_test['id'] == 6553,'budget'] = 280

df_test.loc[df_test['id'] == 6561,'budget'] = 7

df_test.loc[df_test['id'] == 6582,'budget'] = 218

df_test.loc[df_test['id'] == 6638,'budget'] = 5

df_test.loc[df_test['id'] == 6749,'budget'] = 8 

df_test.loc[df_test['id'] == 6759,'budget'] = 50 

df_test.loc[df_test['id'] == 6856,'budget'] = 10

df_test.loc[df_test['id'] == 6858,'budget'] =  100

df_test.loc[df_test['id'] == 6876,'budget'] =  250

df_test.loc[df_test['id'] == 6972,'budget'] = 1

df_test.loc[df_test['id'] == 7079,'budget'] = 8000000

df_test.loc[df_test['id'] == 7150,'budget'] = 118

df_test.loc[df_test['id'] == 6506,'budget'] = 118

df_test.loc[df_test['id'] == 7225,'budget'] = 6

df_test.loc[df_test['id'] == 7231,'budget'] = 85

df_test.loc[df_test['id'] == 5222,'budget'] = 5

df_test.loc[df_test['id'] == 5322,'budget'] = 90

df_test.loc[df_test['id'] == 5350,'budget'] = 70

df_test.loc[df_test['id'] == 5378,'budget'] = 10

df_test.loc[df_test['id'] == 5545,'budget'] = 80

df_test.loc[df_test['id'] == 5810,'budget'] = 8

df_test.loc[df_test['id'] == 5926,'budget'] = 300

df_test.loc[df_test['id'] == 5927,'budget'] = 4

df_test.loc[df_test['id'] == 5986,'budget'] = 1

df_test.loc[df_test['id'] == 6053,'budget'] = 20

df_test.loc[df_test['id'] == 6104,'budget'] = 1

df_test.loc[df_test['id'] == 6130,'budget'] = 30

df_test.loc[df_test['id'] == 6301,'budget'] = 150

df_test.loc[df_test['id'] == 6276,'budget'] = 100

df_test.loc[df_test['id'] == 6473,'budget'] = 100

df_test.loc[df_test['id'] == 6842,'budget'] = 30



df_train.loc[df_train['id'] == 90,'budget'] = 30000000                  

df_train.loc[df_train['id'] == 118,'budget'] = 60000000       

df_train.loc[df_train['id'] == 149,'budget'] = 18000000       

df_train.loc[df_train['id'] == 464,'budget'] = 20000000       

df_train.loc[df_train['id'] == 470,'budget'] = 13000000       

df_train.loc[df_train['id'] == 513,'budget'] = 930000         

df_train.loc[df_train['id'] == 797,'budget'] = 8000000        

df_train.loc[df_train['id'] == 819,'budget'] = 90000000       

df_train.loc[df_train['id'] == 850,'budget'] = 90000000       

df_train.loc[df_train['id'] == 1007,'budget'] = 2              

df_train.loc[df_train['id'] == 1112,'budget'] = 7500000       

df_train.loc[df_train['id'] == 1131,'budget'] = 4300000        

df_train.loc[df_train['id'] == 1359,'budget'] = 10000000       

df_train.loc[df_train['id'] == 1542,'budget'] = 1             

df_train.loc[df_train['id'] == 1570,'budget'] = 15800000       

df_train.loc[df_train['id'] == 1571,'budget'] = 4000000        

df_train.loc[df_train['id'] == 1714,'budget'] = 46000000       

df_train.loc[df_train['id'] == 1721,'budget'] = 17500000       

   

df_train.loc[df_train['id'] == 1885,'budget'] = 12             

df_train.loc[df_train['id'] == 2091,'budget'] = 10             

df_train.loc[df_train['id'] == 2268,'budget'] = 17500000       

df_train.loc[df_train['id'] == 2491,'budget'] = 6              

df_train.loc[df_train['id'] == 2602,'budget'] = 31000000       

df_train.loc[df_train['id'] == 2612,'budget'] = 15000000       

df_train.loc[df_train['id'] == 2696,'budget'] = 10000000      

df_train.loc[df_train['id'] == 2801,'budget'] = 10000000       

df_train.loc[df_train['id'] == 335,'budget'] = 2 

df_train.loc[df_train['id'] == 348,'budget'] = 12

df_train.loc[df_train['id'] == 470,'budget'] = 13000000 

df_train.loc[df_train['id'] == 513,'budget'] = 1100000

df_train.loc[df_train['id'] == 640,'budget'] = 6 

df_train.loc[df_train['id'] == 696,'budget'] = 1

df_train.loc[df_train['id'] == 797,'budget'] = 8000000 

df_train.loc[df_train['id'] == 850,'budget'] = 1500000

df_train.loc[df_train['id'] == 1199,'budget'] = 5 

df_train.loc[df_train['id'] == 1282,'budget'] = 9              

df_train.loc[df_train['id'] == 1347,'budget'] = 1

df_train.loc[df_train['id'] == 1755,'budget'] = 2

df_train.loc[df_train['id'] == 1801,'budget'] = 5

df_train.loc[df_train['id'] == 1918,'budget'] = 592 

df_train.loc[df_train['id'] == 2033,'budget'] = 4

df_train.loc[df_train['id'] == 2118,'budget'] = 344 

df_train.loc[df_train['id'] == 2252,'budget'] = 130

df_train.loc[df_train['id'] == 2256,'budget'] = 1 

df_train.loc[df_train['id'] == 2696,'budget'] = 10000000
"""

This is slightly cheating - features are known after the movie is released

"""



train_add_features = pd.read_csv('../input/tmdb-additional-features/TrainAdditionalFeatures.csv')

test_add_features = pd.read_csv('../input/tmdb-additional-features/TestAdditionalFeatures.csv')



df_train = df_train.merge(train_add_features, how='left',left_on='imdb_id',right_on='imdb_id')

df_test = df_test.merge(test_add_features, how='left',left_on='imdb_id',right_on='imdb_id')



train_data_cols = df_train.columns

test_data_cols = df_test.columns





for col_test in test_data_cols:

    if str(col_test) not in train_data_cols and str(col_test) != 'year_bucket' and not str(col_test).__contains__('0') and not str(col_test).__contains__('1') and not str(col_test).__contains__('2'):

        print ('dropping ' + str(col_test))

        df_test.drop(str(col_test),axis=1,inplace=True)





for col_test in train_data_cols:

    if str(col_test) not in test_data_cols and str(col_test) != 'year_bucket' and not str(col_test).__contains__('0') and not str(col_test).__contains__('1') and not str(col_test).__contains__('2'):

        if (str(col_test) == 'revenue'):

            continue

        print ('dropping ' + str(col_test))

        df_train.drop(str(col_test),axis=1,inplace=True)
print ('The final shapes of train and test data')

print (df_train.shape)

print (df_test.shape)
def is_collection(row):

    if (row == 0):

        return 0

    else:

        return 1

    

def is_homepage(row):

    if (row == 0):

        return 0

    else:

        return 1



df_train['is_collection'] = df_train.belongs_to_collection.apply(is_collection)

df_train['is_homepage'] = df_train.homepage.apply(is_homepage)

df_test['is_collection'] = df_test.belongs_to_collection.apply(is_collection)

df_test['is_homepage'] = df_test.homepage.apply(is_homepage)



df_train['has_production_comp'] = df_train.production_companies.apply(is_collection)

df_test['has_production_comp'] = df_test.production_companies.apply(is_collection)



df_train['has_production_count'] = df_train.production_countries.apply(is_collection)

df_test['has_production_count'] = df_test.production_countries.apply(is_collection)



df_train['has_keywords'] = df_train.Keywords.apply(is_collection)

df_test['has_keywords'] = df_test.Keywords.apply(is_collection)



df_train['has_tagline'] = df_train.tagline.apply(is_collection)

df_test['has_tagline'] = df_test.tagline.apply(is_collection)
df_train.drop('belongs_to_collection',axis=1,inplace=True)

df_train.drop('collection',axis=1,inplace=True)

df_train.drop('genres',axis=1,inplace=True)

df_train.drop('homepage',axis=1,inplace=True)

df_train.drop('imdb_id',axis=1,inplace=True)

df_train.drop('original_language',axis=1,inplace=True)

df_train.drop('original_title',axis=1,inplace=True)

df_train.drop('overview',axis=1,inplace=True)



df_train.drop('poster_path',axis=1,inplace=True)

df_train.drop('production_companies',axis=1,inplace=True)

df_train.drop('production_countries',axis=1,inplace=True)

df_train.drop('release_date',axis=1,inplace=True)

#df_train.drop('release_date_2',axis=1,inplace=True)

df_train.drop('spoken_languages',axis=1,inplace=True)

df_train.drop('status',axis=1,inplace=True)

df_train.drop('tagline',axis=1,inplace=True)

df_train.drop('title',axis=1,inplace=True)

df_train.drop('Keywords',axis=1,inplace=True)

df_train.drop('cast',axis=1,inplace=True)

df_train.drop('crew',axis=1,inplace=True)



df_test.drop('belongs_to_collection',axis=1,inplace=True)

df_test.drop('collection',axis=1,inplace=True)

df_test.drop('genres',axis=1,inplace=True)

df_test.drop('homepage',axis=1,inplace=True)

df_test.drop('imdb_id',axis=1,inplace=True)

df_test.drop('original_language',axis=1,inplace=True)

df_test.drop('original_title',axis=1,inplace=True)

df_test.drop('overview',axis=1,inplace=True)



df_test.drop('poster_path',axis=1,inplace=True)

df_test.drop('production_companies',axis=1,inplace=True)

df_test.drop('production_countries',axis=1,inplace=True)

df_test.drop('release_date',axis=1,inplace=True)

#df_test.drop('release_date_2',axis=1,inplace=True)

df_test.drop('spoken_languages',axis=1,inplace=True)

df_test.drop('status',axis=1,inplace=True)

df_test.drop('tagline',axis=1,inplace=True)

df_test.drop('title',axis=1,inplace=True)

df_test.drop('Keywords',axis=1,inplace=True)

df_test.drop('cast',axis=1,inplace=True)

df_test.drop('crew',axis=1,inplace=True)
df_train['revenue'].fillna(0,inplace=True)

df_train['log_revenue'] = np.log(df_train['revenue'])

df_train['budget'].fillna(0,inplace=True)

df_train['log_budget'] = np.log(df_train['budget'])



df_train.drop('revenue',axis=1,inplace=True)

df_train.drop('budget',axis=1,inplace=True)



df_test['budget'].fillna(0,inplace=True)

df_test['log_budget'] = np.log(df_test['budget'])

df_test.drop('budget',axis=1,inplace=True)



df_train['budget_year'] = df_train['log_budget'] / df_train['release_date_year']

df_test['budget_year'] = df_test['log_budget'] / df_test['release_date_year']



df_train['popularity_year'] = df_train['popularity'] / df_train['release_date_year']

df_test['popularity_year'] = df_test['popularity'] / df_test['release_date_year']



df_train.runtime.fillna(0,inplace=True)

df_test.runtime.fillna(0,inplace=True)

bins = [0, 40, 50, 60, 70, 80,90, 100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350]

labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

df_train['binned_runtime'] = pd.cut(df_train['runtime'], bins=bins, labels=labels)

df_test['binned_runtime'] = pd.cut(df_test['runtime'], bins=bins, labels=labels)



df_train['binned_runtime'] = pd.to_numeric(df_train['binned_runtime'])

df_test['binned_runtime'] = pd.to_numeric(df_test['binned_runtime'])
import xgboost as xgb

from xgboost import XGBRegressor



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score



num_rounds = 10000



y_label = df_train['log_revenue']

X = df_train.drop('log_revenue',axis=1)

X.drop('id',axis=1,inplace=True)

X['cast_1_gender'] = X['cast_1_gender'].apply(int)

X['cast_2_gender'] = X['cast_2_gender'].apply(int)

X['cast_3_gender'] = X['cast_3_gender'].apply(int)

X = X.replace([np.inf, -np.inf], 0).fillna(0)

X.fillna(0,inplace=True)







X_test = df_test.copy()

X_test.fillna(0,inplace=True)

X_test_id = df_test['id']

X_test.drop('id',axis=1,inplace=True)

X_test['cast_1_gender'] = X_test['cast_1_gender'].apply(int)

X_test['cast_2_gender'] = X_test['cast_2_gender'].apply(int)

X_test['cast_3_gender'] = X_test['cast_3_gender'].apply(int)

X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

X_test.fillna(0,inplace=True)



X = X.loc[:,~X.columns.duplicated()]

X_test = X_test.loc[:,~X_test.columns.duplicated()]



dtrain = xgb.DMatrix(data=X,label=y_label)

dtest = xgb.DMatrix(data=X_test)

watchlist = [(dtest, 'eval'), (dtrain, 'train')]



param = {'eta':0.01, 'max_depth': 2, 'booster': 'gbtree', 'colsample_bytree' : 0.3,'subsample' : 0.9}



res = xgb.cv(param,dtrain,num_rounds,10,metrics={'rmse'})

print ('cross validation')

print (res)





bst = xgb.train(param,dtrain,num_rounds)

y_pred = bst.predict(dtrain)

print ('Train rmsle')

print (mean_squared_error(y_label, y_pred) ** 0.5)





y_pred_reg = bst.predict(dtest)

y_pred_reg_exp = np.exp(y_pred_reg)

y_pred_reg_exp = pd.DataFrame(y_pred_reg_exp)

test_preds = pd.concat([X_test_id,y_pred_reg_exp],axis=1)

test_preds.columns = ['id','revenue']

#test_preds.head(10)

test_preds.to_csv('submission_xgb.csv',index=False)