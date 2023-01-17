# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.listdir('/kaggle/input')
from collections import Counter 

def extract_response_title(response):

    all_wrds=[]

    try:

        for annotation in response["annotations"]:

            if annotation["cosine"]>0.1:

                if len(annotation["dbPediaTypes"])>0:

                #print("%s (%s) %s" % (annotation["title"], annotation["url"], annotation["dbPediaTypes"]))

                    all_wrds.append(annotation["title"])

    except:

        1==1

    return [key for key, value in Counter(all_wrds).most_common()] 



def extract_response_val(response):

    all_wrds=[]

    try:

        for annotation in response["annotations"]:

            if annotation["cosine"]>0.1:

                #if len(annotation["dbPediaTypes"])>0:

                #print("%s (%s) %s" % (annotation["title"], annotation["url"], annotation["dbPediaTypes"]))

                all_wrds.append(annotation["dbPediaTypes"])

    except:

        1==1

    all_wrds=[k for j in all_wrds for k in j ]

    return [key for key, value in Counter(all_wrds).most_common()]  

read_data_All=pd.read_pickle('/kaggle/input/covid-all-wiki-columns-data/ALL_study_with_wiki_columns.pkl')



read_data_All['title_kw']=read_data_All['all_txt_resp'].apply(lambda x: extract_response_title(x))

read_data_All['Cat_mach']=read_data_All['all_txt_resp'].apply(lambda x: extract_response_val(x))
read_data_All=read_data_All[['title','title_kw','Cat_mach']]

read_data_All
ref_table=pd.read_pickle('/kaggle/input/covid-adjency/res_ref.pkl')

ref_table
main_Data_csv=pd.read_csv('/kaggle/input/covid-pickle-csv/title.csv')
main_Data_csv_new=main_Data_csv.merge(ref_table,left_on='sha',right_on='paper_id',how='inner')
main_Data_csv_new=main_Data_csv_new.merge(main_Data_csv_new[['pid','title_x']].drop_duplicates(),left_on=['ref_title_y'],right_on=['title_x'],how='inner')
main_Data_csv_new.shape
main_Data_csv_new[main_Data_csv_new['pid_y'].isna()]
main_Data_csv_new=main_Data_csv_new[['pid_x','pid_y']]
ref_table=pd.read_pickle('/kaggle/input/covid-adjency/result.pkl')
ref_table.columns=['pid_x','pid_y']
ref_table=ref_table.append(main_Data_csv_new)
ref_table=ref_table.merge(main_Data_csv,left_on='pid_x',right_on='pid',how='left')
ref_table.to_pickle('Graph_adj_matrix.pkl')
read_data_All=read_data_All.merge(main_Data_csv[['pid','title']],on='title',how='right')
read_data_All.to_pickle('Graph_kw_matrix.pkl')
ref_table.columns
read_data_All.columns