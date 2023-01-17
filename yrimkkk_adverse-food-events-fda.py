import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/CAERS_ASCII_2004_2017Q2.csv')
df['RA_CAERS Created Date']=pd.to_datetime(df['RA_CAERS Created Date'], format='%m/%d/%Y')
df['AEC_Event Start Date']=pd.to_datetime(df['AEC_Event Start Date'], format='%m/%d/%Y')
df.head()
#성별 비율 확인
fig, ax = plt.subplots(figsize=(10,6))
sns.countplot(df['CI_Gender'],edgecolor=sns.color_palette('dark',7))
#산업별로 사건 발생 카운트
fig, ax = plt.subplots(figsize=(8,12))
sns.set_style("ticks")
sns.countplot(y=df['PRI_FDA Industry Name']).set_title('Health event counts by Industry Name')
#브랜드/제품 명 column에서 빈도수가 적은 항목 삭제 
fig, ax = plt.subplots(figsize=(8,12))
product_count=df.groupby('PRI_Reported Brand/Product Name').size()
product_count_large=product_count[(product_count>200) & (product_count.index!='REDACTED')]

#빈도수 많은 제품 나타내기
product_count_df=pd.DataFrame({'product':product_count_large.index,'count':product_count_large}, index=None)
df_new=product_count_df.merge(df[['PRI_Reported Brand/Product Name','PRI_FDA Industry Name']],how='inner', left_on='product', right_on='PRI_Reported Brand/Product Name').drop_duplicates()[['count','product','PRI_FDA Industry Name']]
sns.barplot(x='count',y='product',hue='PRI_FDA Industry Name',data=df_new,dodge=False).set_title("Products with more than 200 health events")
fig, ax = plt.subplots(figsize=(14,10))
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Soft Drink/Water')]['CI_Age at Adverse Event'], label='Soft Drink/Water')
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Fishery/Seafood Prod')]['CI_Age at Adverse Event'], label='Fishery/Seafood Prod')
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Vit/Min/Prot/Unconv Diet(Human/Animal)')]['CI_Age at Adverse Event'], label='Vit/Min/Prot/Unconv Diet(Human/Animal)')
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Nuts/Edible Seed')]['CI_Age at Adverse Event'], label='Nuts/Edible Seed')
ax = sns.kdeplot(df[(df['CI_Age Unit'] == 'Year(s)') & (df['PRI_FDA Industry Name']=='Cosmetics')]['CI_Age at Adverse Event'], label='Cosmetics').set(xlim=(0, 100))
deadly=df[(df['SYM_One Row Coded Symptoms']!=np.NaN) & (df['SYM_One Row Coded Symptoms'].str.contains('DEATH'))]
fig, ax = plt.subplots(figsize=(10,20))
product_count=deadly.groupby('PRI_Reported Brand/Product Name').size()
product_count_large=product_count[(product_count>1) & (product_count.index!='REDACTED')]
product_count_df=pd.DataFrame({'product':product_count_large.index,'count':product_count_large}, index=None)
new=product_count_df.merge(deadly[['PRI_Reported Brand/Product Name','PRI_FDA Industry Name']],how='inner', left_on='product', right_on='PRI_Reported Brand/Product Name').drop_duplicates()[['count','product','PRI_FDA Industry Name']]
sns.barplot(x='count',y='product',hue='PRI_FDA Industry Name',data=new).set_title("Products which were consumed by more than one patient who died")







































