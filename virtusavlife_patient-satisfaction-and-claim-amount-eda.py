import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = None
pd.set_option('display.max_colwidth', -1)
%matplotlib inline 
def capitalize_after_hyphen(x):
    a=list(x)
    a[p.index('-')+1]=a[p.index('-')+1].capitalize()
    x=''.join(a)
    return ''.join(a)

import pandas as pd
import requests  
#l=['patients','admdissions','diagnoses','drg-codes','icu-stays','procedures','prescriptions','d-icd-diagnoses','d-icd-procedures']
url1="http://ec2-54-88-151-77.compute-1.amazonaws.com:3004/v1/hrrd-table?limit=10000&offset=0"
url2="http://ec2-54-88-151-77.compute-1.amazonaws.com:3004/v1/ipps-table?limit=10000&offset=0"
url3="http://ec2-54-88-151-77.compute-1.amazonaws.com:3004/v1/hcahps-table?limit=10000&offset=0"
url4="http://ec2-54-88-151-77.compute-1.amazonaws.com:3002/v1/state-codes?limit=10000&offset=0"
url5="http://ec2-54-88-151-77.compute-1.amazonaws.com:3002/v1/beneficiary-summaries?limit=10000&offset=0"

d={}
url=[url1,url2,url3,url4,url5]

for x in url:  
    p = x[(x.index('v1/')+len('v1/')):x.index('?limit')]
    if p=='state-codes':
        p='stateCode'
    else:
        
        try:
            p=capitalize_after_hyphen(p)
        except:
            pass
        try:
            p=p[:p.index('-')]+p[p.index('-')+1:]
        except:
            pass

        try:
            p=capitalize_after_hyphen(p)
        except:
            pass
        try:
            p=p[:p.index('-')]+p[p.index('-')+1:]
        except:
            pass
    
    
    
    print(p)
    
    d['{}'.format(p)]=pd.DataFrame(requests.get(x).json()['{}'.format(p)])

d['bene_summart'] = d['beneficiarySummaries']
d['state_code'] = d['stateCode']
d['hcahps'] = d['hcahpsTable']
d['ipps'] = d['ippsTable']
d['hrrd'] = d['hrrdTable']
df=pd.merge(d['bene_summart'],d['state_code'],left_on='SP_STATE_CODE',right_on='id',how='left')
df.drop(['index'],axis=1,inplace=True)
df
d['hcahps']
d['hrrd']
d['ipps']
df.drop(['SP_STATE_CODE','id'],axis=1,inplace=True)
df=df[['state_code','MEDREIMB_IP','BENRES_IP']]
df['total']=df[['MEDREIMB_IP','BENRES_IP']].sum(axis=1)
df.columns=['state','insurance','personal','total']
df['percent_personal']=df['personal']/df['total']
df=df.fillna(0)
df[['insurance','personal','total','percent_personal']]=df[['insurance','personal','total','percent_personal']].astype(float).round(2)
a=df.groupby('state')
plt.figure(figsize=(15,10))
a.get_group('CA')['percent_personal'].hist()
plt.figure(figsize=(15,10))
df.groupby('state')['percent_personal'].mean().plot.bar()
b=d['hcahps'].groupby('state')
d['hcahps']=d['hcahps'][['state','hcahps_answer_description','hcahps_answer_percent']]
d['hcahps'].columns=['state','answer','percent']
states=list(d['hcahps']['state'].unique())
l=list(d['hcahps']['answer'].unique())
df2=pd.DataFrame(index=states,columns=l)
for x in list(df2.index): #state
    for j in df2.columns: #question
        for a in range(0,d['hcahps'].shape[0]):
            if (x==d['hcahps'].loc[a,'state'])&(j==d['hcahps'].loc[a,'answer']):
                df2.loc[x,j]=d['hcahps'].loc[a,'percent']
                
                
df1=pd.merge(df.groupby('state')[['insurance','personal','total']].mean().reset_index(),df2.reset_index(),left_on='state',right_on='index',how='right')
df1.drop('index',axis=1,inplace=True)
df1=df1.set_index('state')
df1[['insurance','personal','total']]=df1[['insurance','personal','total']].round(2)
for x in df1.columns:
    try:
        df1[x]=df1[x].str.strip('%')
    except:
        pass
df1=df1.astype(float)
plt.rcParams["figure.figsize"] = [15,10]
for x in df1.columns:
    sns.lmplot(x=x,y='total',data=df1,fit_reg=True)
    plt.xlim(0,100)
    plt.show()
df1=df1.dropna()
for x in df1.columns:
    
    df1[x].plot.bar()
    plt.ylabel(x)
    plt.show()
    
df1
dissatisfied=['never','did not','Disagree','low','not recommend']

l=[]
for x in df1.columns:
    for j in dissatisfied:
        if j in x:
            l.append(x)

df1['patient_dissatisfaction']=df1[l].sum(axis=1)
happy=['always','did','Strongly Agree','high','definitely']

p=[]
for x in df1.columns:
    for j in happy:
        if j in x:
            p.append(x)

df1['patient_happiness']=df1[p].sum(axis=1)
neutral=['usually','did','Agree','medium','probably']

q=[]
for x in df1.columns:
    for j in neutral:
        if j in x:
            q.append(x)

df1['patient_neutral']=df1[q].sum(axis=1)
list(df1)
df_=df1[['insurance',
 'personal',
 'total','patient_dissatisfaction',
 'patient_happiness',
 'patient_neutral']]
df_['percent']=df_['personal']/df_['total']
df_=df_.sort_values(by='percent')
df_
sns.heatmap(df_.corr())
df2
