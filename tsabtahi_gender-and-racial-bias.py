# numpy, pandas

import numpy as np 

import pandas as pd 

import datetime

import numpy as np



# plots

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib

matplotlib.style.use('ggplot')



from subprocess import check_output



# Supress unnecessary warnings so that presentation looks clean

import warnings

warnings.filterwarnings('ignore')



#Print all rows and columns. Dont hide any

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)
df_pr= pd.read_csv('../input/person.csv')

df_pr.head(2)



df_pr = df_pr[df_pr.VEH_NO==1]

df_pr = df_pr[df_pr.PER_NO==1]

df_pr.head(2)
df_prr=df_pr[['ST_CASE','STATE','DAY','MONTH','HOUR','AGE','SEX','DRINKING','RACE']]

states = {1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 

          6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE', 

          11: 'DC', 12: 'FL', 13: 'GA', 15: 'HI', 

          16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS', 

          21: 'KY', 22: 'LA', 23: 'ME', 24: 'MD', 

          25: 'MA', 26: 'MI', 27: 'MN', 

          28:'MS', 29: 'MO', 30: 'MT', 31: 'NE', 

          32: 'NV', 33: 'NH', 34: 'NJ', 35: 'NM', 

          36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 

          40: 'OK', 41: 'OR', 42: 'PN', 43: 'PR', 

          44: 'RI', 45: 'SC', 46: 'SD', 47: 'TN', 

          48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 52: 'VI', 

          53: 'WA', 54: 'WV', 55: 'WI', 56: 'WY'}





df_prr['STATE']=df_prr['STATE'].apply(lambda x: states[x])





month = {1: '1jan', 2: '2feb', 3: '3mar', 4: '4april', 

          5: '5may', 6: '6june', 7: '7july', 8: '8aug', 

          9: '90sep', 10: '91oct',11: '92nov',12: '93dec'}



df_prr['MONTH']=df_prr['MONTH'].apply(lambda x: month[x])



df_prr=df_prr[df_prr.HOUR != 99]



df_prr.head(3)
df_prr=df_prr[df_prr.SEX <3]

df_prr=df_prr[df_prr.DRINKING <8]

df_prr=df_prr[df_prr.AGE <130]

df_prr=df_prr[df_prr.RACE <68]

df_prr.head(2)
sex = {1: 'M', 2: 'F'}

drinking={0:'noDrnk',1:'drnk'}

race={0:'notAFatality',1:'white',2:'black',3:'americanIndian',4:'chinese',5:'japanese',6:'hawaian',7:'filipino',

      18:'asianIndian',19:'otherIndian',28:'korean',38:'samoan',48:'vietnamese',58:'guamanian'}





df_prr['SEX']=df_prr['SEX'].apply(lambda x: sex[x])

df_prr['DRINKING']=df_prr['DRINKING'].apply(lambda x: drinking[x])

df_prr['RACE']=df_prr['RACE'].apply(lambda x: race[x])



df_prr.head(2)
df_prr['AGE']=df_prr['AGE'].apply(lambda x: int(x/3)*3)



df_prr.head(2)
from pylab import rcParams

rcParams['figure.figsize'] = 8, 6

a=df_prr['AGE'].value_counts().sort_index()

a.plot(fontsize=10,title='Fatality over age (All Community)')

plt.legend(fontsize=8)
M=df_prr[df_prr.SEX == 'M']

F=df_prr[df_prr.SEX == 'F']



M_w=M[M.RACE == 'white']

F_w=F[F.RACE == 'white']



M_b=M[M.RACE == 'black']

F_b=F[F.RACE == 'black']
rcParams['figure.figsize'] = 8,6

a=M['AGE'].value_counts().sort_index()

a=a/a.max()

a.plot()



rcParams['figure.figsize'] = 8, 6

b=F['AGE'].value_counts().sort_index()

b=b/b.max()





b.plot(fontsize=10,title='Normalized gender bias \n on Fatality over age (All Community)')

plt.legend(('Male','Female'),fontsize=8)
rcParams['figure.figsize'] = 8,6

a=M_w['AGE'].value_counts().sort_index()

a=a/a.max()

a.plot()



rcParams['figure.figsize'] = 8, 6

b=F_w['AGE'].value_counts().sort_index()

b=b/b.max()

b.plot(fontsize=10,title='Normalized gender bias on Fatality over age (White Community)')



plt.legend(('Male White','Female White'),fontsize=8)
rcParams['figure.figsize'] = 8,6

a=M_b['AGE'].value_counts().sort_index()

a=a/a.max()

a.plot()



rcParams['figure.figsize'] = 8,6

b=F_b['AGE'].value_counts().sort_index()

b=b/b.max()

b.plot(fontsize=10,title='Normalized gender bias on \n Fatality over age (African american Community)')



plt.legend(('Male African American','Female African American'),fontsize=8)
M_w_dr=M_w[M_w.DRINKING == 'drnk']

F_w_dr=F_w[F_w.DRINKING == 'drnk']



M_b_dr=M_b[M_b.DRINKING == 'drnk']

F_b_dr=F_b[F_b.DRINKING == 'drnk']
rcParams['figure.figsize'] = 8,6

a=M_w_dr['AGE'].value_counts().sort_index()

a=a/a.max()

a.plot()



rcParams['figure.figsize'] =  8,6

b=F_w_dr['AGE'].value_counts().sort_index()

b=b/b.max()

b.plot(fontsize=10,title='Normalized gender bias on \n Fatality over age (White Community - Drinking)')



plt.legend(('Male White-Drinking','Female White-Drinking'),fontsize=8)