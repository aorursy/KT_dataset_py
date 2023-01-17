import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.units as units

import matplotlib.ticker as ticker

import random

import time                   # To time processes

import warnings               # To suppress warnings

import itertools

import datetime

import tensorflow as tf

import csv

import math

import calendar



from random import shuffle

from pandas import read_csv

from sklearn import metrics

from sklearn import svm

from matplotlib import pyplot

from numpy import array

from numpy import argmax

from scipy import stats

from datetime import datetime

from IPython.display import Image

from prettytable import PrettyTable



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report, confusion_matrix  

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.cluster import KMeans

from sklearn.metrics import precision_recall_curve



from sklearn.utils import class_weight

from sklearn.utils.fixes import signature

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.base import TransformerMixin

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.callbacks import Callback



from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.over_sampling import RandomOverSampler



from matplotlib.pyplot import *

import matplotlib.pyplot as plotter





warnings.filterwarnings("ignore")
df=read_csv("../input/matches.csv")
df['result'].value_counts()
temp=df.loc[df['win_by_runs'] != 0]

print("\nFor a maximum of "+ str(temp['winner'].value_counts()[0]) + " times, " + temp['winner'].value_counts().index[0] + " has defended their score")

temp1=df.loc[df['win_by_wickets'] != 0]

print("\nFor a maximum of "+ str(temp1['winner'].value_counts()[0]) + " times, " + temp1['winner'].value_counts().index[0] + " has chased the totals successfully")

temp=df[df['win_by_runs']==df['win_by_runs'].max()].reset_index(drop=True)



print("\n"+temp['team1'][0]+ " won by " + str(temp['win_by_runs'][0]) + " against " + temp['team2'][0]+ " in IPL "+ str(temp["season"][0]))
temp=df[df['win_by_wickets']==df['win_by_wickets'].max()]



temp=temp.sort_values(['season']).reset_index(drop=True)



for i in range(0,temp.shape[0]):

    print(temp['team1'][i]+ " won by " + str(temp['win_by_wickets'][i]) + " against " + temp['team2'][i]+ " in IPL "+ str(temp["season"][i]))

    print("\n")
df1=df.dropna(subset=['winner'])
for i in range(0,df1.shape[0]):

    d=df1.iloc[i]['date']

    if(d[2]=='/' and d[5]=='/'):

        #print(type(datetime.strptime(d, "%d/%m/%y").strftime("%Y-%m-%d")))

        df1.iloc[i, df1.columns.get_loc('date')] = datetime.strptime(d, "%d/%m/%y").strftime("%Y-%m-%d")

        #df1.iloc[i]['date']=datetime.strptime(d, "%d/%m/%y").strftime("%Y-%m-%d")

        #print(df1.iloc[i]['date'])
df2=df1.sort_values(['season','date']).reset_index(drop=True)
kkr=[]

rcb=[]

mi=[]

csk=[]

kxip=[]

dd=[]

dc=[]

srh=[]

rr=[]

rpsg=[]

pwi=[]

gl=[]

ktk=[]



tot_win_count=[]

tot_name=[]

tot_loss_count=[]

c=0

nightmares=[]
for i in range(0,df2.shape[0]):

    if(df2['team1'][i]=="Kolkata Knight Riders" or df2['team2'][i]=="Kolkata Knight Riders"):

        kkr.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Royal Challengers Bangalore" or df2['team2'][i]=="Royal Challengers Bangalore"):

        rcb.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Mumbai Indians" or df2['team2'][i]=="Mumbai Indians"):

        mi.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Chennai Super Kings" or df2['team2'][i]=="Chennai Super Kings"):

        csk.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Kings XI Punjab" or df2['team2'][i]=="Kings XI Punjab"):

        kxip.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Delhi Daredevils" or df2['team2'][i]=="Delhi Daredevils"):

        dd.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Deccan Chargers" or df2['team2'][i]=="Deccan Chargers"):

        dc.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Sunrisers Hyderabad" or df2['team2'][i]=="Sunrisers Hyderabad"):

        srh.append(df2['winner'][i])

    

    if(df2['team1'][i]=="Rajasthan Royals" or df2['team2'][i]=="Rajasthan Royals"):

        rr.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Rising Pune Supergiants" or df2['team2'][i]=="Rising Pune Supergiants"):

        rpsg.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Pune Warriors" or df2['team2'][i]=="Pune Warriors"):

        pwi.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Gujarat Lions" or df2['team2'][i]=="Gujarat Lions"):

        gl.append(df2['winner'][i])

        

    if(df2['team1'][i]=="Kochi Tuskers Kerala" or df2['team2'][i]=="Kochi Tuskers Kerala"):

        ktk.append(df2['winner'][i])

    

        
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(kkr)-1):

    if(kkr[i]=="Kolkata Knight Riders" and kkr[i+1]=="Kolkata Knight Riders"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(kkr[i]!="Kolkata Knight Riders" and kkr[i+1]!="Kolkata Knight Riders"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Kolkata Knight Riders")



s=""

t1="Kolkata Knight Riders"

x=pd.Series(kkr)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)

pos=[]



for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(rcb)-1):

    if(rcb[i]=="Royal Challengers Bangalore" and rcb[i+1]=="Royal Challengers Bangalore"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(rcb[i]!="Royal Challengers Bangalore" and rcb[i+1]!="Royal Challengers Bangalore"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Royal Challengers Bangalore")



s=""

t1="Royal Challengers Bangalore"

x=pd.Series(rcb)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(mi)-1):

    if(mi[i]=="Mumbai Indians" and mi[i+1]=="Mumbai Indians"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(mi[i]!="Mumbai Indians" and mi[i+1]!="Mumbai Indians"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Mumbai Indians")



s=""

t1="Mumbai Indians"

x=pd.Series(mi)

t2=x.value_counts().tolist()



t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)

s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(csk)-1):

    if(csk[i]=="Chennai Super Kings" and csk[i+1]=="Chennai Super Kings"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(csk[i]!="Chennai Super Kings" and csk[i+1]!="Chennai Super Kings"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Chennai Super Kings")



s=""

t1="Chennai Super Kings"

x=pd.Series(csk)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)

        

s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(kxip)-1):

    if(kxip[i]=="Kings XI Punjab" and kxip[i+1]=="Kings XI Punjab"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(kxip[i]!="Kings XI Punjab" and kxip[i+1]!="Kings XI Punjab"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Kings XI Punjab")



s=""

t1="Kings XI Punjab"

x=pd.Series(kxip)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)

        

s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(dd)-1):

    if(dd[i]=="Delhi Daredevils" and dd[i+1]=="Delhi Daredevils"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(dd[i]!="Delhi Daredevils" and dd[i+1]!="Delhi Daredevils"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Delhi Daredevils")



s=""

t1="Delhi Daredevils"

x=pd.Series(dd)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(dc)-1):

    if(dc[i]=="Deccan Chargers" and dc[i+1]=="Deccan Chargers"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(dc[i]!="Deccan Chargers" and dc[i+1]!="Deccan Chargers"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Deccan Chargers")



s=""

t1="Deccan Chargers"

x=pd.Series(dc)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(srh)-1):

    if(srh[i]=="Sunrisers Hyderabad" and srh[i+1]=="Sunrisers Hyderabad"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(srh[i]!="Sunrisers Hyderabad" and srh[i+1]!="Sunrisers Hyderabad"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Sunrisers Hyderabad")



s=""

t1="Sunrisers Hyderabad"

x=pd.Series(srh)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(rr)-1):

    if(rr[i]=="Rajasthan Royals" and rr[i+1]=="Rajasthan Royals"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(rr[i]!="Rajasthan Royals" and rr[i+1]!="Rajasthan Royals"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Rajasthan Royals")



s=""

t1="Rajasthan Royals"

x=pd.Series(rr)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(rpsg)-1):

    if(rpsg[i]=="Rising Pune Supergiants" and rpsg[i+1]=="Rising Pune Supergiants"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(rpsg[i]!="Rising Pune Supergiants" and rpsg[i+1]!="Rising Pune Supergiants"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Rising Pune Supergiants")



s=""

t1="Rising Pune Supergiants"

x=pd.Series(rpsg)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(pwi)-1):

    if(pwi[i]=="Pune Warriors" and pwi[i+1]=="Pune Warriors"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(pwi[i]!="Pune Warriors" and pwi[i+1]!="Pune Warriors"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Pune Warriors")



s=""

t1="Pune Warriors"

x=pd.Series(pwi)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(gl)-1):

    if(gl[i]=="Gujarat Lions" and gl[i+1]=="Gujarat Lions"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(gl[i]!="Gujarat Lions" and gl[i+1]!="Gujarat Lions"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Gujarat Lions")



s=""

t1="Gujarat Lions"

x=pd.Series(gl)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
win_count=1

loss_count=1

maxi_win=-1

maxi_loss=-1



for i in range(0,len(ktk)-1):

    if(ktk[i]=="Kochi Tuskers Kerala" and ktk[i+1]=="Kochi Tuskers Kerala"):

        win_count+=1

    else:

        maxi_win=max(win_count,maxi_win)

        win_count=1

    if(ktk[i]!="Kochi Tuskers Kerala" and ktk[i+1]!="Kochi Tuskers Kerala"):

        loss_count+=1

    else:

        maxi_loss=max(loss_count, maxi_loss)

        loss_count=1

        

tot_win_count.append(maxi_win)

tot_loss_count.append(maxi_loss)

tot_name.append("Kochi Tuskers Kerala")



s=""

t1="Kochi Tuskers Kerala"

x=pd.Series(ktk)

t2=x.value_counts().tolist()

t2.pop(0)

m=max(t2)



pos=[]

for i in range(0,len(t2)):

    if(t2[i]==m):

        pos.append(i)



s+=t1+" has lost against "

for i in range(0,len(pos)):

    s+=x.value_counts().index[pos[i]+1]

    if(i!=len(pos)-1):

        s+=", "



s+=" for a maximum of "+str(m)+" times.\n\n"

#print(s)

nightmares.append(s)
m=max(tot_win_count)

pos=[]

for i in range(0,len(tot_win_count)):

    if(tot_win_count[i]==m):

        pos.append(i)



for i in range(0,len(pos)):

    print("\n"+tot_name[pos[i]]+" - "+str(m)+"\n")
m=max(tot_loss_count)

pos=[]

for i in range(0,len(tot_loss_count)):

    if(tot_loss_count[i]==m):

        pos.append(i)



for i in range(0,len(pos)):

    print("\n"+tot_name[pos[i]]+" - "+str(m)+"\n")
for i in range(0,13):

    print(nightmares[i])