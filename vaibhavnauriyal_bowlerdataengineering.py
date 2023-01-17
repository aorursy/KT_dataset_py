
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset=pd.read_csv('/kaggle/input/cricket-player/Men ODI Player Innings Stats - 21st Century.csv')
dataset=dataset[dataset.Opposition!='v U.A.E.']
dataset=dataset[dataset.Opposition!='v P.N.G.']
dataset=dataset[dataset.Opposition!='v U.S.A.']
dataset=dataset[dataset.Opposition!='v ICC World XI']
bowler=dataset[dataset['Innings Runs Scored'].isnull()]
drop=[ 'Innings Runs Scored', 'Innings Runs Scored Num',
       'Innings Minutes Batted', 'Innings Batted Flag', 'Innings Not Out Flag',
       'Innings Balls Faced', 'Innings Boundary Fours',
       'Innings Boundary Sixes', 'Innings Batting Strike Rate',
      "50's", "100's", 'Innings Runs Scored Buckets','Innings Economy Rate'
     ,'Innings Wickets Taken Buckets','Innings Bowled Flag','10 Wickets']
bowler.drop(drop,axis=1,inplace=True)
bowler.columns
bowler['Innings Wickets Taken'].value_counts()
#creating a function to replace all unwanted values with 0

def fill(df,col):
    df[col]=np.where((df[col]=='-') | (df[col]=='DNB') |(df[col]=='TDNB'),'0',df[col])
    
fill(bowler,'Innings Wickets Taken')
fill(bowler,'Innings Maidens Bowled')
fill(bowler,'Innings Runs Conceded')
fill(bowler,'Innings Number')
fill(bowler,'Innings Overs Bowled')
#creating a function to convert object type features to numeric features

def convert(df,col2,col1):
    index_col1=df.columns.get_loc(col1)
    index_col2=df.columns.get_loc(col2)
    for row in range(0,len(df)):
        col= df.iat[row,index_col1]
        df.iat[row,index_col2]=int(col)
    
    
bowler['4_Wickets']=0
bowler['5_Wickets']=0
bowler['Maidens_Bowled']=0
bowler['Wickets_Taken']=0
bowler['Runs_conceded']=0
bowler['Innings_Number']=0
bowler['Overs_Bowled']=0.0

convert(bowler,'4_Wickets','4 Wickets')
convert(bowler,'5_Wickets','5 Wickets')
convert(bowler,'Maidens_Bowled','Innings Maidens Bowled')
convert(bowler,'Wickets_Taken','Innings Wickets Taken')
convert(bowler,'Runs_conceded','Innings Runs Conceded')
convert(bowler,'Innings_Number','Innings Number')

def convert2(df,col2,col1):
    index_col1=df.columns.get_loc(col1)
    index_col2=df.columns.get_loc(col2)
    for row in range(0,len(df)):
        col= df.iat[row,index_col1]
        if col!='sub':
            df.iat[row,index_col2]=float(col)

convert2(bowler,'Overs_Bowled','Innings Overs Bowled')
bowler.columns
#removing redundant features

drop=['Innings Overs Bowled',
       'Innings Maidens Bowled', 'Innings Runs Conceded',
       'Innings Wickets Taken', '4 Wickets', '5 Wickets','Innings Number']
bowler.drop(drop,axis=1,inplace=True)
import re
#creating regular expressions to extract Opposition country team's name
index_3=bowler.columns.get_loc('Opposition')
opp = r'[^v][A-Z]+[a-z]*[" "]*[A-Z]*[a-z]*'
for row in range(0,len(bowler)):
    opps=re.search(opp,bowler.iat[row,index_3]).group()
    bowler.iat[row,index_3]=opps
#extracting year from Date feature
bowler['Year']=0
years=r'([0-9]{4})'
index_3=bowler.columns.get_loc('Innings Date')
index_year=bowler.columns.get_loc('Year')


for row in range(0,len(bowler)):
    year=re.search(years,bowler.iat[row,index_3]).group()
    bowler.iat[row,index_year]=int(year)
bowler['Month']=0
bowler['Innings Date']=pd.to_datetime(bowler['Innings Date'])
index_month=bowler.columns.get_loc('Month')
for row in range(0,len(bowler)):
    bowler.iat[row,index_month]=int(bowler.iat[row, index_3].month)
#extracting day from Date feature
bowler['Day']=''
index_day=bowler.columns.get_loc('Day')
import calendar
for row in range(0,len(bowler)):
    bowler.iat[row,index_day]=calendar.day_name[bowler.iat[row, index_3].weekday()]
dataset=pd.read_csv('/kaggle/input/project/personal_male.csv')
#function to extract Initials of players' name.
#We are doing this in order to match it with the names column of our original dataset.

def name(s): 
  
    # split the string into a list  
    l = s.split() 
    new = "" 
  
    # traverse in the list  
    for i in range(len(l)-1): 
        s = l[i] 
          
        # adds the capital first character  
        new += (s[0].upper()) 
          
    # l[-1] gives last item of list l. We 
    # use title to print first character in 
    # capital. 
    new=new+" "+l[-1].title() 
      
    return new  
      
# Driver code             
index_name=dataset.columns.get_loc("fullName")
dataset['New_name']=""
index_new=dataset.columns.get_loc("New_name")

for row in range(len(dataset)):
    cname=name(dataset.iat[row,index_name])
    dataset.iat[row,index_new]=cname
bowler['Name']=bowler['Innings Player']
dataset['Name']=dataset['New_name']
drop=['name', 'fullName', 'dob', 'country', 'birthPlace', 'nationalTeam',
       'teams','battingStyle', 'New_name']
dataset.drop(drop, axis=1,inplace=True)
#merging with original dataset
bowler=pd.merge(bowler,dataset,on='Name', how='inner')
bowler.drop('Name',axis=1,inplace=True)
bowler
#calculating total balls bowled by a bowler in a particular match. We will use this to create new attributes

bowler['Balls_bowled']=0
index_col1=bowler.columns.get_loc('Balls_bowled')
index_col2=bowler.columns.get_loc('Overs_Bowled')
for row in range(0,len(bowler)):
    balls=bowler.iat[row,index_col2]*6
    bowler.iat[row,index_col1]=int(balls)
#Wickets Haul(FF) column represents number of innings a player has taken more than 3 wickets

bowler['FF']=0
index_col1=bowler.columns.get_loc('FF')
index_col2=bowler.columns.get_loc('Innings_Number')
index_col4=bowler.columns.get_loc('4_Wickets')
index_col5=bowler.columns.get_loc('5_Wickets')
for row in range(0,len(bowler)):
    four=bowler.iat[row,index_col4]
    five=bowler.iat[row,index_col5]
    if four==1:
        n=bowler.iat[row,index_col2]
    if five==1:
        n=bowler.iat[row,index_col2]
    else:
        n=0
    bowler.iat[row,index_col1]=int(n)
def attribute(df,col_name):
    df['Average']=0.0
    index_ba=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Wickets_Taken")
    index_inruns=df.columns.get_loc("Runs_conceded")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=inruns/inumber

    df['Strike_rate']=0.0
    index_ba=df.columns.get_loc("Strike_rate")
    index_in=df.columns.get_loc("Balls_bowled")
    index_inruns=df.columns.get_loc("Wickets_Taken")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=(inruns/inumber)*100  
     
    index_new=df.columns.get_loc(col_name)
    index_ob=df.columns.get_loc("Overs_Bowled")
    index_sr=df.columns.get_loc('Strike_rate')
    index_av=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_ff=df.columns.get_loc("FF")

    for row in range(len(df)):
        f=0.4174*(df.iat[row,index_ob])
        f=f+0.2634*(df.iat[row,index_in])
        f+=0.1602*(df.iat[row,index_sr])
        f+=0.0975*(df.iat[row,index_av])
        f+=0.0615*(df.iat[row,index_ff])

        df.iat[row,index_new]=f
    
    return(df)
    
    
    
g=bowler.groupby('Innings Player')
df=g.sum()
df['consistency']=0.0
df=attribute(df,'consistency')
df['Average_Career']=df['Average']
df['Strike_rate_Career']=df['Strike_rate']
drop=['4_Wickets', '5_Wickets', 'Maidens_Bowled', 'Wickets_Taken',
       'Runs_conceded', 'Innings_Number', 'Overs_Bowled', 'Year', 'Month',
       'Balls_bowled', 'FF','Average', 'Strike_rate']
df.drop(drop,axis=1,inplace=True)
bowler=pd.merge(bowler,df,on='Innings Player', how='inner')
def attribute(df,col_name):
    df['Average']=0.0
    index_ba=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Wickets_Taken")
    index_inruns=df.columns.get_loc("Runs_conceded")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=inruns/inumber

    df['Strike_rate']=0.0
    index_ba=df.columns.get_loc("Strike_rate")
    index_in=df.columns.get_loc("Balls_bowled")
    index_inruns=df.columns.get_loc("Wickets_Taken")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=(inruns/inumber)*100  
     
    index_new=df.columns.get_loc(col_name)
    index_ob=df.columns.get_loc("Overs_Bowled")
    index_sr=df.columns.get_loc('Strike_rate')
    index_av=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_ff=df.columns.get_loc("FF")

    for row in range(len(df)):
        f=0.3269*(df.iat[row,index_ob])
        f=f+0.2846*(df.iat[row,index_in])
        f+=0.1877*(df.iat[row,index_sr])
        f+=0.1210*(df.iat[row,index_av])
        f+=0.0798*(df.iat[row,index_ff])

        df.iat[row,index_new]=f
    
    return(df)
    
    
    
g=bowler.groupby(['Innings Player','Year'])
df=g.sum()
df['form']=0.0
df=attribute(df,'form')
df['Average_Yearly']=df['Average']
df['Strike_rate_Yearly']=df['Strike_rate']
drop=['4_Wickets', '5_Wickets', 'Maidens_Bowled', 'Wickets_Taken',
       'Runs_conceded', 'Innings_Number', 'Overs_Bowled', 'Month',
       'Balls_bowled', 'FF', 'consistency', 'Average_Career',
       'Strike_rate_Career','Average', 'Strike_rate']
df.drop(drop,axis=1,inplace=True)
on=['Innings Player','Year']
bowler=pd.merge(bowler,df,on=on, how='inner')
def attribute(df,col_name):
    df['Average']=0.0
    index_ba=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Wickets_Taken")
    index_inruns=df.columns.get_loc("Runs_conceded")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=inruns/inumber

    df['Strike_rate']=0.0
    index_ba=df.columns.get_loc("Strike_rate")
    index_in=df.columns.get_loc("Balls_bowled")
    index_inruns=df.columns.get_loc("Wickets_Taken")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=(inruns/inumber)*100  
     
    index_new=df.columns.get_loc(col_name)
    index_ob=df.columns.get_loc("Overs_Bowled")
    index_sr=df.columns.get_loc('Strike_rate')
    index_av=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_ff=df.columns.get_loc("FF")

    for row in range(len(df)):
        f=0.3177*(df.iat[row,index_ob])
        f=f+0.3177*(df.iat[row,index_in])
        f+=0.1933*(df.iat[row,index_sr])
        f+=0.1465*(df.iat[row,index_av])
        f+=0.00943*(df.iat[row,index_ff])

        df.iat[row,index_new]=f
    
    return(df)
    
    
    
g=bowler.groupby(['Innings Player','Opposition'])
df=g.sum()
df['opposition']=0.0
df=attribute(df,'opposition')
df['Average_opposition']=df['Average']
df['Strike_rate_opposition']=df['Strike_rate']
drop=['4_Wickets', '5_Wickets', 'Maidens_Bowled', 'Wickets_Taken',
       'Runs_conceded', 'Innings_Number', 'Overs_Bowled', 'Year', 'Month',
       'Balls_bowled', 'FF', 'consistency', 'Average_Career',
       'Strike_rate_Career', 'form', 'Average_Yearly',
       'Strike_rate_Yearly', 'Average', 'Strike_rate']
df.drop(drop,axis=1,inplace=True)
on=['Innings Player','Opposition']
bowler=pd.merge(bowler,df,on=on, how='inner')
def attribute(df,col_name):
    df['Average']=0.0
    index_ba=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Wickets_Taken")
    index_inruns=df.columns.get_loc("Runs_conceded")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=inruns/inumber

    df['Strike_rate']=0.0
    index_ba=df.columns.get_loc("Strike_rate")
    index_in=df.columns.get_loc("Balls_bowled")
    index_inruns=df.columns.get_loc("Wickets_Taken")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=(inruns/inumber)*100  
     
    index_new=df.columns.get_loc(col_name)
    index_ob=df.columns.get_loc("Overs_Bowled")
    index_sr=df.columns.get_loc('Strike_rate')
    index_av=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_ff=df.columns.get_loc("FF")

    for row in range(len(df)):
        f=0.3018*(df.iat[row,index_ob])
        f=f+0.2783*(df.iat[row,index_in])
        f+=0.1836*(df.iat[row,index_sr])
        f+=0.1391*(df.iat[row,index_av])
        f+=0.0972*(df.iat[row,index_ff])

        df.iat[row,index_new]=f
    
    return(df)
    
    
    
g=bowler.groupby(['Innings Player','Ground'])
df=g.sum()
df['venue']=0.0
df=attribute(df,'venue')
df['Average_venue']=df['Average']
df['Strike_rate_venue']=df['Strike_rate']
drop=['4_Wickets', '5_Wickets', 'Maidens_Bowled', 'Wickets_Taken',
       'Runs_conceded', 'Innings_Number', 'Overs_Bowled', 'Year', 'Month',
       'Balls_bowled', 'FF', 'consistency', 'Average_Career',
       'Strike_rate_Career', 'form', 'Average_Yearly', 'Strike_rate_Yearly',
       'opposition', 'Average_opposition', 'Strike_rate_opposition',
       'Average', 'Strike_rate']
df.drop(drop,axis=1,inplace=True)
on=['Innings Player','Ground']
bowler=pd.merge(bowler,df,on=on, how='inner')
bowler.drop("Innings Date",axis=1,inplace=True)
def average(df,col_name):
    dummy=[df]
    for dataset in dummy:
        dataset.loc[dataset[col_name]<=24.99, col_name]=5,
        dataset.loc[(dataset[col_name]>=25.00) & (dataset[col_name]<=29.99), col_name]=4,
        dataset.loc[(dataset[col_name]>=30.00) & (dataset[col_name]<=34.99), col_name]=3,
        dataset.loc[(dataset[col_name]>=35.00) & (dataset[col_name]<=49.99), col_name]=2,
        dataset.loc[(dataset[col_name])>=50,col_name]=1
    


average(bowler,'Average_Career')
average(bowler,'Average_Yearly')
average(bowler,'Average_opposition')
average(bowler,'Average_venue')  
def SR(df,col_name):
    dummy=[df]
    for dataset in dummy:
        dataset.loc[dataset[col_name]<=29.99, col_name]=5,
        dataset.loc[(dataset[col_name]>=30.00) & (dataset[col_name]<=39.99), col_name]=4,
        dataset.loc[(dataset[col_name]>=40.00) & (dataset[col_name]<=49.99), col_name]=3,
        dataset.loc[(dataset[col_name]>=50.00) & (dataset[col_name]<=59.99), col_name]=2,
        dataset.loc[(dataset[col_name])>=60,col_name]=1
        
    
SR(bowler,'Strike_rate_Career')
SR(bowler,'Strike_rate_Yearly')
SR(bowler,'Strike_rate_opposition')
SR(bowler,'Strike_rate_venue')
dummy=[bowler]

for dataset in dummy:
    dataset.loc[dataset['consistency']<=99, 'consistency']=1,
    dataset.loc[(dataset['consistency']>99) & (dataset['consistency']<=249), 'consistency']=2,
    dataset.loc[(dataset['consistency']>249) & (dataset['consistency']<=499), 'consistency']=3,
    dataset.loc[(dataset['consistency']>499) & (dataset['consistency']<=1000), 'consistency']=4,
    dataset.loc[dataset['consistency']>1000, 'consistency']=5 
dummy=[bowler]
for dataset in dummy:
    dataset.loc[dataset['form']<=9, 'form']=1,
    dataset.loc[(dataset['form']>9) & (dataset['form']<=24), 'form']=2,
    dataset.loc[(dataset['form']>24) & (dataset['form']<=49), 'form']=3,
    dataset.loc[(dataset['form']>49) & (dataset['form']<=100), 'form']=4,
    dataset.loc[(dataset['form']>100), 'form']=5
    
dummy=[bowler]
for dataset in dummy:
    dataset.loc[dataset['opposition']<=9, 'opposition']=1,
    dataset.loc[(dataset['opposition']>9) & (dataset['opposition']<=24), 'opposition']=2,
    dataset.loc[(dataset['opposition']>24) & (dataset['opposition']<=49), 'opposition']=3,
    dataset.loc[(dataset['opposition']>49) & (dataset['opposition']<=100), 'opposition']=4,
    dataset.loc[dataset['opposition']>100, 'opposition']=5
dummy=[bowler]
for dataset in dummy:
    dataset.loc[dataset['venue']<=9, 'venue']=1,
    dataset.loc[(dataset['venue']>9) & (dataset['venue']<=19), 'venue']=2,
    dataset.loc[(dataset['venue']>19) & (dataset['venue']<=29), 'venue']=3,
    dataset.loc[(dataset['venue']>29) & (dataset['venue']<=39), 'venue']=4,
    dataset.loc[(dataset['venue'])>=39,'venue']=5
dummy=[bowler]
col='Wickets_Taken'
for dataset in dummy:
    dataset.loc[dataset[col]<=1, col]=1,
    dataset.loc[(dataset[col]>1) & (dataset[col]<=3), col]=2,
    dataset.loc[dataset[col]>=4, col]=3,
bowler.to_csv('cricket_bowler_information.csv', header=True, index=False)