import numpy as np 
import pandas as pd 
import re
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset=pd.read_csv('/kaggle/input/cricinfo-statsguru-data/Men ODI Player Innings Stats - 21st Century.csv')
dataset.columns
batsman=dataset[dataset['Innings Overs Bowled'].isnull()]
#removing unnecessary columns
drop=['Innings Overs Bowled',
       'Innings Bowled Flag', 'Innings Maidens Bowled',
       'Innings Runs Conceded', 'Innings Wickets Taken', '4 Wickets',
       '5 Wickets', '10 Wickets', 'Innings Wickets Taken Buckets',
       'Innings Economy Rate','Innings Runs Scored Num', 'Innings Minutes Batted', 'Innings Batted Flag'
     ,'Innings Not Out Flag']
batsman=batsman.drop(drop, axis=1)
batsman['Innings_Runs_Score']=0
batsman=batsman[(batsman['Innings Runs Scored']!='DNB') & (batsman['Innings Runs Scored']!='TDNB')]
#writing regular expressions to extract runs scored
runs = r'([0-9]*)'
index_2=batsman.columns.get_loc('Innings Runs Scored')
index_runs=batsman.columns.get_loc('Innings_Runs_Score')
for row in range(0,len(batsman)):
    run=re.search(runs,batsman.iat[row,index_2]).group()
    if run!='':
        batsman.iat[row,index_runs]=int(run)
#number of 4's

batsman['Innings_Boundary_Fours']=0
batsman['Innings Boundary Fours']= np.where(batsman['Innings Boundary Fours']==' ',
                                            0,batsman['Innings Boundary Fours'])
index_3=batsman.columns.get_loc('Innings Boundary Fours')
index_fours=batsman.columns.get_loc('Innings_Boundary_Fours')
for row in range(0,len(batsman)):
    fours= batsman.iat[row,index_3]
    if fours!='-':
        batsman.iat[row,index_fours]=int(fours)
    
#number of 6's

batsman['Innings_Boundary_Sixes']=0
batsman['Innings Boundary Sixes']=np.where(batsman['Innings Boundary Sixes']==' ',
                                           0,batsman['Innings Boundary Sixes'])
index_3=batsman.columns.get_loc('Innings Boundary Sixes')
index_sixes=batsman.columns.get_loc('Innings_Boundary_Sixes')

for row in range(0,len(batsman)):
    sixes= batsman.iat[row,index_3]
    if sixes!='-':
        batsman.iat[row,index_sixes]=int(sixes)
# current innings strike rate
batsman['Innings_Batting_Strike_Rate']=0.0
index_3=batsman.columns.get_loc('Innings Batting Strike Rate')
index_sr=batsman.columns.get_loc('Innings_Batting_Strike_Rate')

for row in range(0,len(batsman)):
    sr= batsman.iat[row,index_3]
    if sr!='-':
        batsman.iat[row,index_sr]=float(sr)
# Innings played

batsman['Innings_Number']=0
index_3=batsman.columns.get_loc('Innings Number')
index_in=batsman.columns.get_loc('Innings_Number')

for row in range(0,len(batsman)):
    inn= batsman.iat[row,index_3]
    if inn!='-':
        batsman.iat[row,index_in]=int(inn)
#Innings Balls Faced
batsman['Innings_Balls_Faced']=0
index_3=batsman.columns.get_loc('Innings Balls Faced')
index_in=batsman.columns.get_loc('Innings_Balls_Faced')

for row in range(0,len(batsman)):
    inn= batsman.iat[row,index_3]
    if inn!='-':
        batsman.iat[row,index_in]=int(inn)
#Extracting names of Opposition teams

index_3=batsman.columns.get_loc('Opposition')
opp = r'[^v][A-Z]+[a-z]*[" "]*[A-Z]*[a-z]*'
for row in range(0,len(batsman)):
    opps=re.search(opp,batsman.iat[row,index_3]).group()
    batsman.iat[row,index_3]=opps
#Extracting year of match from date feature

batsman['Year']=0
years=r'([0-9]{4})'
index_3=batsman.columns.get_loc('Innings Date')
index_year=batsman.columns.get_loc('Year')

for row in range(0,len(batsman)):
    year=re.search(years,batsman.iat[row,index_3]).group()
    batsman.iat[row,index_year]=int(year)
#Extracting month in which match was played from date feature

batsman['Month']=0
batsman['Innings Date']=pd.to_datetime(batsman['Innings Date'])
index_month=batsman.columns.get_loc('Month')

for row in range(0,len(batsman)):
    batsman.iat[row,index_month]=int(batsman.iat[row, index_3].month)
#Extracting day name from date feature

batsman['Day']=''
index_day=batsman.columns.get_loc('Day')

import calendar
for row in range(0,len(batsman)):
    batsman.iat[row,index_day]=calendar.day_name[batsman.iat[row, index_3].weekday()]
#number of 50s and 100s scored

batsman['50s']=0
batsman['100s']=0

fifty=batsman.columns.get_loc('50s')
hundred=batsman.columns.get_loc('100s')

index_fifty=batsman.columns.get_loc("50's")
index_hundred=batsman.columns.get_loc("100's")

for row in range(0,len(batsman)):
    fifties= batsman.iat[row,index_fifty]
    hundreds=batsman.iat[row,index_hundred]
    if fifties!='-':
        batsman.iat[row,fifty]=int(fifties)
    if hundreds!='-':
        batsman.iat[row,hundred]=int(hundreds)
#Numbers of zeroes scored in innings is necessary for further feature engineering.
#Formula used for creating this feature:
# Number of zeroes = 0 + Number of innings played (if player has scored 0 runs in total)

batsman['0s']=0
index_0=batsman.columns.get_loc('0s')
index_runs=batsman.columns.get_loc('Innings_Runs_Score')
index_inn=batsman.columns.get_loc('Innings_Number')
zeros=0

for row in range(len(batsman)):
    if batsman.iat[row,index_runs]==0:
        zeros=0+batsman.iat[row,index_inn]
    batsman.iat[row,index_0]=zeros
drop=['Innings Runs Scored', 'Innings Balls Faced','Innings Boundary Fours',
       'Innings Boundary Sixes', 'Innings Batting Strike Rate',
       'Innings Number','Innings Date', "50's", "100's"]
batsman=batsman.drop(drop, axis=1)
#creating batting average

batsman['Batting_Average']=0.0
index_ba=batsman.columns.get_loc("Batting_Average")
index_in=batsman.columns.get_loc("Innings_Number")
index_inruns=batsman.columns.get_loc("Innings_Runs_Score")

for row in range(len(batsman)):
    inumber=batsman.iat[row,index_in]
    inruns=batsman.iat[row,index_inruns]
    batsman.iat[row,index_ba]=inruns/inumber
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
batsman['Name']=batsman['Innings Player']
dataset['Name']=dataset['New_name']
drop=['name', 'fullName', 'dob', 'country', 'birthPlace', 'nationalTeam',
       'teams','bowlingStyle', 'New_name']
dataset.drop(drop, axis=1,inplace=True)
#merging with original dataset
batsman=pd.merge(batsman,dataset,on='Name', how='inner')
def attribute(df,col_name):
    df['Average']=0.0
    index_ba=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_inruns=df.columns.get_loc("Innings_Runs_Score")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=inruns/inumber

    df['Strike_rate']=0.0
    index_ba=df.columns.get_loc("Strike_rate")
    index_in=df.columns.get_loc("Innings_Balls_Faced")
    index_inruns=df.columns.get_loc("Innings_Runs_Score")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=(inruns/inumber)*100  
     
    index_new=df.columns.get_loc(col_name)
    index_sr=df.columns.get_loc("Strike_rate")
    index_av=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_100=df.columns.get_loc("100s")
    index_50=df.columns.get_loc('50s')
    index_0=df.columns.get_loc('0s')

    for row in range(len(df)):
        f=0.4262*(df.iat[row,index_av])
        f=f+0.2566*(df.iat[row,index_in])
        f+=0.1510*(df.iat[row,index_sr])
        f+=0.0787*(df.iat[row,index_100])
        f+=0.0556*(df.iat[row,index_50])
        f=f-(0.0328*(df.iat[row,index_0]))
        df.iat[row,index_new]=f
    
    return(df)
    
    
    
g=batsman.groupby('Innings Player')
df=g.sum()
df['consistency']=0.0
df=attribute(df,'consistency')
df['Average_Career']=df['Average']
df['Strike_rate_Career']=df['Strike_rate']
drop=['Innings_Runs_Score','Innings_Boundary_Fours',
       'Innings_Boundary_Sixes', 'Innings_Batting_Strike_Rate',
       'Innings_Number', 'Innings_Balls_Faced', 'Year', 'Month', '50s', '100s',
       '0s', 'Batting_Average','Average','Strike_rate']
df.drop(drop,axis=1,inplace=True)
batsman=pd.merge(batsman,df,on='Innings Player', how='inner')
def attribute(df,col_name):
    df['Average']=0.0
    index_ba=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_inruns=df.columns.get_loc("Innings_Runs_Score")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=inruns/inumber

    df['Strike_rate']=0.0
    index_ba=df.columns.get_loc("Strike_rate")
    index_in=df.columns.get_loc("Innings_Balls_Faced")
    index_inruns=df.columns.get_loc("Innings_Runs_Score")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=(inruns/inumber)*100  
     
    index_new=df.columns.get_loc(col_name)
    index_sr=df.columns.get_loc("Strike_rate")
    index_av=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_100=df.columns.get_loc("100s")
    index_50=df.columns.get_loc('50s')
    index_0=df.columns.get_loc('0s')

    for row in range(len(df)):
        f=0.4262*(df.iat[row,index_av])
        f=f+0.2566*(df.iat[row,index_in])
        f+=0.1510*(df.iat[row,index_sr])
        f+=0.0787*(df.iat[row,index_100])
        f+=0.0556*(df.iat[row,index_50])
        f=f-(0.0328*(df.iat[row,index_0]))
        df.iat[row,index_new]=f
    
    return(df)
    


g=batsman.groupby(['Innings Player','Year'])
df=g.sum()
df['form']=0.0
df=attribute(df,'form')
df['Average_Yearly']=df['Average']
df['Strike_rate_Yearly']=df['Strike_rate']
drop=['Innings_Runs_Score','Innings_Boundary_Fours',
       'Innings_Boundary_Sixes', 'Innings_Batting_Strike_Rate',
       'Innings_Number', 'Innings_Balls_Faced', 'Month', '50s', '100s',
       '0s', 'Batting_Average','consistency','Average_Career','Strike_rate_Career','Average','Strike_rate']
df.drop(drop,axis=1,inplace=True)
on=['Innings Player','Year']
batsman=pd.merge(batsman,df,on=on, how='inner')
def attribute(df,col_name):
    df['Average']=0.0
    index_ba=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_inruns=df.columns.get_loc("Innings_Runs_Score")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=inruns/inumber

    df['Strike_rate']=0.0
    index_ba=df.columns.get_loc("Strike_rate")
    index_in=df.columns.get_loc("Innings_Balls_Faced")
    index_inruns=df.columns.get_loc("Innings_Runs_Score")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=(inruns/inumber)*100  
     
    index_new=df.columns.get_loc(col_name)
    index_sr=df.columns.get_loc("Strike_rate")
    index_av=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_100=df.columns.get_loc("100s")
    index_50=df.columns.get_loc('50s')
    index_0=df.columns.get_loc('0s')

    for row in range(len(df)):
        f=0.4262*(df.iat[row,index_av])
        f=f+0.2566*(df.iat[row,index_in])
        f+=0.1510*(df.iat[row,index_sr])
        f+=0.0787*(df.iat[row,index_100])
        f+=0.0556*(df.iat[row,index_50])
        f=f-(0.0328*(df.iat[row,index_0]))
        df.iat[row,index_new]=f
    
    return(df)
    


g=batsman.groupby(['Innings Player','Opposition'])
df=g.sum()
df['opposition']=0.0
df=attribute(df,'opposition')
df['Average_Opposition']=df['Average']
df['Strike_rate_Opposition']=df['Strike_rate']
drop=['Innings_Runs_Score','Innings_Boundary_Fours',
       'Innings_Boundary_Sixes', 'Innings_Batting_Strike_Rate',
       'Innings_Number', 'Innings_Balls_Faced', 'Year', 'Month', '50s', '100s',
       '0s', 'Batting_Average','consistency', 'form','Average_Career','Strike_rate_Career','Average_Yearly',
     'Strike_rate_Yearly','Average','Strike_rate']
df.drop(drop,axis=1,inplace=True)
on=['Innings Player','Opposition']
batsman=pd.merge(batsman,df,on=on, how='inner')
def attribute(df,col_name):
    df['Average']=0.0
    index_ba=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_inruns=df.columns.get_loc("Innings_Runs_Score")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=inruns/inumber

    df['Strike_rate']=0.0
    index_ba=df.columns.get_loc("Strike_rate")
    index_in=df.columns.get_loc("Innings_Balls_Faced")
    index_inruns=df.columns.get_loc("Innings_Runs_Score")
    for row in range(len(df)):
        inumber=df.iat[row,index_in]
        inruns=df.iat[row,index_inruns]
        df.iat[row,index_ba]=(inruns/inumber)*100  
     
    index_new=df.columns.get_loc(col_name)
    index_sr=df.columns.get_loc("Strike_rate")
    index_av=df.columns.get_loc("Average")
    index_in=df.columns.get_loc("Innings_Number")
    index_100=df.columns.get_loc("100s")
    index_50=df.columns.get_loc('50s')
    index_HS=df.columns.get_loc('Innings_Runs_Score')

    for row in range(len(df)):
        f=0.4262*(df.iat[row,index_av])
        f=f+0.2566*(df.iat[row,index_in])
        f+=0.1510*(df.iat[row,index_sr])
        f+=0.0787*(df.iat[row,index_100])
        f+=0.0556*(df.iat[row,index_50])
        f=f+(0.0328*(df.iat[row,index_HS]))
        df.iat[row,index_new]=f
    
    return(df)
    


g=batsman.groupby(['Innings Player','Ground'])
df=g.max()
df['venue']=0.0
df=attribute(df,'venue')
df['Average_venue']=df['Average']
df['Strike_rate_venue']=df['Strike_rate']
drop=['Innings_Runs_Score','Innings Runs Scored Buckets','Innings_Boundary_Fours',
       'Innings_Boundary_Sixes', 'Innings_Batting_Strike_Rate',
       'Innings_Number', 'Innings_Balls_Faced', 'Year', 'Month', '50s', '100s',
       '0s', 'Batting_Average','consistency', 'form','Average_Career','Strike_rate_Career','Average_Yearly',
     'Strike_rate_Yearly','Average','Strike_rate','opposition','Average_Opposition','Strike_rate_Opposition']
df.drop(drop,axis=1,inplace=True)
on=['Innings Player','Ground']
batsman=pd.merge(batsman,df,on=on, how='inner')
g=batsman.groupby(['Innings Player'])
df=g.sum()
drop=['Innings_Runs_Score','Innings_Boundary_Fours',
       'Innings_Boundary_Sixes', 'Innings_Batting_Strike_Rate',
       'Year', 'Month', 'Batting_Average','consistency', 'form','Average_Career','Strike_rate_Career','Average_Yearly',
     'Strike_rate_Yearly','opposition','Average_Opposition','venue','Strike_rate_Opposition','Average_venue','Strike_rate_venue']
df.drop(drop,axis=1,inplace=True)
on=['Innings Player']
batsman=pd.merge(batsman,df,on=on, how='inner')
batsman['50s']=batsman['50s_y']
batsman['100s']=batsman['100s_y']
batsman['0s']=batsman['0s_y']
batsman['Innings_Balls_Faced']=batsman['Innings_Balls_Faced_y']
batsman['Innings_Number']=batsman['Innings_Number_y']

drop=['Opposition_y','Innings_Boundary_Fours','Innings_Boundary_Sixes','Day_y','Country_y','50s_x',
      '100s_x','0s_x','Innings_Balls_Faced_x','Innings_Number_x','50s_y',
      '100s_y','0s_y','Innings_Balls_Faced_y','Innings_Number_y',
      'battingStyle_y','Name_y']
batsman.drop(drop,axis=1,inplace=True)
dummy=[batsman]

for dataset in dummy:
    dataset.loc[dataset['consistency']<=49, 'consistency']=1,
    dataset.loc[(dataset['consistency']>49) & (dataset['consistency']<=99), 'consistency']=2,
    dataset.loc[(dataset['consistency']>99) & (dataset['consistency']<=124), 'consistency']=3,
    dataset.loc[(dataset['consistency']>124) & (dataset['consistency']<=149), 'consistency']=4,
    dataset.loc[dataset['consistency']>149, 'consistency']=5    
dummy=[batsman]
for dataset in dummy:
    dataset.loc[dataset['form']<=4, 'form']=1,
    dataset.loc[(dataset['form']>4) & (dataset['form']<=9), 'form']=2,
    dataset.loc[(dataset['form']>9) & (dataset['form']<=11), 'form']=3,
    dataset.loc[(dataset['form']>11) & (dataset['form']<=14), 'form']=4,
    dataset.loc[(dataset['form']>14), 'form']=5
dummy=[batsman]
for dataset in dummy:
    dataset.loc[dataset['opposition']<=2, 'opposition']=1,
    dataset.loc[(dataset['opposition']>2) & (dataset['opposition']<=4), 'opposition']=2,
    dataset.loc[(dataset['opposition']>4) & (dataset['opposition']<=6), 'opposition']=3,
    dataset.loc[(dataset['opposition']>6) & (dataset['opposition']<=9), 'opposition']=4,
    dataset.loc[dataset['opposition']>9, 'opposition']=5
    
dummy=[batsman]
for dataset in dummy:
    dataset.loc[dataset['venue']<=1, 'venue']=1,
    dataset.loc[(dataset['venue']>1) & (dataset['venue']<=2), 'venue']=2,
    dataset.loc[(dataset['venue']>2) & (dataset['venue']<=3), 'venue']=3,
    dataset.loc[(dataset['venue']>3) & (dataset['venue']<=4), 'venue']=4,
    dataset.loc[(dataset['venue'])>=5,'venue']=5
def average(df,col_name):
    dummy=[df]
    for dataset in dummy:
        dataset.loc[dataset[col_name]<=9.99, col_name]=1,
        dataset.loc[(dataset[col_name]>=10.00) & (dataset[col_name]<=19.99), col_name]=2,
        dataset.loc[(dataset[col_name]>=20.00) & (dataset[col_name]<=29.99), col_name]=3,
        dataset.loc[(dataset[col_name]>=30.00) & (dataset[col_name]<=39.99), col_name]=4,
        dataset.loc[(dataset[col_name])>=40,col_name]=5
    

average(batsman,'Batting_Average')
average(batsman,'Average_Career')
average(batsman,'Average_Yearly')
average(batsman,'Average_Opposition')
average(batsman,'Average_venue')  
def SR(df,col_name):
    dummy=[df]
    for dataset in dummy:
        dataset.loc[dataset[col_name]<=49.99, col_name]=1,
        dataset.loc[(dataset[col_name]>=50.00) & (dataset[col_name]<=59.99), col_name]=2,
        dataset.loc[(dataset[col_name]>=60.00) & (dataset[col_name]<=79.99), col_name]=3,
        dataset.loc[(dataset[col_name]>=80.00) & (dataset[col_name]<=100), col_name]=4,
        dataset.loc[(dataset[col_name])>100,col_name]=5
        
    
SR(batsman,'Innings_Batting_Strike_Rate')
SR(batsman,'Strike_rate_Career')
SR(batsman,'Strike_rate_Yearly')
SR(batsman,'Strike_rate_Opposition')
SR(batsman,'Strike_rate_venue')
dummy=[batsman]
col='Innings_Runs_Score'
for dataset in dummy:
    dataset.loc[dataset[col]<=24, col]=1,
    dataset.loc[(dataset[col]>24) & (dataset[col]<=49), col]=2,
    dataset.loc[(dataset[col]>=50) & (dataset[col]<=74), col]=3,
    dataset.loc[(dataset[col]>74) & (dataset[col]<=99), col]=4,
    dataset.loc[(dataset[col])>=100,col]=5
batsman.to_csv('cricket_batsman_information.csv', header=True, index=False)