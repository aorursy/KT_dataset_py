import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
#Load data
df=pd.read_csv('../input/yugioh-tcg/original/original/cards.csv',header=None,index_col=0)
#Quick inspection. There are no column names!
df.head()
#Check for duplicates.
#One card is in there 30 times!
df[1].value_counts().head()
#Remove duplicates & check how many
print(df.shape)
df=df.drop_duplicates()
print(df.shape)
#There were apparentlly ~1800 duplicate entries
#Check I've not missed anything
df[1].value_counts().head()
#Problem 2: Missing passcodes
#None is a legitimate entry.
#'Monster' seems to have been a mistake & the column was missed off
#Not sure why the passcodes along are duplicated, so will check.
df[2].value_counts().head()
#'Monster' entries have the password missing, so want to correct that by shifting the rows
#A quick inspection showed some of these rows have plenty of other problems too
df[df[2]=='Monster'].head()
#Save the names, so I can add passcodes later
passcode_names=df[df[2]=='Monster'][1]
#Shift the rows along by 1 where the passcode is missing.
#The rest of the rows are also a big mess, with other data out of order.
#To avoid pushing it out of the dataframe, I added a dummy column to put it in for now
mask = df[2] == 'Monster'
df['Dummy']=""
c = [2,3,4,5,6,7,8,9,10,11,12,13,'Dummy']
#shift columns
df.loc[mask, c] = df.loc[mask, c].shift(1, axis=1)
#Something to save me time / effort
for i in list(passcode_names):
    print('df.loc[df[1]==\"'+i+'\",2]')
df.loc[df[1]=="Performapal Card Gardna",2]='37256334'
df.loc[df[1]=="D/D/D Destiny King Zero Laplace",2]='21686473'
df.loc[df[1]=="Odd-Eyes Wing Dragon",2]='58074177'
df.loc[df[1]=="D/D/D Superdoom King Purplish Armageddon",2]='84569886'
df.loc[df[1]=="SPYRAL Sleeper",2]='00035699'
df.loc[df[1]=="Subterror Fiendess",2]='74762582'

#Tokens are not real cards, so drop
df=df[df[1]!="Ancient Gear Token"]

df.loc[df[1]=="SPYRAL GEAR - Last Resort",2]='37433748'
df.loc[df[1]=="Subterror Behemoth Phospheroglacier",2]='01151281'
df.loc[df[1]=="Subterror Behemoth Speleogeist",2]='47556396'
df.loc[df[1]=="Link Disciple",2]='32995276'

#These 2 seem to be prize cards
df.loc[df[1]=="Iron Knight of Revolution",2]=np.nan
df.loc[df[1]=="Sanctity of Dragon",2]=np.nan


df.loc[df[1]=="Hallohallo",2]='77994337'
df.loc[df[1]=="Mudragon of the Swamp",2]='54757758'

#Tokens are not real cards, so drop
df=df[df[1]!="Token"]


df.loc[df[1]=="Heavymetalfoes Electrumite",2]='24094258'
#Duplicate passcodes is because of name changes & spelling errors
#This seems to be the case for 22 cards.
#Some are actually just passcode errors.
#Unfortunately, therefore this will probably need to be fixed manually.
df[2].value_counts().head(22)
#Corrections

#Name change & spelling error
df.drop(df[df[2]=='62279666'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='62279666'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='48152161'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='12097275'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='97273514'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='40854824'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='58374719'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='07969770'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='43464884'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='99674361'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='87475570'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='87259933'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='93236220'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='50548657'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='96150936'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='25163979'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='01735088'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='14469229'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='85763457'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='08192327'].iloc[[0]].index,inplace=True)
df.drop(df[df[2]=='33541430'].iloc[[0]].index,inplace=True)

#Passcode error
df.loc[df[1]=="Metaphys Ragnarok",2]='19476824'
#One more check
df[2].value_counts().head()
df[3].value_counts()
df[4].value_counts()
#Check for any problems
df[5].value_counts()
#Shift values
mask=~df[5].isin(['1','2','3','4','5','6','7','8','9','10','11','12',np.nan])
c = [5,6,7,8,9,10,11,12,13,'Dummy']
#shift columns
df.loc[mask, c] = df.loc[mask, c].shift(1, axis=1)
#Check everything is now correct
df[5].value_counts()
df[6].value_counts().head()
#Splitting the string into new columns
str_df = df[6].str.split('/',expand=True).add_prefix("Card Type ")
df=pd.concat([df, str_df], axis=1).replace({None:np.NaN})
#Reorder columns & drop the original column
cols=[1,2,3,4,5,6,'Card Type 0', 'Card Type 1',
       'Card Type 2', 'Card Type 3',7,8,9,10,11,12,13,'Dummy']
df=df[cols]
df.drop(6,axis=1,inplace=True)
#Now all these columns need to be re-checked
#Nothing looks like an error
Type0=df['Card Type 0'].value_counts()
Type1=df['Card Type 1'].value_counts()
Type2=df['Card Type 2'].value_counts()
Type3=df['Card Type 3'].value_counts()

print(Type0.index)
print(Type1.index)
print(Type2.index)
print(Type3.index)
df[7].value_counts().head()
#Split into two columns
str_df = df[7].str.split('/',expand=True).add_prefix("Stat ")
df=pd.concat([df, str_df], axis=1).replace({None:np.NaN})
#Reorder & rename columns
cols=[1,2,3,4,5,'Card Type 0', 'Card Type 1',
       'Card Type 2', 'Card Type 3','Stat 0','Stat 1',8,9,10,11,12,13,'Dummy',7]
df=df[cols]
df.drop(7,axis=1,inplace=True)
df.rename(columns={'Stat 0':'Attack','Stat 1':'Defense'},inplace=True)
#Check the attack
df.Attack.value_counts().index
# Both ? and X000 exist as non-integer values
#These will be converted to NAN for now
df.loc[df['Attack']=="?",'Attack']=np.nan
df.loc[df['Attack']=="X000",'Attack']=np.nan
#A small number of cards have their scales in the stat slot, which is wrong.
#This will need to be corrected. 
#Thanfully, this data is (seemingly randomly) in one of the other columns

#Scale 1
df.loc[df[1]=="D/D/D Destiny King Zero Laplace",'Attack']=np.nan
df.loc[df[1]=="D/D/D Destiny King Zero Laplace",'Defense']='0'
df.loc[df[1]=="D/D/D Superdoom King Purplish Armageddon",'Attack']='3500'
df.loc[df[1]=="D/D/D Superdoom King Purplish Armageddon",'Defense']='3000'
df.loc[df[1]=="Yoko-Zuna Sumo Spirit",'Attack']='2400'
df.loc[df[1]=="Yoko-Zuna Sumo Spirit",'Defense']='1000'

#Scale 2
df.loc[df[1]=="Foucault's Cannon",'Attack']='2200'
df.loc[df[1]=="Foucault's Cannon",'Defense']='1200'
df.loc[df[1]=="Mandragon",'Attack']='2500'
df.loc[df[1]=="Mandragon",'Defense']='1000'
df.loc[df[1]=="Risebell the Summoner",'Attack']='800'
df.loc[df[1]=="Risebell the Summoner",'Defense']='800'
df.loc[df[1]=="Hallohallo",'Attack']='800'
df.loc[df[1]=="Hallohallo",'Defense']='600'

#Scale 3
df.loc[df[1]=="Dragon Horn Hunter",'Attack']='2300'
df.loc[df[1]=="Dragon Horn Hunter",'Defense']='1000'
df.loc[df[1]=="Magical Abductor",'Attack']='1700'
df.loc[df[1]=="Magical Abductor",'Defense']='1400'
df.loc[df[1]=="Samurai Cavalry of Reptier",'Attack']='1800'
df.loc[df[1]=="Samurai Cavalry of Reptier",'Defense']='1200'

#Scale 4
df.loc[df[1]=="Ghost Beef",'Attack']='2000'
df.loc[df[1]=="Ghost Beef",'Defense']='1000'
df.loc[df[1]=="Metrognome",'Attack']='1800'
df.loc[df[1]=="Metrognome",'Defense']='1600'
df.loc[df[1]=="Pandora's Jewelry Box",'Attack']='1500'
df.loc[df[1]=="Pandora's Jewelry Box",'Defense']='1500'

#Scale 5
df.loc[df[1]=="P.M. Captor",'Attack']='1800'
df.loc[df[1]=="P.M. Captor",'Defense']='0'
df.loc[df[1]=="Steel Cavalry of Dinon",'Attack']='1600'
df.loc[df[1]=="Steel Cavalry of Dinon",'Defense']='2600'

#Scale 7
df.loc[df[1]=="Dragong",'Attack']='500'
df.loc[df[1]=="Dragong",'Defense']='2100'
df.loc[df[1]=="Flash Knight",'Attack']='1800'
df.loc[df[1]=="Flash Knight",'Defense']='600'
df.loc[df[1]=="Lancephorhynchus",'Attack']='2500'
df.loc[df[1]=="Lancephorhynchus",'Defense']='800'
df.loc[df[1]=="Mild Turkey",'Attack']='1000'
df.loc[df[1]=="Mild Turkey",'Defense']='2000'
df.loc[df[1]=="Zany Zebra",'Attack']='0'
df.loc[df[1]=="Zany Zebra",'Defense']='2000'

#Scale 8
df.loc[df[1]=="Performapal Card Gardna",'Attack']='1000'
df.loc[df[1]=="Performapal Card Gardna",'Defense']='1000'

#Scale 9
df.loc[df[1]=="Kuro-Obi Karate Spirit",'Attack']='2400'
df.loc[df[1]=="Kuro-Obi Karate Spirit",'Defense']='1000'
df.loc[df[1]=="Kai-Den Kendo Spirit",'Attack']='2400'
df.loc[df[1]=="Kai-Den Kendo Spirit",'Defense']='1000'

#Scale 10
df.loc[df[1]=="Odd-Eyes Wing Dragon",'Attack']='3000'
df.loc[df[1]=="Odd-Eyes Wing Dragon",'Defense']='2500'

test_df=df.loc[df['Attack']=='0']
test_df.loc[test_df['Card Type 1']=='Pendulum']
#Check defense column
df.Defense.value_counts().index
# Both ? and X000 exist as non-integer values
#These will be converted to NAN for now
df.loc[df['Defense']=="?",'Defense']=np.nan
df.loc[df['Defense']=="X000",'Defense']=np.nan
#Let's take a moment to rename some columns
df.rename(columns={1:'Name',2:'Passcode',3:'Category',4:'Attribute',5: 'Level',
                  'Card Type 0':'Type'},inplace=True)
df.head()
#Columns to write
header=['Name','Passcode','Category','Attribute','Level','Type','Card Type 1','Card Type 2',
        'Card Type 3','Attack','Defense']
#Full DF
df.to_csv('YGO_partial.csv',columns=header,index=False)
#Monsters only
df_monster=df[df['Category']=='Monster']
#Monsters only
df_monster.to_csv('YGO_Monster_partial.csv',columns=header,index=False)
YGO_df=pd.read_csv('../input/ygo-data/YGO_Cards_v2.csv',encoding = "ISO-8859-1")
YGO_P_df=pd.read_csv('../input/ygo-prices-data/YGO_Cards_v3.csv',encoding = "ISO-8859-1")
YGO_df.rename(columns={'Unnamed: 0':'Name'},inplace=True)
df_monster[~df_monster.Name.isin(YGO_df.Name.values)].Name
YGO_df[~YGO_df.Name.isin(df_monster.Name.values)].Name
Missing_monster_df1=df_monster[~df_monster.Name.isin(YGO_df.Name.values)]
Missing_monster_df1[~Missing_monster_df1.Passcode.isin(YGO_df.number.values)]
Missing_monster_df2=YGO_df[~YGO_df.Name.isin(df_monster.Name.values)]
len(Missing_monster_df2[~Missing_monster_df2.number.isin(df_monster.Passcode.values)])