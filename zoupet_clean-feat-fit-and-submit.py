import pandas as pd
import numpy as np
import scipy as sc
from sklearn import preprocessing

train_=pd.read_csv('../input/train.csv',delimiter=',')
test_=pd.read_csv('../input/test.csv',delimiter=',')
pd.options.mode.chained_assignment = None
Id_ts=test_['ID']
test_.drop('ID',axis=1,inplace=True)

#Clean Data V1
def clean(tr,ts):
    le = preprocessing.LabelEncoder()
    #Save ID
    Id_tr=tr['AnimalID']
    tr.drop('AnimalID',axis=1,inplace=True)
    
    tr['Aggressive']=0
    tr['At Vet']=0
    tr['Barn']=0
    tr['Behavior']=0
    tr['Court/Investigation']=0
    tr['Enroute']=0
    tr['Foster']=0
    tr['In Foster']=0
    tr['In Kennel']=0
    tr['In Surgery']=0
    tr['Medical']=0
    tr['Offsite']=0
    tr['Partner']=0
    tr['Rabies Risk']=0
    tr['SCRP']=0
    tr['Suffering']=0
    
    #OutcomeSubtype exploit
    tr.ix[tr.OutcomeSubtype == 'Aggressive', 'Aggressive'] = 1
    tr.ix[tr.OutcomeSubtype == 'At Vet', 'At Vet'] = 1
    tr.ix[tr.OutcomeSubtype == 'Barn', 'Barn'] = 1
    tr.ix[tr.OutcomeSubtype == 'Behavior', 'Behavior'] = 1
    tr.ix[tr.OutcomeSubtype == 'Court/Investigation', 'Court/Investigation'] = 1
    tr.ix[tr.OutcomeSubtype == 'Enroute', 'Enroute'] = 1
    tr.ix[tr.OutcomeSubtype == 'Foster', 'Foster'] = 1
    tr.ix[tr.OutcomeSubtype == 'In Foster', 'In Foster'] = 1
    tr.ix[tr.OutcomeSubtype == 'In Kennel', 'In Kennel'] = 1
    tr.ix[tr.OutcomeSubtype == 'In Surgery', 'In Surgery'] = 1
    tr.ix[tr.OutcomeSubtype == 'Medical', 'Medical'] = 1
    tr.ix[tr.OutcomeSubtype == 'Offsite', 'Offsite'] = 1
    tr.ix[tr.OutcomeSubtype == 'Partner', 'Partner'] = 1
    tr.ix[tr.OutcomeSubtype == 'Rabies Risk', 'Rabies Risk'] = 1
    tr.ix[tr.OutcomeSubtype == 'SCRP', 'SCRP'] = 1
    tr.ix[tr.OutcomeSubtype == 'Suffering', 'Suffering'] = 1
    
    global right1
    global right2
    global right3
    global right4
    global right5
    global right6
    global right7
    global right8
    global right9
    global right10
    global right11
    global right12
    global right13
    global right14
    global right15
    global right16
    
    
    right1=pd.DataFrame(tr.groupby(['Breed'])['Aggressive'].sum())
    right2=pd.DataFrame(tr.groupby(['Breed'])['At Vet'].sum())
    right3=pd.DataFrame(tr.groupby(['Breed'])['Barn'].sum())
    right4=pd.DataFrame(tr.groupby(['Breed'])['Behavior'].sum())
    right5=pd.DataFrame(tr.groupby(['Breed'])['Court/Investigation'].sum())
    right6=pd.DataFrame(tr.groupby(['Breed'])['Enroute'].sum())
    right7=pd.DataFrame(tr.groupby(['Breed'])['Foster'].sum())
    right8=pd.DataFrame(tr.groupby(['Breed'])['In Foster'].sum())
    right9=pd.DataFrame(tr.groupby(['Breed'])['In Kennel'].sum())
    right10=pd.DataFrame(tr.groupby(['Breed'])['In Surgery'].sum())
    right11=pd.DataFrame(tr.groupby(['Breed'])['Medical'].sum())
    right12=pd.DataFrame(tr.groupby(['Breed'])['Offsite'].sum())
    right13=pd.DataFrame(tr.groupby(['Breed'])['Partner'].sum())
    right14=pd.DataFrame(tr.groupby(['Breed'])['Rabies Risk'].sum())
    right15=pd.DataFrame(tr.groupby(['Breed'])['SCRP'].sum())
    right16=pd.DataFrame(tr.groupby(['Breed'])['Suffering'].sum())
    
    #Name Binarizer
    tr['Name_bool']=pd.isnull(tr['Name']).astype(int)
    ts['Name_bool']=pd.isnull(ts['Name']).astype(int)
    
    #OutcomeType
    tr.ix[tr.OutcomeType == 'Return_to_owner', 'Target'] = 0
    tr.ix[tr.OutcomeType == 'Euthanasia', 'Target'] = 1
    tr.ix[tr.OutcomeType == 'Adoption', 'Target'] = 2
    tr.ix[tr.OutcomeType == 'Transfer', 'Target'] = 3
    tr.ix[tr.OutcomeType == 'Died', 'Target'] = 4
    
    #Animal Binarizer
    tr.ix[tr.AnimalType == 'Dog', 'AnimalType_cat'] = 0
    tr.ix[tr.AnimalType == 'Cat', 'AnimalType_cat'] = 1
    
    ts.ix[ts.AnimalType == 'Dog', 'AnimalType_cat'] = 0
    ts.ix[ts.AnimalType == 'Cat', 'AnimalType_cat'] = 1
    
    #SexuponOutcome
    tr.ix[tr.SexuponOutcome == 'Intact Female', 'SexuponOutcome_cat'] = 0
    tr.ix[tr.SexuponOutcome == 'Spayed Female', 'SexuponOutcome_cat'] = 1
    tr.ix[tr.SexuponOutcome == 'Neutered Male', 'SexuponOutcome_cat'] = 2
    tr.ix[tr.SexuponOutcome == 'Intact Male', 'SexuponOutcome_cat'] = 3
    tr.ix[tr.SexuponOutcome == 'Unknown', 'SexuponOutcome_cat'] = 4
    
    ts.ix[ts.SexuponOutcome == 'Intact Female', 'SexuponOutcome_cat'] = 0
    ts.ix[ts.SexuponOutcome == 'Spayed Female', 'SexuponOutcome_cat'] = 1
    ts.ix[ts.SexuponOutcome == 'Neutered Male', 'SexuponOutcome_cat'] = 2
    ts.ix[ts.SexuponOutcome == 'Intact Male', 'SexuponOutcome_cat'] = 3
    ts.ix[ts.SexuponOutcome == 'Unknown', 'SexuponOutcome_cat'] = 4
    
    #Male Female
    tr.ix[tr.SexuponOutcome == 'Intact Female', 'Type_sex'] = 1
    tr.ix[tr.SexuponOutcome == 'Spayed Female', 'Type_sex'] = 1
    tr.ix[tr.SexuponOutcome == 'Neutered Male', 'Type_sex'] = 2
    tr.ix[tr.SexuponOutcome == 'Intact Male', 'Type_sex'] = 2
    tr.ix[tr.SexuponOutcome == 'Unknown', 'Type_sex'] = 3
    
    ts.ix[ts.SexuponOutcome == 'Intact Female', 'Type_sex'] = 1
    ts.ix[ts.SexuponOutcome == 'Spayed Female', 'Type_sex'] = 1
    ts.ix[ts.SexuponOutcome == 'Neutered Male', 'Type_sex'] = 2
    ts.ix[ts.SexuponOutcome == 'Intact Male', 'Type_sex'] = 2
    ts.ix[ts.SexuponOutcome == 'Unknown', 'Type_sex'] = 3
  
    #Top sprayed
    tr.ix[tr.SexuponOutcome == 'Intact Female', 'sprayed'] = 1
    tr.ix[tr.SexuponOutcome == 'Spayed Female', 'sprayed'] = 2
    tr.ix[tr.SexuponOutcome == 'Neutered Male', 'sprayed'] = 2
    tr.ix[tr.SexuponOutcome == 'Intact Male', 'sprayed'] = 1
    tr.ix[tr.SexuponOutcome == 'Unknown', 'sprayed'] = 3
    
    ts.ix[ts.SexuponOutcome == 'Intact Female', 'sprayed'] = 1
    ts.ix[ts.SexuponOutcome == 'Spayed Female', 'sprayed'] = 2
    ts.ix[ts.SexuponOutcome == 'Neutered Male', 'sprayed'] = 2
    ts.ix[ts.SexuponOutcome == 'Intact Male', 'sprayed'] = 1
    ts.ix[ts.SexuponOutcome == 'Unknown', 'sprayed'] = 3
    
    #AgeuponOutcome split + old--young
    tr['AgeuponOutcome_num']=0
    tr['AgeuponOutcome_lad']=''
    
    ts['AgeuponOutcome_num']=0
    ts['AgeuponOutcome_lad']=''
    
    for i in range(0,tr.shape[0]):
        try:
            tr['AgeuponOutcome_num'][i]=tr['AgeuponOutcome'][i].split()[0]
        except AttributeError:
            tr['AgeuponOutcome_num'][i]=0
        try:
            tr['AgeuponOutcome_lad'][i]=tr['AgeuponOutcome'][i].split()[1]
        except AttributeError:
            tr['AgeuponOutcome_lad'][i]='day'
        
    for i in range(0,ts.shape[0]):
        try:
            ts['AgeuponOutcome_num'][i]=ts['AgeuponOutcome'][i].split()[0]
        except AttributeError:
            ts['AgeuponOutcome_num'][i]=0
        try:
            ts['AgeuponOutcome_lad'][i]=ts['AgeuponOutcome'][i].split()[1]
        except AttributeError:
            ts['AgeuponOutcome_lad'][i]='day'
    
    tr.replace('year','years',inplace=True)
    tr.replace('day','days',inplace=True)
    tr.replace('month','months',inplace=True)
    tr.replace('week','weeks',inplace=True)
    ts.replace('year','years',inplace=True)
    ts.replace('day','days',inplace=True)
    ts.replace('month','months',inplace=True)
    ts.replace('week','weeks',inplace=True)
    
    for i in range(0,tr.shape[0]):
            if tr['AgeuponOutcome_lad'][i]=='years':
                tr['AgeuponOutcome_num'][i]=tr['AgeuponOutcome_num'][i]*365
            if tr['AgeuponOutcome_lad'][i]=='months':
                tr['AgeuponOutcome_num'][i]=tr['AgeuponOutcome_num'][i]*30
            if tr['AgeuponOutcome_lad'][i]=='weeks':
                tr['AgeuponOutcome_num'][i]=tr['AgeuponOutcome_num'][i]*7
    for i in range(0,ts.shape[0]):
            if ts['AgeuponOutcome_lad'][i]=='years':
                ts['AgeuponOutcome_num'][i]=ts['AgeuponOutcome_num'][i]*365
            if ts['AgeuponOutcome_lad'][i]=='months':
                ts['AgeuponOutcome_num'][i]=ts['AgeuponOutcome_num'][i]*30
            if ts['AgeuponOutcome_lad'][i]=='weeks':
                ts['AgeuponOutcome_num'][i]=ts['AgeuponOutcome_num'][i]*7
    
    tr['AgeuponOutcome_cat']=2
    ts['AgeuponOutcome_cat']=2
    
    for i in range(0,tr.shape[0]):
        if tr['AgeuponOutcome_num'][i]<=480:
            tr['AgeuponOutcome_cat'][i]=1
        if 480<tr['AgeuponOutcome_num'][i]<=2555:
            tr['AgeuponOutcome_cat'][i]=2
        if 2555<tr['AgeuponOutcome_num'][i]:
            tr['AgeuponOutcome_cat'][i]=3
    for i in range(0,ts.shape[0]):
        if ts['AgeuponOutcome_num'][i]<=480:
            ts['AgeuponOutcome_cat'][i]=1
        if 480<ts['AgeuponOutcome_num'][i]<2555:
            ts['AgeuponOutcome_cat'][i]=2
        if 2555<ts['AgeuponOutcome_num'][i]:
            ts['AgeuponOutcome_cat'][i]=3            
            
    #Breed split and bin
    tr['Breed1']=''
    tr['Breed2']=''
    tr['Breed11']=''    
    ts['Breed1']=''
    ts['Breed2']=''
    ts['Breed11']=''
    
    for i in range(0,ts.shape[0]):
        try:
            ts['Breed1'][i]=ts['Breed'][i].split('/')[0]
        except IndexError:
            ts['Breed1'][i]=ts['Breed'][i]
        try:
            ts['Breed2'][i]=ts['Breed'][i].split('/')[1]
        except IndexError:
            ts['Breed2'][i]='One_Breed'
    for i in range(0,tr.shape[0]):
        try:
            tr['Breed1'][i]=tr['Breed'][i].split('/')[0]
        except IndexError:
            tr['Breed1'][i]=tr['Breed'][i]
        try:
            tr['Breed2'][i]=tr['Breed'][i].split('/')[1]
        except IndexError:
            tr['Breed2'][i]='One_Breed'

    for i in range(0,ts.shape[0]):
        try:
            ts['Breed11'][i]=ts['Breed1'][i].split(' Mix')[0]
        except IndexError:
            ts['Breed11'][i]=ts['Breed1'][i]
    for i in range(0,tr.shape[0]):
        try:
            tr['Breed11'][i]=tr['Breed1'][i].split(' Mix')[0]
        except IndexError:
            tr['Breed11'][i]=tr['Breed1'][i]            
    
    tr['Breed11']=tr['Breed11'].str.upper()
    ts['Breed11']=ts['Breed11'].str.upper()
    
    #Mix or not Mix
    tr['Top_Mix']=0
    ts['Top_Mix']=0
    for i in range(0,tr.shape[0]):
        if tr['Breed'][i].find(' Mix') >= 0:
            tr['Top_Mix'][i]=1
            
    for i in range(0,ts.shape[0]):        
        if ts['Breed'][i].find(' Mix') >= 0:
            ts['Top_Mix'][i]=1    
        
    
    #Color split
    tr['Color1']=''
    tr['Color2']=''
    tr['Color_prin1']=''
    tr['Shade1']=''
    tr['Color_prin2']=''
    tr['Shade2']=''
    
    ts['Color1']=''
    ts['Color2']=''
    ts['Color_prin1']=''
    ts['Shade1']=''
    ts['Color_prin2']=''
    ts['Shade2']=''
    
    for i in range(0,tr.shape[0]):
        tr['Color1'][i]=tr['Color'][i].split('/')[0]
        try:
            tr['Color2'][i]=tr['Color'][i].split('/')[1]
        except IndexError:
            tr['Color2'][i]=''
        tr['Color_prin1'][i]=tr['Color1'][i].split(' ')[0]
        try:
            tr['Shade1'][i]=tr['Color1'][i].split(' ')[1]
        except IndexError:
            tr['Shade1'][i]=''        
        tr['Color_prin2'][i]=tr['Color2'][i].split(' ')[0]
        try:    
            tr['Shade2'][i]=tr['Color2'][i].split(' ')[1]
        except IndexError:
            tr['Shade2'][i]=''
            
            
    for i in range(0,ts.shape[0]):        
        ts['Color1'][i]=ts['Color'][i].split('/')[0]
        try:
            ts['Color2'][i]=ts['Color'][i].split('/')[1]
        except IndexError:
            ts['Color2'][i]=''
        ts['Color_prin1'][i]=ts['Color1'][i].split(' ')[0]
        try:
            ts['Shade1'][i]=ts['Color1'][i].split(' ')[1]
        except IndexError:
            ts['Shade1'][i]=''        
        ts['Color_prin2'][i]=ts['Color2'][i].split(' ')[0]
        try: 
            ts['Shade2'][i]=ts['Color2'][i].split(' ')[1]
        except IndexError:
            ts['Shade2'][i]=''  

    #UniColor
    tr['Unicolor']=0
    ts['Unicolor']=0
    
    for i in range(0,tr.shape[0]):
         if tr['Color2'][i]=='':
            tr['Unicolor'][i]=1

    for i in range(0,ts.shape[0]):
         if ts['Color2'][i]=='':
            ts['Unicolor'][i]=1
            
    #Datetime
    tr['year'] = pd.DatetimeIndex(tr['DateTime']).year
    tr['month'] = pd.DatetimeIndex(tr['DateTime']).month
    tr['day'] = pd.DatetimeIndex(tr['DateTime']).day
    tr['hour'] = pd.DatetimeIndex(tr['DateTime']).hour
    tr['minute'] = pd.DatetimeIndex(tr['DateTime']).minute
    tr['dayofweek'] = pd.DatetimeIndex(tr['DateTime']).dayofweek
    tr['quarter'] = pd.DatetimeIndex(tr['DateTime']).quarter
    tr['weekofyear'] = pd.DatetimeIndex(tr['DateTime']).weekofyear

    ts['year'] = pd.DatetimeIndex(ts['DateTime']).year
    ts['month'] = pd.DatetimeIndex(ts['DateTime']).month
    ts['day'] = pd.DatetimeIndex(ts['DateTime']).day
    ts['hour'] = pd.DatetimeIndex(ts['DateTime']).hour
    ts['minute'] = pd.DatetimeIndex(ts['DateTime']).minute
    ts['dayofweek'] = pd.DatetimeIndex(ts['DateTime']).dayofweek
    ts['quarter'] = pd.DatetimeIndex(ts['DateTime']).quarter
    ts['weekofyear'] = pd.DatetimeIndex(ts['DateTime']).weekofyear
    
    tr['am_pm']=0
    ts['am_pm']=0
    
    for i in range(0,tr.shape[0]):
        if 8<=tr['hour'][i]<=11:
            tr['am_pm'][i]=1
        if 12<=tr['hour'][i]<=14:
            tr['am_pm'][i]=2
        if 14<=tr['hour'][i]<=18:
            tr['am_pm'][i]=3
        if 18<tr['hour'][i]:
            tr['am_pm'][i]=4
    for i in range(0,ts.shape[0]):
        if 8<=ts['hour'][i]<=11:
            ts['am_pm'][i]=1
        if 12<=ts['hour'][i]<=14:
            ts['am_pm'][i]=2
        if 14<=ts['hour'][i]<=18:
            ts['am_pm'][i]=3
        if 18<tr['hour'][i]:
            ts['am_pm'][i]=4               
    #Season
    tr['Season']=0
    ts['Season']=0
    
    for i in range(0,ts.shape[0]):
            if 4<=ts['month'][i]<=6:
                ts['Season'][i]=1
            if 7<=ts['month'][i]<=9:
                ts['Season'][i]=2
            if 10<=ts['month'][i]<=12:
                ts['Season'][i]=3
    for i in range(0,tr.shape[0]):
            if 4<=tr['month'][i]<=6:
                tr['Season'][i]=1
            if 7<=tr['month'][i]<=9:
                tr['Season'][i]=2
            if 10<=tr['month'][i]<=12:
                tr['Season'][i]=3
                
    #Holidays
    tr['Holidays']=0
    ts['Holidays']=0
    
    for i in range(0,tr.shape[0]):
            #confederate Heroes
            if tr['month'][i]==1 and tr['day'][i]==19:
                 tr['Holidays'][i]=1
            #Texas Independance
            if tr['month'][i]==3 and tr['day'][i]==2:
                 tr['Holidays'][i]=2
            #San Jacinto Day        
            if tr['month'][i]==4 and tr['day'][i]==21:
                 tr['Holidays'][i]=3                       
            #Mothers Day        
            if tr['month'][i]==5 and tr['day'][i]==10:
                 tr['Holidays'][i]=4                       
            #Emancipation Day        
            if tr['month'][i]==6 and tr['day'][i]==19:
                 tr['Holidays'][i]=5
            #Fathers Day        
            if tr['month'][i]==6 and 18<=tr['day'][i]<=22:
                 tr['Holidays'][i]=6
            #Lyndon day       
            if tr['month'][i]==8 and tr['day'][i]==27:
                 tr['Holidays'][i]=7                       
            #Veterans day       
            if tr['month'][i]==11 and tr['day'][i]==11:
                 tr['Holidays'][i]=8 
            #ThanksGiving                    
            if tr['month'][i]==11 and 25<=tr['day'][i]<=27:
                 tr['Holidays'][i]=9
            #Christmas                    
            if tr['month'][i]==12 and 23<=tr['day'][i]<=27:
                 tr['Holidays'][i]=10                        
                    
                    
    for i in range(0,ts.shape[0]):
            #confederate Heroes
            if ts['month'][i]==1 and ts['day'][i]==19:
                 ts['Holidays'][i]=1
            #Texas Independance
            if ts['month'][i]==3 and ts['day'][i]==2:
                 ts['Holidays'][i]=2
            #San Jacinto Day        
            if ts['month'][i]==4 and ts['day'][i]==21:
                 ts['Holidays'][i]=3                       
            #Mothers Day        
            if ts['month'][i]==5 and ts['day'][i]==10:
                 ts['Holidays'][i]=4                       
            #Emancipation Day        
            if ts['month'][i]==6 and ts['day'][i]==19:
                 ts['Holidays'][i]=5
            #Fathers Day        
            if ts['month'][i]==6 and 18<=ts['day'][i]<=22:
                 ts['Holidays'][i]=6
            #Lyndon day       
            if ts['month'][i]==8 and ts['day'][i]==27:
                 ts['Holidays'][i]=7                       
            #Veterans day       
            if ts['month'][i]==11 and ts['day'][i]==11:
                 ts['Holidays'][i]=8 
            #ThanksGiving                    
            if ts['month'][i]==11 and 25<=ts['day'][i]<=27:
                 ts['Holidays'][i]=9
            #Christmas                    
            if ts['month'][i]==12 and 23<=ts['day'][i]<=27:
                 ts['Holidays'][i]=10   

    global left1
    global left2
    global left3
    global left4
    global left5
    global left6
    global left7
    global left8
    global left9
    global left10
    global left11
    global left12
    global left13
    global left14
    global left15
    global left16

    left1=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Aggressive'].sum())
    left2=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['At Vet'].sum())
    left3=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Barn'].sum())
    left4=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Behavior'].sum())
    left5=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Court/Investigation'].sum())
    left6=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Enroute'].sum())
    left7=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Foster'].sum())
    left8=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['In Foster'].sum())
    left9=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['In Kennel'].sum())
    left10=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['In Surgery'].sum())
    left11=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Medical'].sum())
    left12=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Offsite'].sum())
    left13=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Partner'].sum())
    left14=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Rabies Risk'].sum())
    left15=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['SCRP'].sum())
    left16=pd.DataFrame(tr.groupby(['AgeuponOutcome_num'])['Suffering'].sum())   
    
    global a1
    global a2
    global a3
    global a4
    global a5
    global a6
    global a7
    global a8
    global a9
    global a10
    global a11
    global a12
    global a13
    global a14
    global a15
    global a16

    a1=pd.DataFrame(tr.groupby(['Breed11'])['Aggressive'].sum())
    a2=pd.DataFrame(tr.groupby(['Breed11'])['At Vet'].sum())
    a3=pd.DataFrame(tr.groupby(['Breed11'])['Barn'].sum())
    a4=pd.DataFrame(tr.groupby(['Breed11'])['Behavior'].sum())
    a5=pd.DataFrame(tr.groupby(['Breed11'])['Court/Investigation'].sum())
    a6=pd.DataFrame(tr.groupby(['Breed11'])['Enroute'].sum())
    a7=pd.DataFrame(tr.groupby(['Breed11'])['Foster'].sum())
    a8=pd.DataFrame(tr.groupby(['Breed11'])['In Foster'].sum())
    a9=pd.DataFrame(tr.groupby(['Breed11'])['In Kennel'].sum())
    a10=pd.DataFrame(tr.groupby(['Breed11'])['In Surgery'].sum())
    a11=pd.DataFrame(tr.groupby(['Breed11'])['Medical'].sum())
    a12=pd.DataFrame(tr.groupby(['Breed11'])['Offsite'].sum())
    a13=pd.DataFrame(tr.groupby(['Breed11'])['Partner'].sum())
    a14=pd.DataFrame(tr.groupby(['Breed11'])['Rabies Risk'].sum())
    a15=pd.DataFrame(tr.groupby(['Breed11'])['SCRP'].sum())
    a16=pd.DataFrame(tr.groupby(['Breed11'])['Suffering'].sum())
                
    return tr.shape, ts.shape

clean(train_,test_)

Target=train_['Target']
train_.drop('OutcomeSubtype',axis=1,inplace=True)
train_.drop('Aggressive',axis=1,inplace=True)
train_.drop('At Vet',axis=1,inplace=True)
train_.drop('Barn',axis=1,inplace=True)
train_.drop('Behavior',axis=1,inplace=True)
train_.drop('Court/Investigation',axis=1,inplace=True)
train_.drop('Enroute',axis=1,inplace=True)
train_.drop('Foster',axis=1,inplace=True)
train_.drop('In Foster',axis=1,inplace=True)
train_.drop('In Kennel',axis=1,inplace=True)
train_.drop('In Surgery',axis=1,inplace=True)
train_.drop('Medical',axis=1,inplace=True)
train_.drop('Offsite',axis=1,inplace=True)
train_.drop('Partner',axis=1,inplace=True)
train_.drop('Rabies Risk',axis=1,inplace=True)
train_.drop('SCRP',axis=1,inplace=True)
train_.drop('Suffering',axis=1,inplace=True)
train_.drop('OutcomeType',axis=1,inplace=True)
train_.drop('Target',axis=1,inplace=True)
#Clean Data V2
def clean_v2(train):
	l1=['American Hairless Terrier','American Leopard Hound','Appenzeller Sennenhunde','Azawakh','Barbet','Basset Fauve de Bretagne','Belgian Laekenois','Biewer Terrier','Bolognese','Bracco Italiano','Braque du Bourbonnais','Broholmer','Catahoula Leopard Dog','Caucasian Ovcharka','Central Asian Shepherd Dog','Czechoslovakian Vlcak','Danish-Swedish Farmdog','Deutscher Wachtelhund','Dogo Argentino','Drentsche Patrijshond','Drever','Dutch Shepherd','Estrela Mountain Dog','Eurasier','French Spaniel','German Longhaired Pointer','German Spitz','Grand Basset Griffon Vendeen','Hamiltonstovare','Hovawart','Jagdterrier','Jindo','Kai Ken','Karelian Bear Dog','Kishu Ken','Kromfohrlander','Lancashire Heeler','Mudi','Nederlandse Kooikerhondje','Norrbottenspets','Perro de Presa Canario','Peruvian Inca Orchid','Portuguese Podengo','Portuguese Pointer','Portuguese Sheepdog','Pumi','Pyrenean Mastiff','Rafeiro do Alentejo','Russian Toy','Russian Tsvetnaya Bolonka','Schapendoes','Shikoku','Slovensky Cuvac','Slovensky Kopov','Sloughi','Small Munsterlander Pointer','Spanish Mastiff','Stabyhoun','Swedish Lapphund','Thai Ridgeback','Tornjak','Tosa','Transylvanian Hound','Treeing Tennessee Brindle','Working Kelpie','Australian Cattle Dog','Australian Shepherd','Bearded Collie','Beauceron','Belgian Malinois','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Berger Picard','Border Collie','Bouvier des Flandres','Briard','Canaan Dog','Cardigan Welsh Corgi','Collie','Entlebucher Mountain Dog','Finnish Lapphund','German Shepherd Dog','Icelandic Sheepdog','Miniature American Shepherd','Norwegian Buhund','Old English Sheepdog','Pembroke Welsh Corgi','Polish Lowland Sheepdog','Puli','Pyrenean Shepherd','Shetland Sheepdog','Spanish Water Dog','Swedish Vallhund','Afghan Hound','American English Coonhound','American Foxhound','Basenji','Basset Hound','Beagle','Black and Tan Coonhound','Bloodhound','Bluetick Coonhound','Borzoi','Cirneco de l etna','Dachshund','English Foxhound','Greyhound','Harrier','Ibizan Hound','Irish Wolfhound','Norwegian Elkhound','Otterhound','Petit Basset Griffon Vendeen','Pharaoh Hound','Plott','Portuguese Podengo Pequeno','Redbone Coonhound','Rhodesian Ridgeback','Saluki','Scottish Deerhound','Treeing Walker Coonhound','Whippet','American Eskimo Dog','Bichon Frise','Boston Terrier','Bulldog','Chinese Shar-Pei','Chow Chow','Coton de Tulear','Dalmatian','Finnish Spitz','French Bulldog','Keeshond','Lhasa Apso','Lowchen','Norwegian Lundehund','Poodle','Schipperke','Shiba Inu','Tibetan Spaniel','Tibetan Terrier','Xoloitzcuintli','American Water Spaniel','Boykin Spaniel','Brittany','Chesapeake Bay Retriever','Clumber Spaniel','Cocker Spaniel','Curly-Coated Retriever','English Cocker Spaniel','English Setter','English Springer Spaniel','Field Spaniel','Flat-Coated Retriever','German Shorthaired Pointer','German Wirehaired Pointer','Golden Retriever','Gordon Setter','Irish Red and White Setter','Irish Setter','Irish Water Spaniel','Labrador Retriever','Lagotto Romagnolo','Nova Scotia Duck Tolling Retriever','Pointer','Spinone Italiano','Sussex Spaniel','Vizsla','Weimaraner','Welsh Springer Spaniel','Wirehaired Pointing Griffon','Wirehaired Vizsla','Airedale Terrier','American Staffordshire Terrier','Australian Terrier','Bedlington Terrier','Border Terrier','Bull Terrier','Cairn Terrier','Cesky Terrier','Dandie Dinmont Terrier','Glen of Imaal Terrier','Irish Terrier','Kerry Blue Terrier','Lakeland Terrier','Manchester Terrier','Miniature Bull Terrier','Miniature Schnauzer','Norfolk Terrier','Norwich Terrier','Parson Russell Terrier','Rat Terrier','Russell Terrier','Scottish Terrier','Sealyham Terrier','Skye Terrier','Smooth Fox Terrier','Soft Coated Wheaten Terrier','Staffordshire Bull Terrier','Welsh Terrier','West Highland White Terrier','Wire Fox Terrier','Affenpinscher','Brussels Griffon','Cavalier King Charles Spaniel','Chihuahua','Chinese Crested','English Toy Spaniel','Havanese','Italian Greyhound','Japanese Chin','Maltese','Miniature Pinscher','Papillon','Pekingese','Pomeranian','Pug','Shih Tzu','Silky Terrier','Toy Fox Terrier','Yorkshire Terrier','Akita','Alaskan Malamute','Anatolian Shepherd Dog','Bernese Mountain Dog','Black Russian Terrier','Boerboel','Boxer','Bullmastiff','Cane Corso','Chinook','Doberman Pinscher','Dogue de Bordeaux','German Pinscher','Giant Schnauzer','Great Dane','Great Pyrenees','Greater Swiss Mountain Dog','Komondor','Kuvasz','Leonberger','Mastiff','Neapolitan Mastiff','Newfoundland','Portuguese Water Dog','Rottweiler','Samoyed','Siberian Husky','Standard Schnauzer','Tibetan Mastiff','St. Bernard']
	l2=['FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','FSS','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Herding','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Hound','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Non_sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Sporting','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Terrier','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Toy','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working','Working']
	
	l3=pd.DataFrame(l1)
	l3['Groups']=pd.DataFrame(l1)
	l3['Breed11']=pd.DataFrame(l2)
	l3.drop(0,axis=1,inplace=True)
	l3['Breed11']=l3['Breed11'].str.upper()
	
	#2013
	l4=['LABRADOR RETRIEVER','GERMAN SHEPHERD','GOLDEN RETRIEVER','AMERICAN BULLDOG','BEAGLE','FRENCH BULLDOG','YORKSHIRE TERRIER','MINIATURE POODLE','ROTTWEILER','BOXER','German Shorthair Pointer','SIBERIAN HUSKY','DACHSHUND','Pinscher','GREAT DANES','MINIATURE SCHNAUZER','AUSTRALIAN SHEPHERD','CAVALIER KING CHARLES SPANIEL','SHIH TZU','PEMBROKE WELSH CORGI','POMERANIAN','BOSTON TERRIER','SHETLAND SHEEPDOG','HAVANESE','MASTIFF','BRITTANY','ENGLISH SPRINGER SPANIEL','CHIHUAHUA','BERNESE MOUNTAIN DOG','COCKER SPANIEL','MALTESE','VIZSLA','PUG','WEIMARANER','CANE CORSO','COLLIE','NEWFOUNDLAND','BORDER COLLIE','BASSET HOUND','RHODESIAN RIDGEBACK','WEST HIGHLAND WHITE TERRIER','CHESAPEAKE BAY RETRIEVER','BULLMASTIFF','BICHON FRISE','SHIBA INU','AKITA','SOFT COATED WHEATEN TERRIER','PAPILLON','BLOODHOUND','St. Bernard Smooth Coat','St. Bernard Rough Coat','BELGIAN MALINOIS','PORTUGUESE WATER DOG','AIREDALE TERRIER','ALASKAN MALAMUTE','BULL TERRIER','AUSTRALIAN CATTLE DOG','WHIPPET','SCOTTISH TERRIER','CHINESE SHARPEI','ENGLISH COCKER SPANIEL','SAMOYED','DALMATIAN','DOGUE DE BORDEAUX','MINIATURE PINSCHER','LHASA APSO','WIREHAIRED POINTING GRIFFON','GREAT PYRENEES','GERMAN WIREHAIRED POINTER','IRISH WOLFHOUND','CAIRN TERRIER','ITALIAN GREYHOUND','IRISH SETTER','CHOW CHOW','OLD ENGLISH SHEEPDOG','CHINESE CRESTED','CARDIGAN WELSH CORGI','AMERICAN STAFFORDSHIRE TERRIER','GREATER SWISS MOUNTAIN DOG','STAFFORDSHIRE BULL TERRIER','PEKINGESE','GIANT SCHNAUZER','BORDER TERRIER','BOUVIER DES FLANDRES','KEESHONDEN','COTON DE TULEAR','FLAT COAT RETRIEVER','BASENJI','NORWEGIAN ELKHOUND','BORZOI','TIBETAN TERRIER','STANDARD SCHNAUZER','ANATOLIAN SHEPHERD DOG','LEONBERGER','WIRE FOX TERRIER','BRUSSELS GRIFFON','ENGLISH SETTER','JAPANESE CHIN','BELGIAN TERVUREN','NOVA SCOTIA DUCK TOLLING RETRIEVER','AFGHAN HOUND','RAT TERRIER','SILKY TERRIER','NORWICH TERRIER','RUSSELL TERRIER','GORDON SETTER','NEAPOLITAN MASTIFF','BOYKIN SPANIEL','WELSH TERRIER','SCHIPPERKE','TOY FOX TERRIER','PARSON RUSSELL TERRIER','SPINONI ITALIANI','IRISH TERRIER','POINTER','TIBETAN SPANIEL','BLACK RUSSIAN TERRIER','TREEING WALKER COONHOUND','AMERICAN ESKIMO DOG','BEARDED COLLIE','BELGIAN SHEEPDOG','MINIATURE BULL TERRIER','SMOOTH FOX TERRIER','BLUETICK COONHOUND','KERRY BLUE TERRIER','AUSTRALIAN TERRIER','BOERBOEL','BLACK AND TAN COONHOUND','WELSH SPRINGER SPANIEL','ENGLISH TOY SPANIEL','BRIARD','NORFOLK TERRIER','SALUKI','TIBETAN MASTIFF','CLUMBER SPANIEL','XOLOITZCUINTLI','AFFENPINSCHER','MANCHESTER TERRIER','GERMAN PINSCHER','REDBONE COONHOUND','ICELANDIC SHEEPDO','LAKELAND TERRIER','BEAUCERON','PETIT BASSET GRIFFON VENDEEN','IRISH WATER SPANIEL','FIELD SPANIEL','BEDLINGTON TERRIER','GREYHOUND','IRISH RED AND WHITE SETTER','PLOTT','KUVASZOK','CURLY COAT RETRIEVER','SCOTTISH DEERHOUND','PORTUGUESE PODENGO PEQUENO','PULIK','SWEDISH VALLHUND','WIREHAIRED VIZSLA','AMERICAN WATER SPANIEL','SEALYHAM TERRIER','ENTLEBUCHER MOUNTAIN DOG','IBIZAN HOUND','LOWCHEN','CIRNECHI DE L ETNA','KOMONDOROK','POLISH LOWLAND SHEEPDOG','NORWEGIAN BUHUND','AMERICAN ENGLISH COONHOUND','SPANISH WATER DOG','GLEN OF IMAAL TERRIER','FINNISH LAPPHUND','CANAAN DOG','PHARAOH HOUND','DANDIE DINMONT TERRIER','SUSSEX SPANIEL','BERGAMASCO','SKYE TERRIER','PYRENEAN SHEPHERD','CHINOOK','FINNISH SPITZ','CESKY TERRIER','OTTERHOUND','AMERICAN FOXHOUND','NORWEGIAN LUNDEHUND','HARRIER','ENGLISH FOXHOUND']
	l5=['1','2','3','5','4','11','6','8','9','7','13','14','10','12','16','17','20','18','15','24','19','23','21','25','26','30','28','22','32','29','27','34','31','33','50','35','37','44','41','39','36','43','40','40','46','45','51','38','48','47','47','60','49','56','57','52','58','59','55','54','62','67','64','65','53','63','80','69','71','73','61','66','72','70','78','68','75','76','74','79','77','83','81','82','86','-9999','94','85','103','99','88','90','93','98','96','84','91','87','108','97','95','-9999','92','89','102','105','111','121','104','109','107','100','117','123','114','106','118','101','110','112','119','125','116','128','126','122','-9999','113','124','135','127','129','115','132','131','139','143','120','130','133','142','134','152','138','141','140','137','148','145','149','150','163','165','153','136','147','-9999','144','158','155','151','154','-9999','159','157','166','146','-9999','167','171','164','160','168','162','-9999','161','169','156','170','174','172','176','175','173','177']
	
	l6=pd.DataFrame(l4)
	l6['Breed11']=pd.DataFrame(l4)
	l6['Classement_2013']=pd.DataFrame(l5)
	l6.drop(0,axis=1,inplace=True)
	l6['Breed11']=l6['Breed11'].str.upper()
	
	#2014
	l7=['LABRADOR RETRIEVER','GERMAN SHEPHERD','GOLDEN RETRIEVER','AMERICAN BULLDOG','BEAGLE','FRENCH BULLDOG','YORKSHIRE TERRIER','MINIATURE POODLE','ROTTWEILER','BOXER','German Shorthair Pointer','SIBERIAN HUSKY','DACHSHUND','Pinscher','GREAT DANES','MINIATURE SCHNAUZER','AUSTRALIAN SHEPHERD','CAVALIER KING CHARLES SPANIEL','SHIH TZU','PEMBROKE WELSH CORGI','POMERANIAN','BOSTON TERRIER','SHETLAND SHEEPDOG','HAVANESE','MASTIFF','BRITTANY','ENGLISH SPRINGER SPANIEL','CHIHUAHUA','BERNESE MOUNTAIN DOG','COCKER SPANIEL','MALTESE','VIZSLA','PUG','WEIMARANER','CANE CORSO','COLLIE','NEWFOUNDLAND','BORDER COLLIE','BASSET HOUND','RHODESIAN RIDGEBACK','WEST HIGHLAND WHITE TERRIER','CHESAPEAKE BAY RETRIEVER','BULLMASTIFF','BICHON FRISE','SHIBA INU','AKITA','SOFT COATED WHEATEN TERRIER','PAPILLON','BLOODHOUND','St. Bernard Smooth Coat','St. Bernard Rough Coat','BELGIAN MALINOIS','PORTUGUESE WATER DOG','AIREDALE TERRIER','ALASKAN MALAMUTE','BULL TERRIER','AUSTRALIAN CATTLE DOG','WHIPPET','SCOTTISH TERRIER','CHINESE SHARPEI','ENGLISH COCKER SPANIEL','SAMOYED','DALMATIAN','DOGUE DE BORDEAUX','MINIATURE PINSCHER','LHASA APSO','WIREHAIRED POINTING GRIFFON','GREAT PYRENEES','GERMAN WIREHAIRED POINTER','IRISH WOLFHOUND','CAIRN TERRIER','ITALIAN GREYHOUND','IRISH SETTER','CHOW CHOW','OLD ENGLISH SHEEPDOG','CHINESE CRESTED','CARDIGAN WELSH CORGI','AMERICAN STAFFORDSHIRE TERRIER','GREATER SWISS MOUNTAIN DOG','STAFFORDSHIRE BULL TERRIER','PEKINGESE','GIANT SCHNAUZER','BORDER TERRIER','BOUVIER DES FLANDRES','KEESHONDEN','COTON DE TULEAR','FLAT COAT RETRIEVER','BASENJI','NORWEGIAN ELKHOUND','BORZOI','TIBETAN TERRIER','STANDARD SCHNAUZER','ANATOLIAN SHEPHERD DOG','LEONBERGER','WIRE FOX TERRIER','BRUSSELS GRIFFON','ENGLISH SETTER','JAPANESE CHIN','BELGIAN TERVUREN','NOVA SCOTIA DUCK TOLLING RETRIEVER','AFGHAN HOUND','RAT TERRIER','SILKY TERRIER','NORWICH TERRIER','RUSSELL TERRIER','GORDON SETTER','NEAPOLITAN MASTIFF','BOYKIN SPANIEL','WELSH TERRIER','SCHIPPERKE','TOY FOX TERRIER','PARSON RUSSELL TERRIER','SPINONI ITALIANI','IRISH TERRIER','POINTER','TIBETAN SPANIEL','BLACK RUSSIAN TERRIER','TREEING WALKER COONHOUND','AMERICAN ESKIMO DOG','BEARDED COLLIE','BELGIAN SHEEPDOG','MINIATURE BULL TERRIER','SMOOTH FOX TERRIER','BLUETICK COONHOUND','KERRY BLUE TERRIER','AUSTRALIAN TERRIER','BOERBOEL','BLACK AND TAN COONHOUND','WELSH SPRINGER SPANIEL','ENGLISH TOY SPANIEL','BRIARD','NORFOLK TERRIER','SALUKI','TIBETAN MASTIFF','CLUMBER SPANIEL','XOLOITZCUINTLI','AFFENPINSCHER','MANCHESTER TERRIER','GERMAN PINSCHER','REDBONE COONHOUND','ICELANDIC SHEEPDO','LAKELAND TERRIER','BEAUCERON','PETIT BASSET GRIFFON VENDEEN','IRISH WATER SPANIEL','FIELD SPANIEL','BEDLINGTON TERRIER','GREYHOUND','IRISH RED AND WHITE SETTER','PLOTT','KUVASZOK','CURLY COAT RETRIEVER','SCOTTISH DEERHOUND','PORTUGUESE PODENGO PEQUENO','PULIK','SWEDISH VALLHUND','WIREHAIRED VIZSLA','AMERICAN WATER SPANIEL','SEALYHAM TERRIER','ENTLEBUCHER MOUNTAIN DOG','IBIZAN HOUND','LOWCHEN','CIRNECHI DE L ETNA','KOMONDOROK','POLISH LOWLAND SHEEPDOG','NORWEGIAN BUHUND','AMERICAN ENGLISH COONHOUND','SPANISH WATER DOG','GLEN OF IMAAL TERRIER','FINNISH LAPPHUND','CANAAN DOG','PHARAOH HOUND','DANDIE DINMONT TERRIER','SUSSEX SPANIEL','BERGAMASCO','SKYE TERRIER','PYRENEAN SHEPHERD','CHINOOK','FINNISH SPITZ','CESKY TERRIER','OTTERHOUND','AMERICAN FOXHOUND','NORWEGIAN LUNDEHUND','HARRIER','ENGLISH FOXHOUND']
	l8=['1','2','3','4','5','9','6','7','10','8','12','13','11','14','15','16','18','19','17','22','20','23','21','25','26','27','28','24','32','30','29','34','33','35','48','36','37','40','42','39','38','41','45','44','47','46','49','43','50','51','51','60','52','57','54','53','55','56','59','58','62','68','66','63','61','67','76','75','71','72','69','74','73','70','77','65','78','84','80','79','82','83','85','81','87','31','92','86','103','102','88','90','94','104','95','91','89','93','110','99','98','111','101','97','105','100','114','108','106','109','115','116','118','125','119','122','127','112','120','121','123','129','124','130','128','139','64','126','133','138','132','131','134','135','143','142','144','136','141','148','146','149','145','150','152','137','140','147','155','154','161','163','157','166','151','158','107','160','169','153','159','165','117','168','170','172','156','113','162','173','176','164','167','175','96','177','178','171','174','182','179','180','184','181','183']
	
	l9=pd.DataFrame(l7)
	l9['Breed11']=pd.DataFrame(l7)
	l9['Classement_2014']=pd.DataFrame(l8)
	l9.drop(0,axis=1,inplace=True)
	l9['Breed11']=l9['Breed11'].str.upper()
	
	#2015
	l10=['LABRADOR RETRIEVER','GERMAN SHEPHERD','GOLDEN RETRIEVER','AMERICAN BULLDOG','BEAGLE','FRENCH BULLDOG','YORKSHIRE TERRIER','MINIATURE POODLE','ROTTWEILER','BOXER','German Shorthair Pointer','SIBERIAN HUSKY','DACHSHUND','Pinscher','GREAT DANES','MINIATURE SCHNAUZER','AUSTRALIAN SHEPHERD','CAVALIER KING CHARLES SPANIEL','SHIH TZU','PEMBROKE WELSH CORGI','POMERANIAN','BOSTON TERRIER','SHETLAND SHEEPDOG','HAVANESE','MASTIFF','BRITTANY','ENGLISH SPRINGER SPANIEL','CHIHUAHUA','BERNESE MOUNTAIN DOG','COCKER SPANIEL','MALTESE','VIZSLA','PUG','WEIMARANER','CANE CORSO','COLLIE','NEWFOUNDLAND','BORDER COLLIE','BASSET HOUND','RHODESIAN RIDGEBACK','WEST HIGHLAND WHITE TERRIER','CHESAPEAKE BAY RETRIEVER','BULLMASTIFF','BICHON FRISE','SHIBA INU','AKITA','SOFT COATED WHEATEN TERRIER','PAPILLON','BLOODHOUND','St. Bernard Smooth Coat','St. Bernard Rough Coat','BELGIAN MALINOIS','PORTUGUESE WATER DOG','AIREDALE TERRIER','ALASKAN MALAMUTE','BULL TERRIER','AUSTRALIAN CATTLE DOG','WHIPPET','SCOTTISH TERRIER','CHINESE SHARPEI','ENGLISH COCKER SPANIEL','SAMOYED','DALMATIAN','DOGUE DE BORDEAUX','MINIATURE PINSCHER','LHASA APSO','WIREHAIRED POINTING GRIFFON','GREAT PYRENEES','GERMAN WIREHAIRED POINTER','IRISH WOLFHOUND','CAIRN TERRIER','ITALIAN GREYHOUND','IRISH SETTER','CHOW CHOW','OLD ENGLISH SHEEPDOG','CHINESE CRESTED','CARDIGAN WELSH CORGI','AMERICAN STAFFORDSHIRE TERRIER','GREATER SWISS MOUNTAIN DOG','STAFFORDSHIRE BULL TERRIER','PEKINGESE','GIANT SCHNAUZER','BORDER TERRIER','BOUVIER DES FLANDRES','KEESHONDEN','COTON DE TULEAR','FLAT COAT RETRIEVER','BASENJI','NORWEGIAN ELKHOUND','BORZOI','TIBETAN TERRIER','STANDARD SCHNAUZER','ANATOLIAN SHEPHERD DOG','LEONBERGER','WIRE FOX TERRIER','BRUSSELS GRIFFON','ENGLISH SETTER','JAPANESE CHIN','BELGIAN TERVUREN','NOVA SCOTIA DUCK TOLLING RETRIEVER','AFGHAN HOUND','RAT TERRIER','SILKY TERRIER','NORWICH TERRIER','RUSSELL TERRIER','GORDON SETTER','NEAPOLITAN MASTIFF','BOYKIN SPANIEL','WELSH TERRIER','SCHIPPERKE','TOY FOX TERRIER','PARSON RUSSELL TERRIER','SPINONI ITALIANI','IRISH TERRIER','POINTER','TIBETAN SPANIEL','BLACK RUSSIAN TERRIER','TREEING WALKER COONHOUND','AMERICAN ESKIMO DOG','BEARDED COLLIE','BELGIAN SHEEPDOG','MINIATURE BULL TERRIER','SMOOTH FOX TERRIER','BLUETICK COONHOUND','KERRY BLUE TERRIER','AUSTRALIAN TERRIER','BOERBOEL','BLACK AND TAN COONHOUND','WELSH SPRINGER SPANIEL','ENGLISH TOY SPANIEL','BRIARD','NORFOLK TERRIER','SALUKI','TIBETAN MASTIFF','CLUMBER SPANIEL','XOLOITZCUINTLI','AFFENPINSCHER','MANCHESTER TERRIER','GERMAN PINSCHER','REDBONE COONHOUND','ICELANDIC SHEEPDO','LAKELAND TERRIER','BEAUCERON','PETIT BASSET GRIFFON VENDEEN','IRISH WATER SPANIEL','FIELD SPANIEL','BEDLINGTON TERRIER','GREYHOUND','IRISH RED AND WHITE SETTER','PLOTT','KUVASZOK','CURLY COAT RETRIEVER','SCOTTISH DEERHOUND','PORTUGUESE PODENGO PEQUENO','PULIK','SWEDISH VALLHUND','WIREHAIRED VIZSLA','AMERICAN WATER SPANIEL','SEALYHAM TERRIER','ENTLEBUCHER MOUNTAIN DOG','IBIZAN HOUND','LOWCHEN','CIRNECHI DE L ETNA','KOMONDOROK','POLISH LOWLAND SHEEPDOG','NORWEGIAN BUHUND','AMERICAN ENGLISH COONHOUND','SPANISH WATER DOG','GLEN OF IMAAL TERRIER','FINNISH LAPPHUND','CANAAN DOG','PHARAOH HOUND','DANDIE DINMONT TERRIER','SUSSEX SPANIEL','BERGAMASCO','SKYE TERRIER','PYRENEAN SHEPHERD','CHINOOK','FINNISH SPITZ','CESKY TERRIER','OTTERHOUND','AMERICAN FOXHOUND','NORWEGIAN LUNDEHUND','HARRIER','ENGLISH FOXHOUND']
	l11=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184']
	
	l12=pd.DataFrame(l10)
	l12['Breed11']=pd.DataFrame(l10)
	l12['Classement_2015']=pd.DataFrame(l11)
	l12.drop(0,axis=1,inplace=True)
	l12['Breed11']=l12['Breed11'].str.upper()
	
	#join with train
	train=pd.merge(train, l3, on = 'Breed11',how='left')
	train=pd.merge(train, l6, on = 'Breed11',how='left')
	train=pd.merge(train, l9, on = 'Breed11',how='left')
	train=pd.merge(train, l12, on = 'Breed11',how='left')
	
	train['Classement_2013']=train['Classement_2013'].fillna(-9999).astype(float)
	train['Classement_2014']=train['Classement_2014'].fillna(-9999).astype(float)
	train['Classement_2015']=train['Classement_2015'].fillna(-9999).astype(float)
	
	#Type2
	l111=['Great Dane','Mastiff','Neapolitan Mastiff','Bullmastiff','Newfoundland','Dogue de Bordeaux','Cane Corso','Great Pyrenees','Bernese Mountain Dog','Tibetan Mastiff','Black Russian Terrier','Leonberger','Irish Wolfhound','Scottish Deerhound','Brussels Griffon','Cavalier King Charles Spaniel','Chihuahua','Chinese Crested','Dachshund','English Toy Spaniel','Havanese','Italian Greyhound','Japanese Chin','Maltese','Miniature Pinscher','Norfolk Terrier','Norwich Terrier','Papillon','Pekingese','Pomeranian','Pug','Schipperke','Shih Tzu','Silky Terrier','Toy Fox Terrier','Manchester Terrier','Poodle','Yorkshire Terrier','Brittany','Bulldog','Cocker Spaniel','English Springer Spaniel','Pembroke Welsh Corgi','Shetland Sheepdog','Whippet']
	l222=['Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Medium','Medium','Medium','Medium','Medium','Medium','Medium']
	
	l13=pd.DataFrame(l111)
	l13['Breed11']=pd.DataFrame(l111)
	l13['Type2']=pd.DataFrame(l222)
	l13.drop(0,axis=1,inplace=True)
	l13['Breed11']=l13['Breed11'].str.upper()
	train=pd.merge(train, l13, on = 'Breed11',how='left')
	train['Type2']=train['Type2'].fillna('Autres')
	
	####Appt
	l_1=['Bichon Frise','Bulldog','Cavalier King Charles Spaniel','Chinese Crested','French Bulldog','Greyhound','Havanese','Maltese','Pug','Shih Tzu']
	l_2=['Appartments','Appartments','Appartments','Appartments','Appartments','Appartments','Appartments','Appartments','Appartments','Appartments']
	
	l13_1=pd.DataFrame(l_1)
	l13_1['Breed11']=pd.DataFrame(l_1)
	l13_1['App']=pd.DataFrame(l_2)
	l13_1.drop(0,axis=1,inplace=True)
	l13_1['Breed11']=l13_1['Breed11'].str.upper()
	train=pd.merge(train, l13_1, on = 'Breed11',how='left')
	train['App']=train['App'].fillna('Autres')
	
	####Family
	l_3=['Beagle','Brussels Griffon','Bulldog','Collie','French Bulldog','Golden Retriever','Irish Setter','Labrador Retriever','Newfoundland','Pug']
	l_4=['Family','Family','Family','Family','Family','Family','Family','Family','Family','Family']
	
	l13_2=pd.DataFrame(l_3)
	l13_2['Breed11']=pd.DataFrame(l_3)
	l13_2['Family']=pd.DataFrame(l_4)
	l13_2.drop(0,axis=1,inplace=True)
	l13_2['Breed11']=l13_2['Breed11'].str.upper()
	train=pd.merge(train, l13_2, on = 'Breed11',how='left')
	train['Family']=train['Family'].fillna('Autres')
	
	####Hairless
	l_5=['American Hairless Terrier','Chinese Crested','Peruvian Inca Orchid','Xoloitzcuintli']
	l_6=['Hairless','Hairless','Hairless','Hairless']
	
	l13_3=pd.DataFrame(l_5)
	l13_3['Breed11']=pd.DataFrame(l_5)
	l13_3['Hairless']=pd.DataFrame(l_6)
	l13_3.drop(0,axis=1,inplace=True)
	l13_3['Breed11']=l13_3['Breed11'].str.upper()
	train=pd.merge(train, l13_3, on = 'Breed11',how='left')
	train['Hairless']=train['Hairless'].fillna('Autres')

	####Hypoa
	l_7=['Afghan Hound','American Hairless Terrier','Bedlington Terrier','Bichon Frise','Chinese Crested','Coton de Tulear','Giant Schnauzer','Irish Water Spaniel','Kerry Blue Terrier','Lagotto Romagnolo','Maltese','Peruvian Inca Orchid','Poodle','Portuguese Water Dog','Soft Coated Wheaten Terrier','Standard Schnauzer','Xoloitzcuintli']
	l_8=['Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic','Hypoallergenic']
	
	l13_4=pd.DataFrame(l_7)
	l13_4['Breed11']=pd.DataFrame(l_7)
	l13_4['Hypoa']=pd.DataFrame(l_8)
	l13_4.drop(0,axis=1,inplace=True)
	l13_4['Breed11']=l13_4['Breed11'].str.upper()
	train=pd.merge(train, l13_4, on = 'Breed11',how='left')
	train['Hypoa']=train['Hypoa'].fillna('Autres')
	
	####Kids
	l_9=['Beagle','Boxer','Bull Terrier','Bulldog','Golden Retriever','Labrador Retriever','Newfoundland','Soft Coated Wheaten Terrier','Weimaraner']
	l_10=['Kids','Kids','Kids','Kids','Kids','Kids','Kids','Kids','Kids']
	
	l13_5=pd.DataFrame(l_9)
	l13_5['Breed11']=pd.DataFrame(l_9)
	l13_5['Kids']=pd.DataFrame(l_10)
	l13_5.drop(0,axis=1,inplace=True)
	l13_5['Breed11']=l13_5['Breed11'].str.upper()
	train=pd.merge(train, l13_5, on = 'Breed11',how='left')
	train['Kids']=train['Kids'].fillna('Autres')
	
	####Smart
	l_11=['Bloodhound','Border Collie','Doberman Pinscher','German Shepherd Dog','Golden Retriever','Labrador Retriever','Papillon','Poodle','Rottweiler','Shetland Sheepdog']
	l_12=['Smart','Smart','Smart','Smart','Smart','Smart','Smart','Smart','Smart','Smart']
	
	l13_6=pd.DataFrame(l_11)
	l13_6['Breed11']=pd.DataFrame(l_11)
	l13_6['Smart']=pd.DataFrame(l_12)
	l13_6.drop(0,axis=1,inplace=True)
	l13_6['Breed11']=l13_6['Breed11'].str.upper()
	train=pd.merge(train, l13_6, on = 'Breed11',how='left')
	train['Smart']=train['Smart'].fillna('Autres')
	
	train=train.merge(right1, left_on='Breed', right_index=True,how='left').merge(right2, left_on='Breed',right_index=True,how='left').merge(right3,left_on='Breed', right_index=True,how='left').merge(right4,left_on='Breed',right_index=True,how='left').merge(right5,left_on='Breed', right_index=True,how='left').merge(right6, left_on='Breed',right_index=True,how='left').merge(right7,left_on='Breed', right_index=True,how='left').merge(right8, left_on='Breed',right_index=True,how='left').merge(right9,left_on='Breed', right_index=True,how='left').merge(right10, left_on='Breed',right_index=True,how='left').merge(right11,left_on='Breed', right_index=True,how='left').merge(right12, left_on='Breed',right_index=True,how='left').merge(right13,left_on='Breed', right_index=True,how='left').merge(right14, left_on='Breed',right_index=True,how='left').merge(right15,left_on='Breed', right_index=True,how='left').merge(right16, left_on='Breed',right_index=True,how='left')
	train=train.rename(columns={'Aggressive_y':'a1','At Vet_y':'a2','Barn_y':'a3','Behavior_y':'a4','Court/Investigation_y':'a5','Enroute_y':'a6','Foster_y':'a7','In Foster_y':'a8',
	            'In Kennel_y':'a9','In Surgery_y':'a10','Medical_y':'a11','Offsite_y':'a12','Partner_y':'a13','Rabies Risk_y':'a14','SCRP_y':'a15','Suffering_y':'a16'})
	
	train=train.merge(left1, left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left2, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left3,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left4,left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left5,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left6, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left7,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left8, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left9,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left10, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left11,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left12, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left13,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left14, left_on='AgeuponOutcome_num',right_index=True,how='left').merge(left15,left_on='AgeuponOutcome_num', right_index=True,how='left').merge(left16, left_on='AgeuponOutcome_num',right_index=True,how='left')
	
	train=train.merge(a1, left_on='Breed11', right_index=True,how='left').merge(a2, left_on='Breed11',right_index=True,how='left').merge(a3,left_on='Breed11', right_index=True,how='left').merge(a4,left_on='Breed11',right_index=True,how='left').merge(a5,left_on='Breed11', right_index=True,how='left').merge(a6, left_on='Breed11',right_index=True,how='left').merge(a7,left_on='Breed11', right_index=True,how='left').merge(a8, left_on='Breed11',right_index=True,how='left').merge(a9,left_on='Breed11', right_index=True,how='left').merge(a10, left_on='Breed11',right_index=True,how='left').merge(a11,left_on='Breed11', right_index=True,how='left').merge(a12, left_on='Breed11',right_index=True,how='left').merge(a13,left_on='Breed11', right_index=True,how='left').merge(a14, left_on='Breed11',right_index=True,how='left').merge(a15,left_on='Breed11', right_index=True,how='left').merge(a16, left_on='Breed11',right_index=True,how='left')
	
	train.drop(['Aggressive_x','At Vet_x','Barn_x','Behavior_x','Court/Investigation_x','Enroute_x','Foster_x','In Foster_x',
	            'In Kennel_x','In Surgery_x','Medical_x','Offsite_x','Partner_x','Rabies Risk_x','SCRP_x','Suffering_x'],axis=1,inplace=True)
	#Type4
	l15=['Akita','Alaskan Malamute','Anatolian Shepherd Dog','Argentine Dogo','Beauceron','Bernese Mountain Dog','Black Russian Terrier','Bloodhound','Borzoi','Bouvier des Flandres','Briard','Bullmastiff','Cane Corso','Caucasian Ovcharka','Central Asian Shepherd Dog','Curly-Coated Retriever','Doberman Pinsch','Greyhound','Otterhounds','Spinone Italiano','Afghan Hound','Airedale Terrier','American Foxhound','American Staffordshire Terrier','Appenzeller Sennenhunde','Belgian Sheepdog','Belgian Tervuren','Bergamasco','Black and Tan Coonhound','Bluetick Coonhound','Boxer','Bracco Italiano','Catahoula Leopard','Chesapeake Bay Retriever','Chinook','Clumber Spaniel','English Setter','Flat Coat Retriever','German Shorthaired Pointer','German Wirehaired Pointer','Golden Retrievers','Gordon Setters','Irish Red and White Setters','Irish Setters','Labrador Retriever','The Standard Poodle','Redbone Coonhound','Thai Ridgeback','Weimaraner','American English Coonhound','Australian Shepherd','Azawakh','Basset Hound','Bearded Collie','Belgian Laekenois','Belgian Malinois','Bull Terrier','Bulldog','Chinese Sharpei','Chow Chow','Collie','Dalmatian','English Springer Spaniel','Entlebucher Mountain Dog','Field Spaniel','Finnish Lapphund','Grand Basset Griffon Vendéen','Harrier','Ibizan Hounds','Irish Water Spaniels','Keeshond','Norwegian Elkhound','Nova Scotia Duck Tolling Retriever','The Pharaoh Hound','The Plott','The Pointer','Portuguese Podengo','The Saluki','Samoyed','Siberian Husky','Sloughis','Stabyhouns','Sussex Spaniel','Treeing Walker Coonhound','Vizslas','Wirehaired Pointing Griffon','Basenji','Beagle','Cardigan Welsh Corgi','Cocker Spaniel','Dandie Dinmont Terrier','English Cocker Spaniel','Finnish Spitz','German Pinscher','Kai Kens','Kerry Blue Terriers','Miniature Bull Terrier','Norwegian Buhunds','Pembroke Welsh Corgi','Petit Bassett Griffon Vendeen','Polish Lowland Sheepdog','Portuguese Podengo (Medio)','Portuguese Water Dog','Pyrenean Shepherd','Sealyham Terrier','Shetland Sheepdog','Skye Terrier','Soft Coated Wheaten Terrier','Spanish Water Dogs','Staffordshire Bull Terrier','Standard Schnauzer','Tibetan Terrier','Treeing Tennessee Brindle','Welsh Springer Spaniel','West Highland White Terrier','Whippet','Wire Fox Terrier','Affenpinscher','American Eskimo','Miniature American Eskimo','Toy American Eskimo','Australian Terrier','Bichon Frise','Bolognese','Border Terrier','Boston Terrier','Brussels Griffon','Cairn Terrier','Cavalier King Charles Spaniel','Cesky Terrier','Chihuahua','Chinese Crested','Coton de Tulear','Dachshund','English Toy Spaniel','French Bulldog','Havanese','Italian Greyhound','Japanese Chin','Lancashire Heeler','Lhasa Apso','Lowchen','Maltese','Manchester Terrier','Standard Manchester Terrier','Toy Manchester Terrier','Miniature Pinscher','Miniature Schnauzer','Norfolk Terrier','Norwich Terrier','Papillon','Parson Russell Terrier','Pekingese','Pomeranian','Poodle','Miniature Poodle','Toy Poodle','Portuguese Podengo (Pequeno)','Pug','Rat Terrier','Schipperke','Scottish Terrier','Shiba Inu','Shih Tzu','Silky Terrier','Smooth Fox Terrier','Tibetan Spaniel','Toy Fox Terrier','Yorkshire Terrier','Xoloitzcuintli']
	
	l16=['Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium Large','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small Medium','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small','Small']
		
	l17=pd.DataFrame(l15)
	l17['Breed11']=pd.DataFrame(l15)
	l17['Type4']=pd.DataFrame(l16)
	l17.drop(0,axis=1,inplace=True)
	l17['Breed11']=l17['Breed11'].str.upper()
	
	train=pd.merge(train, l17, on = 'Breed11',how='left')
	
	#dangerous dogs
	l18=['Dalmatian','Great Dane','Presa Canario','Doberman Pinsch','Chow Chow','Alaskan Malamute','Wolf-Dog Hybrid','German Shepherd','Rottweiler','Pit Bull']
	l19=['Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog','Dangerous Dog']
	
	l20=pd.DataFrame(l18)
	l20['Breed11']=pd.DataFrame(l18)
	l20['DG']=pd.DataFrame(l19)
	l20.drop(0,axis=1,inplace=True)
	l20['Breed11']=l20['Breed11'].str.upper()
	train=pd.merge(train, l20, on = 'Breed11',how='left')
	train['DG']=train['DG'].fillna('Autres')
	return train
	#'Breed11','Breed2','Color1'
#	train = pd.get_dummies(train, columns=['Groups','App','Family','Hairless','Hypoa','Kids','Smart','Type2','DG','Type4','dayofweek','Holidays','Season','am_pm','Unicolor','month','day',
#	                                       'Top_Mix','AgeuponOutcome_cat','sprayed','Type_sex','SexuponOutcome_cat','AnimalType_cat'])


train_2=clean_v2(train_)
train_2 = pd.get_dummies(train_2, columns=['Groups','App','Family','Hairless','Hypoa','Kids','Smart','Type2','DG','Type4','dayofweek','Holidays','Season','am_pm','Unicolor','month','day',
	                                       'Top_Mix','AgeuponOutcome_cat','sprayed','Type_sex','SexuponOutcome_cat','AnimalType_cat'])

test_2=clean_v2(test_)
test_2 = pd.get_dummies(test_2, columns=['Groups','App','Family','Hairless','Hypoa','Kids','Smart','Type2','DG','Type4','dayofweek','Holidays','Season','am_pm','Unicolor','month','day',
	                                       'Top_Mix','AgeuponOutcome_cat','sprayed','Type_sex','SexuponOutcome_cat','AnimalType_cat'])

#Breed-color-shade with Logistic Regression
from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

def featuring(train):
    #train['Breed_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Breed11']]
    #train['Breed_vec2'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Breed2']]
    #train['Color_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color']]
    #train['Color1_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color1']]
    #train['Color2_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Color2']]
    #train['Shade1_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Shade1']]
    #train['Shade2_vec'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in train['Shade2']]

    #corpustr_br = train['Breed_vec']
    #corpustr_br2 = train['Breed_vec2']
    #corpustr_col = train['Color_vec']
    #corpustr_col1 = train['Color1_vec']
    #corpustr_col2 = train['Color2_vec']
    #corpustr_sh1 = train['Shade1_vec']
    #corpustr_sh2 = train['Shade2_vec']

    #vectorizertr = CountVectorizer(stop_words='english',
    #                             ngram_range = ( 1 , 1 ),analyzer="word", 
    #                             binary=True , token_pattern=r'\w+' )

    #tfidftr_br=vectorizertr.fit_transform(corpustr_br).todense()
    #tfidftr_br2=vectorizertr.fit_transform(corpustr_br2).todense()
    #tfidftr_col=vectorizertr.fit_transform(corpustr_col).todense()
    #tfidftr_col1=vectorizertr.fit_transform(corpustr_col1).todense()
    #tfidftr_col2=vectorizertr.fit_transform(corpustr_col2).todense()
    #tfidftr_sh1=vectorizertr.fit_transform(corpustr_sh1).todense()
    #tfidftr_sh2=vectorizertr.fit_transform(corpustr_sh2).todense()

    #dt1=pd.DataFrame(tfidftr_br,dtype=float).add_suffix('_br1')
    #dt2=pd.DataFrame(tfidftr_col,dtype=float).add_suffix('_col')
    #dt3=pd.DataFrame(tfidftr_col1,dtype=float).add_suffix('_col1')
    #dt4=pd.DataFrame(tfidftr_col2,dtype=float).add_suffix('_col2')
    #dt5=pd.DataFrame(tfidftr_sh1,dtype=float).add_suffix('_sh1')
    #dt6=pd.DataFrame(tfidftr_sh2,dtype=float).add_suffix('_sh2')
    #dt7=pd.DataFrame(tfidftr_br2,dtype=float).add_suffix('_br2')
    binarize=train#.join(dt1)#.join(dt2)#.join(dt3).join(dt4).join(dt5).join(dt6).join(dt7)
    return binarize

train_2['version']='Train'
test_2['version']='Test'

result = train_2.append(test_2, ignore_index=True)

binarize=featuring(result)

binarize_train=binarize[binarize['version']=='Train']
binarize_test=binarize[binarize['version']=='Test']
print(binarize.shape)

train_select=pd.DataFrame(binarize_train.select_dtypes(include=['float64','int64','int32','float32']))
test_select=pd.DataFrame(binarize_test.select_dtypes(include=['float64','int64','int32','float32']))

print(train_select.shape)

train_select=train_select.fillna(-9999)
test_select=test_select.fillna(-9999)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import xgboost as xgb

x_train, x_test, y_train, y_test = train_test_split(train_select,Target,test_size=0.30, 
                                                    random_state=30,stratify=Target)
#simple gmb
#clf_gbm=GradientBoostingClassifier(n_estimators=100,random_state=10).fit(x_train,y_train)
#clf_probs = clf_gbm.predict_proba(x_test)
#print(log_loss(y_test, clf_probs))
#feature importance with gbm
#import matplotlib.pyplot as plt
#feature_importance = clf_gbm.feature_importances_
#feature_importance = 100.0 * (feature_importance / feature_importance.max())
#sorted_idx = np.argsort(feature_importance)
#pos = np.arange(sorted_idx.shape[0]) + .5
#plt.subplot(1, 2, 2)
#plt.barh(pos[134:], feature_importance[sorted_idx][134:], align='center')
#plt.yticks(pos[134:], x_train.columns.values[sorted_idx][134:])
#plt.xlabel('Relative Importance')
#plt.title('Variable Importance')
#plt.show()
x_train, x_test, y_train, y_test = train_test_split(train_select,Target,test_size=0.80, 
                                                    random_state=30,stratify=Target)
dtrain1 = xgb.DMatrix(x_train,y_train,missing = -9999)
dtest1 = xgb.DMatrix(x_test,missing = -9999)
num_round = 125

#dtrain = xgb.DMatrix(train_select,Target,missing = -9999)
#dtest = xgb.DMatrix(test_select,missing = -9999)
num_round = 125

param1 = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}

param2 = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}

param3 = {'max_depth':8, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.65,'colsample_bytree':0.75}

param4 = {'max_depth':9, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.55,'colsample_bytree':0.65}

param5 = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':1,'colsample_bytree':1}

bst1 = xgb.train(param1, dtrain1, num_round)
bst2 = xgb.train(param2, dtrain1, num_round)
bst3 = xgb.train(param3, dtrain1, num_round)
bst4 = xgb.train(param3, dtrain1, num_round)
bst5 = xgb.train(param3, dtrain1, num_round)

ypred_submit = (bst1.predict(dtest) + bst2.predict(dtest) + bst3.predict(dtest) +  bst4.predict(dtest) +  bst5.predict(dtest))/5
print(log_loss(y_test, ypred_submit))
param2 = {'max_depth':4, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}

x_train2, x_test2, y_train2, y_test2 = train_test_split(ypred_submit,y_test,test_size=0.80, 
                                                    random_state=30,stratify=y_test)

dtrain2 = xgb.DMatrix(x_train2,y_train2,missing = -9999)
dtest2 = xgb.DMatrix(x_test2,missing = -9999)

bst2 = xgb.train(param2, dtrain2, num_round)
ypred_submit2 = bst2.predict(dtest2)
print(log_loss(y_test2, ypred_submit2))

dtrain = xgb.DMatrix(train_select,Target,missing = -9999)
dtest = xgb.DMatrix(test_select,missing = -9999)
num_round = 125

param1 = {'max_depth':7, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.75,'colsample_bytree':0.85}

param2 = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.85,'colsample_bytree':0.75}

param3 = {'max_depth':8, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.65,'colsample_bytree':0.75}

param4 = {'max_depth':9, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':0.55,'colsample_bytree':0.65}

param5 = {'max_depth':12, 'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class':5,
        'eval_metric':'mlogloss','subsample':1,'colsample_bytree':1}

bst1 = xgb.train(param1, dtrain, num_round)
bst2 = xgb.train(param2, dtrain, num_round)
bst3 = xgb.train(param3, dtrain, num_round)
bst4 = xgb.train(param3, dtrain, num_round)
bst5 = xgb.train(param3, dtrain, num_round)

ypred_submit = (bst1.predict(dtest) + bst2.predict(dtest) + bst3.predict(dtest) +  bst4.predict(dtest) +  bst5.predict(dtest))/5
ypred_submit=pd.DataFrame(ypred_submit)

submission = pd.DataFrame()
submission["id"] = Id_ts.values
submission["Adoption"]= ypred_submit[2]
submission["Died"]= ypred_submit[4]
submission["Euthanasia"]= ypred_submit[1]
submission["Return_to_owner"]= ypred_submit[0]
submission["Transfer"]= ypred_submit[3]

submission.to_csv("sub.csv",index=False)