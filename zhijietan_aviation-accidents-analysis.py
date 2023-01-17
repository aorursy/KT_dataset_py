import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
sb.set()
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
lm = LinearRegression()
logmodel = LogisticRegression()
# df = pd.read_csv('AviationData.csv', encoding = 'latin1')
df = pd.read_csv('../input/aviation-accident-database-synopses/AviationData.csv', sep=',', header=0, encoding = 'iso-8859-1')
df.info()
df['Country'].value_counts().head()
df['Investigation.Type'].value_counts()
df = df[df['Country']=='United States']
df['Country'].value_counts()
df = df[df['Investigation.Type']=='Accident']
df['Investigation.Type'].value_counts()
df.info()
df.drop(['Event.Id','Accident.Number','Airport.Code','Airport.Name','Location','Injury.Severity','Registration.Number','FAR.Description','Air.Carrier','Report.Status','Publication.Date','Number.of.Engines'],axis=1,inplace=True)
sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='Blues')
df.drop(['Latitude','Longitude','Aircraft.Category','Schedule'],axis=1,inplace=True)
# Cleaning of Data
df['Total.Fatal.Injuries'].fillna(0, inplace = True)
df['Total.Serious.Injuries'].fillna(0, inplace = True)
df['Total.Minor.Injuries'].fillna(0, inplace = True)
df['Total.Uninjured'].fillna(0, inplace = True)
df['Broad.Phase.of.Flight'].fillna('UNKNOWN',inplace = True)
df['Weather.Condition'].fillna('UNKNOWN',inplace = True)
df['Weather.Condition'].replace({'UNK':'UNKNOWN'},inplace=True)
df['Aircraft.Damage'].fillna('UNKNOWN',inplace=True)
df['Engine.Type'].fillna('UNKNOWN',inplace=True)
df['Purpose.of.Flight'].fillna('Other Work Use',inplace=True)
df['Amateur.Built'].fillna('No',inplace=True)
sb.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='Blues')
df['Total Injuries'] = df['Total.Fatal.Injuries'] + df['Total.Serious.Injuries'] + df['Total.Minor.Injuries']
df['Event.Date'] = pd.to_datetime(df['Event.Date'])
df['Year'] = df['Event.Date'].apply(lambda time : time.year)
df['Month']=df['Event.Date'].apply(lambda time:time.month)
# Only want data after 1982
df = df[df['Year']>=1982]
df.head()
plt.figure(figsize=(20,8))
sb.countplot(df['Year'],palette = 'coolwarm')
plt.figure(figsize=(20,8))
sb.countplot(df['Month'],palette='coolwarm')
accYear=pd.DataFrame(df.groupby("Year").count())
accYear=accYear.drop(columns=['Event.Date'])
accYear=accYear.rename(columns={'Month':'Count'})
accYear.head()

X=[ [y] for y in accYear.index.values ]
y=[[e] for e in accYear['Count']]

lm.fit(X,y)
accPredict_X=[[y] for y in range (1982, 2025)]
accPredict=lm.predict(accPredict_X)

f, axes = plt.subplots(1, 1, figsize=(15, 8))
plt.plot(X,y)
plt.plot(accPredict_X,accPredict, alpha = 0.5)

print("Accident prediction for the next 5 years:\n" )
for i in range (0,5):
    year=2021+i
    n=-5+i
    print('Year %d: %d' % (year,accPredict[n]))
by_year = df.groupby('Year').sum()
plt.figure(figsize=(12,6))
by_year['Total Injuries'].plot(color='blue',fontsize=15,lw=3,markersize=10,marker='o',markerfacecolor='r')
plt.xlabel('Year',fontsize=13)
plt.ylabel('Total Injury Count',fontsize=13)
by_year[['Total.Fatal.Injuries','Total.Serious.Injuries','Total.Minor.Injuries']].plot(lw = 2, figsize=(12,6))
# to move the legend outside of graph
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
plt.figure(figsize=(12,6))
by_year['Total.Uninjured'].plot(color='blue', fontsize = 15,lw = 3,markersize=10,marker='o',markerfacecolor='r')
plt.xlabel('Year',fontsize = 13)
by_year_2 = by_year.reset_index()
X = by_year_2[['Year']]
y = np.asarray(by_year_2['Total.Fatal.Injuries'])
lm.fit(X,y)
injury_predict_X = [[y] for y in range (1982,2025)]
injury_predict = lm.predict(injury_predict_X)

f,axes = plt.subplots(1,1,figsize = (16,8))
plt.plot(X,y)
plt.plot(injury_predict_X,injury_predict, alpha = 1.0)

print("Total Injuries Predictions for the next 5 years:\n" )

for i in range (0,5):
    year=2021+i
    n=-5+i
    print('Year %d: %d' % (year,injury_predict[n]))
X2 = by_year_2[['Year']]
y2 = np.asarray(by_year_2['Total.Uninjured'])
lm.fit(X2, y2)
injury_predict_X2 = [[y] for y in range (1982,2025)]
injury_predict2 = lm.predict(injury_predict_X2)

f,axes = plt.subplots(1,1,figsize = (16,8))
plt.plot(X2,y2)
plt.plot(injury_predict_X2,injury_predict2, alpha = 1.0)

print("Total Uninjured Predictions for the next 5 years:\n" )

for i in range (0,5):
    year=2021+i
    n=-5+i
    print('Year %d: %d' % (year,injury_predict2[n]))
by_phase = df.groupby('Broad.Phase.of.Flight').sum().reset_index()
by_phase = by_phase.drop(['Year','Month'], axis=1)
by_phase
plt.figure(figsize = (14,8))
sb.barplot(x = 'Broad.Phase.of.Flight',y='Total Injuries' , data = by_phase.reset_index() , palette = 'coolwarm', ec = 'black')
plt.title('Phase Of Flight ' , size = 20)
plt.xlabel('')
plt.ylabel('Total Injury Count', size = 20)
plt.tight_layout()
yearPhase = df.groupby(by = ['Year','Broad.Phase.of.Flight']).sum()['Total Injuries'].unstack()
yearPhase.head()
plt.figure(figsize = (20,10))
sb.heatmap(yearPhase, cmap = 'Blues')
plt.xlabel('')
def other_phases(phase):
    if phase in (['UNKNOWN','TAXI','DESCENT','CLIMB','GO-AROUND','STANDING']):
        return 'OTHER'
    else:
        return phase
df['Phases'] = df['Broad.Phase.of.Flight'].apply(other_phases)
plt.figure(figsize=(8,4))
sb.countplot(df['Phases'], palette='coolwarm')
df.groupby('Aircraft.Damage')['Phases'].value_counts()
plt.figure(figsize=(12,6))
sb.countplot(df['Aircraft.Damage'],hue=df['Phases'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
plt.figure(figsize=(12,6))
sb.countplot(df['Weather.Condition'],hue=df['Phases'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
rfc = RandomForestClassifier(n_estimators=300)
df_phase = pd.get_dummies(df,columns=['Aircraft.Damage','Weather.Condition'],drop_first=True)
df_phase.columns
X = df_phase[['Aircraft.Damage_Minor', 'Aircraft.Damage_Substantial',
       'Aircraft.Damage_UNKNOWN','Total Injuries','Total.Uninjured',
       'Weather.Condition_UNKNOWN','Weather.Condition_VMC']]

y = df_phase['Phases']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print('Classification Accuracy: {:.3f}'.format(rfc.score(X_test,y_test)))
df['Purpose.of.Flight'].value_counts()
def personal(purpose):
    if purpose == 'Personal':
        return 1
    
    else:
        return 0
df['Personal Flight'] = df['Purpose.of.Flight'].apply(personal)
sb.countplot(df['Personal Flight'])
sb.countplot(df['Personal Flight'], hue=df['Weather.Condition'])
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
sb.countplot(df['Personal Flight'],hue=df['Aircraft.Damage'])
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
plt.figure(figsize=(12,8))
sb.countplot(df['Personal Flight'],hue=df['Broad.Phase.of.Flight'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
plt.figure(figsize=(18,10))
sb.countplot(df['Personal Flight'],hue=df['Engine.Type'],palette = 'coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
df_new = df[df['Total Injuries']>=1]
df_new = df_new[df_new['Total.Uninjured']>=1]
f,axes = plt.subplots(2,1,figsize=(18,8))
sb.boxplot(x='Total Injuries',y='Personal Flight',data=df_new,orient ='h',ax=axes[0])
sb.boxplot(x='Total.Uninjured',y='Personal Flight',data=df_new,orient='h',ax=axes[1])
df_new = pd.get_dummies(df_new,columns=['Engine.Type'],drop_first=True)
df_new.columns
df_new = df_new[['Personal Flight','Engine.Type_Reciprocating',
       'Engine.Type_Turbo Fan', 'Engine.Type_Turbo Jet',
       'Engine.Type_Turbo Prop', 'Engine.Type_Turbo Shaft',
       'Engine.Type_UNKNOWN', 'Engine.Type_Unknown','Total Injuries','Total.Uninjured']]
df_new.head(2)
X = df_new.drop('Personal Flight',axis=1)
y = df_new['Personal Flight']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print('Classification Accuracy: {:.3f}'.format(logmodel.score(X_test,y_test)))
df.columns
df['Make'].value_counts().head(30)
def get_company(company):
    
    if company in ['Cessna','Piper','Beech']:
        
        return company.upper()
    
    else:
        
        if company in ['CESSNA','PIPER','BEECH']:
            
            return company
        
        else:
            
            return 'OTHER'

df['Make'].apply(get_company).value_counts()
df_selected = df[df['Make'].isin(['CESSNA','PIPER','BEECH'])]
plt.figure(figsize=(12,6))
sb.countplot(df_selected['Make'],hue=df['Phases'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
plt.figure(figsize=(12,6))
sb.countplot(df_selected['Make'],hue=df['Aircraft.Damage'],palette='coolwarm')
plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
f,axes = plt.subplots(3,1,figsize=(18,12))
sb.boxplot(x='Total Injuries',y='Make',data=df_selected,orient ='h',ax=axes[0])
sb.boxplot(x='Total.Uninjured',y='Make',data=df_selected,orient='h',ax=axes[1])
sb.countplot(df_selected['Make'])
