import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm

import os
print(os.listdir("../input"))
YGO_df=pd.read_csv('../input/YGO_Cards_v2.csv',encoding = "ISO-8859-1")
YGO_df.rename(columns={'Unnamed: 0':'Name'},inplace=True)
YGO_df.head()
YGO_df['attack'].replace('?',np.nan,inplace=True)
YGO_df['attack'].replace('X000',np.nan,inplace=True)
YGO_df['attack'].replace('---',np.nan,inplace=True)
YGO_df['defense'].replace('?',np.nan,inplace=True)
YGO_df['defense'].replace('X000',np.nan,inplace=True)
YGO_df['defense'].replace('---',np.nan,inplace=True)
YGO_df['number'].replace('None',np.nan,inplace=True)
YGO_df.dtypes
YGO_df.attack=pd.to_numeric(YGO_df['attack'])
YGO_df.defense=pd.to_numeric(YGO_df['defense'])
YGO_df.number=pd.to_numeric(YGO_df['number'])
YGO_df.columns
YGO_df.attack.fillna(-1,inplace=True)
YGO_df.defense.fillna(-1,inplace=True)
YGO_df.stars.fillna(0,inplace=True)
YGO_df.link_number.fillna(0,inplace=True)
YGO_df.pendulum_left.fillna(-1,inplace=True)
YGO_df.pendulum_right.fillna(-1,inplace=True)
YGO_df.number.fillna(0,inplace=True)
YGO_df[['attack','defense','stars']].describe()
#Plot of value counts for Attack
_=plt.style.use('fivethirtyeight')
ax=YGO_df.attack.value_counts().plot.bar(figsize=(25,10),rot=90,
                                                   title='Value counts for Monster Attack')
_=ax.set(xlabel='Attack',ylabel='Count')
#Histogram of attack
plt.figure(figsize=(16,8))
_=plt.hist(YGO_df['attack'])
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Frequency',fontsize=25)
_=plt.title('Attack Frequency in Yu-Gi-Oh!')
plt.show()
#Function for mapping density to colours
def makeColours( vals ):
    colours = np.zeros( (len(vals),3) )
    norm = Normalize( vmin=vals.min(), vmax=vals.max() )

    #Can put any colormap you like here.
    colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]

    return colours
#Scatter Plot of Attack vs Defense
densObj = kde([YGO_df.attack,YGO_df.defense])

colours = makeColours( densObj.evaluate([YGO_df.attack,YGO_df.defense]) )

plt.figure(figsize=(12,12))
_=plt.scatter('attack','defense',data=YGO_df,color=colours,s=50)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Defense',fontsize=25)
_=plt.title('Attack Vs Defense')
plt.show()
#Scatterplot of attack vs Stars
densObj = kde([YGO_df.attack,YGO_df.stars])
colours = makeColours( densObj.evaluate([YGO_df.attack,YGO_df.stars]) )

plt.figure(figsize=(12,12))
_=plt.scatter('attack','stars',data=YGO_df,color=colours,s=50)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Stars',fontsize=25)
_=plt.title('Attack Vs Stars (Level/Rank)')
plt.show()
#Above colour-coded to separate Xyz from not-Xyz
sns.set(font_scale=2)
g=sns.pairplot(x_vars=['attack'], y_vars=['stars'], data=YGO_df, hue="is_xyz",
               height=12,plot_kws={"s": 100})
_=g._legend.set_title('Xyz?')
new_labels = ['No', 'Yes']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
_=g.set(xlabel='Attack',ylabel='Stars',title='Attack Vs Stars (Xyz separated)')
#Scatter Plot of Attack vs Link Number
plt.figure(figsize=(12,12))
_=plt.scatter('attack','link_number',data=YGO_df,s=50)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Link Number',fontsize=25)
_=plt.ylim([0.5,5.5])
_=plt.xlim([-100,3100])
_=plt.title('Attack Vs Link Number')
plt.show()
#Scatter plot of attack against pendulum scale
plt.figure(figsize=(12,12))
_=plt.scatter('attack','pendulum_left',data=YGO_df,s=50)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Pendulum Scale',fontsize=25)
_=plt.title('Attack Vs Pendulum Scale')
_=plt.ylim([-0.5,14])
_=plt.xlim([-100,4100])
plt.show()
#Scatterplot of attack vs number
densObj = kde([YGO_df.attack,YGO_df.number])
colours = makeColours( densObj.evaluate([YGO_df.attack,YGO_df.number]) )

plt.figure(figsize=(12,12))
_=plt.scatter('attack','number',data=YGO_df,color=colours)
_=plt.xlabel('Attack',fontsize=25)
_=plt.ylabel('Passcode',fontsize=25)
_=plt.title('Attack Vs Passcode')
plt.show()
YGO_df.groupby(['attribute']).describe()[['attack']]
#Plot of attack wrt Attribute
fig, ax = plt.subplots(figsize=(12, 12))
my_pal = {"EARTH": "brown", "WATER": "blue", "WIND":"green","LIGHT":"yellow",
          "DARK":"purple","FIRE":"red","DIVINE":'gold'}
g=sns.boxplot(y='attribute',x='attack',data=YGO_df,palette=my_pal)
_=g.set(ylabel='Attribute',xlabel='Attack',title="Attack Variation with Attribute")
YGO_df.groupby(['Type']).describe()[['attack']]
#Plot of attack wrt Type
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='Type',x='attack',data=YGO_df)
_=g.set(ylabel='Type',xlabel='Attack',title='Attack Variation with Type')
#Plot of value counts for Defense
ax=YGO_df.defense.value_counts().plot.bar(figsize=(25,10),rot=90,
                                                   title='Value counts for Monster Defense')
_=ax.set(xlabel='Defense',ylabel='Count')
#Histogram of defense
plt.figure(figsize=(16,8))
_=plt.hist(YGO_df['defense'])
_=plt.xlabel('Defense',fontsize=25)
_=plt.ylabel('Frequency',fontsize=25)
_=plt.title('Defense Frequency in Yu-Gi-Oh!')
plt.show()
#Scatterplot of defense vs Stars
densObj = kde([YGO_df.defense,YGO_df.stars])
colours = makeColours( densObj.evaluate([YGO_df.defense,YGO_df.stars]) )

plt.figure(figsize=(12,12))
_=plt.scatter('defense','stars',data=YGO_df,color=colours,s=50)
_=plt.xlabel('Defense',fontsize=25)
_=plt.ylabel('Stars',fontsize=25)
_=plt.title('Defense Vs Stars (Level/Rank)')
plt.show()
#Above colour-coded to separate Xyz from not-Xyz
sns.set(font_scale=2)
g=sns.pairplot(x_vars=['defense'], y_vars=['stars'], data=YGO_df, hue="is_xyz",
               height=12,plot_kws={"s": 100})
_=g._legend.set_title('Xyz?')
new_labels = ['No', 'Yes']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
_=g.set(xlabel='Defense',ylabel='Stars',title='Defense Vs Stars (Xyz separated)')
#Scatter plot of defense against pendulum scale
plt.figure(figsize=(12,12))
_=plt.scatter('defense','pendulum_left',data=YGO_df,s=50)
_=plt.xlabel('Defense',fontsize=25)
_=plt.ylabel('Pendulum Scale',fontsize=25)
_=plt.title('Defense Vs Pendulum Scale')
_=plt.ylim([-0.5,14])
_=plt.xlim([-100,4100])
plt.show()
#Scatterplot of defense vs number
densObj = kde([YGO_df.defense,YGO_df.number])
colours = makeColours( densObj.evaluate([YGO_df.defense,YGO_df.number]) )

plt.figure(figsize=(12,12))
_=plt.scatter('defense','number',data=YGO_df,color=colours)
_=plt.xlabel('Defense',fontsize=25)
_=plt.ylabel('Passcode',fontsize=25)
_=plt.title('Defense Vs Passcode')
plt.show()
YGO_df.groupby(['attribute']).describe()[['defense']]
#Plot of defense wrt Attribute
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='attribute',x='defense',data=YGO_df,palette=my_pal)
_=g.set(ylabel='Attribute',xlabel='Defense',title="Defense Variation with Attribute")
YGO_df.groupby(['Type']).describe()[['defense']]
#Plot of defense wrt Type
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='Type',x='defense',data=YGO_df)
_=g.set(ylabel='Type',xlabel='Defense',title='Defense Variation with Type')
#Plot of value counts for Stars
ax=YGO_df.stars.value_counts().plot.bar(figsize=(20,10),rot=90,
                                                   title='Value counts for Monster Stars')
_=ax.set(xlabel='Stars',ylabel='Count')
#Scatter plot of Stars against pendulum scale
plt.figure(figsize=(12,12))
_=plt.scatter('stars','pendulum_left',data=YGO_df,s=50)
_=plt.xlabel('Stars',fontsize=25)
_=plt.ylabel('Pendulum Scale',fontsize=25)
_=plt.title('Stars Vs Pendulum Scale')
_=plt.ylim([-0.5,14])
#_=plt.xlim([-100,4100])
plt.show()
#Scatterplot of Stars vs number
densObj = kde([YGO_df.stars,YGO_df.number])
colours = makeColours( densObj.evaluate([YGO_df.stars,YGO_df.number]) )

plt.figure(figsize=(12,12))
_=plt.scatter('stars','number',data=YGO_df,color=colours)
_=plt.xlabel('Stars',fontsize=25)
_=plt.ylabel('Passcode',fontsize=25)
_=plt.title('Stars Vs Passcode')
plt.show()
YGO_df.groupby(['attribute']).describe()[['stars']]
#Plot of Stars wrt Attribute
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='attribute',x='stars',data=YGO_df,palette=my_pal)
_=g.set(ylabel='Attribute',xlabel='Stars',title="Stars Variation with Attribute")
YGO_df.groupby(['Type']).describe()[['stars']]
#Plot of Stars wrt Type
fig, ax = plt.subplots(figsize=(12, 12))
g=sns.boxplot(y='Type',x='stars',data=YGO_df)
_=g.set(ylabel='Type',xlabel='Stars',title='Stars Variation with Type')
YGO_df.monster_types.value_counts()
YGO_df[YGO_df.monster_types=='[]']
#Not the best way to do this, but don't feel like generating names from a loop
YGO_df['is_effect']=False
for idx, row in YGO_df.iterrows():
    if 'Effect' in row['monster_types']:
        YGO_df.loc[idx, 'is_effect'] = True

for idx, row in YGO_df.iterrows():
    if '[]' in row['monster_types']:
        YGO_df.loc[idx, 'is_effect'] = True
        
YGO_df['is_normal']=False        
for idx, row in YGO_df.iterrows():
    if 'Normal' in row['monster_types']:
        YGO_df.loc[idx, 'is_normal'] = True

YGO_df['is_tuner']=False
for idx, row in YGO_df.iterrows():
    if 'Tuner' in row['monster_types']:
        YGO_df.loc[idx, 'is_tuner'] = True

YGO_df['is_flip']=False
for idx, row in YGO_df.iterrows():
    if 'Flip' in row['monster_types']:
        YGO_df.loc[idx, 'is_flip'] = True
        
YGO_df['is_gemini']=False
for idx, row in YGO_df.iterrows():
    if 'Gemini' in row['monster_types']:
        YGO_df.loc[idx, 'is_gemini'] = True
        
YGO_df['is_ritual']=False
for idx, row in YGO_df.iterrows():
    if 'Ritual' in row['monster_types']:
        YGO_df.loc[idx, 'is_ritual'] = True
        
YGO_df['is_spirit']=False
for idx, row in YGO_df.iterrows():
    if 'Spirit' in row['monster_types']:
        YGO_df.loc[idx, 'is_spirit'] = True
        
YGO_df['is_union']=False
for idx, row in YGO_df.iterrows():
    if 'Union' in row['monster_types']:
        YGO_df.loc[idx, 'is_union'] = True
        
YGO_df['is_toon']=False
for idx, row in YGO_df.iterrows():
    if 'Toon' in row['monster_types']:
        YGO_df.loc[idx, 'is_toon'] = True
        
YGO_df['is_token']=False
for idx, row in YGO_df.iterrows():
    if 'Token' in row['monster_types']:
        YGO_df.loc[idx, 'is_token'] = True
#List of categories to consider

type_list=['is_normal','is_effect','is_flip','is_gemini','is_union','is_spirit','is_toon','is_tuner',
           'is_token','is_ritual','is_fusion','is_synchro','is_xyz','is_link','is_pendulum']

#Make a dictionary of value counts for each attribute, grouped by sub-categories
d={}
for i in range(len(type_list)):
    d["Att_{0}".format(type_list[i])]=YGO_df.groupby(type_list[i]).attribute.value_counts()[1]

#Convert to a dataframe,and add the Total columns
Attribute_df=pd.DataFrame(d)
Attribute_df['Total']=YGO_df.attribute.value_counts()
Attribute_df['Total_Perc']=round(Attribute_df.Total/sum(Attribute_df.Total)*100,1)
Attribute_df.fillna(0,inplace=True)

#Add a % column for each
for i in range(len(type_list)):
    Attribute_df["Perc_{0}".format(type_list[i])]=round(Attribute_df.iloc[:,i]/np.nansum(
        Attribute_df.iloc[:,i])*100,1)
    
#Add a change relative to Total, called Delta
for i in range(len(type_list)):
    Attribute_df["Delta_{0}".format(type_list[i])]=round(Attribute_df["Perc_{0}".format(type_list[i])]
    -Attribute_df['Total_Perc'],1)

Attribute_df
#Let's also output some %'s'
print('Overall Attribute distribution')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Total_Perc.values[i])+'%')
    
#Attribute count
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Total,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of All Monsters")
#Let's also output some %'s'
print('Normal Attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_normal.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_normal.values[i])+'%)')
    
#Attribute count
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_normal,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Normal Monsters")
#Effect monsters
print('Effect monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_effect.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_effect.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_effect,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Effect Monsters")
#Flip monsters
print('Flip monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_flip.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_flip.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_flip,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Flip Monsters")
#Gemini monsters
print('Gemini monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_gemini.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_gemini.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_gemini,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Gemini Monsters")
#Union monsters
print('Union monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_union.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_union.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_union,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Union Monsters")
#Spirit monsters
print('Spirit monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_spirit.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_spirit.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_spirit,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Spirit Monsters")
#Toon monsters
print('Toon monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_toon.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_toon.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_toon,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Toon Monsters")
#Tuner monsters
print('Tuner monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_tuner.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_tuner.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_tuner,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Tuner Monsters")
#Token monsters
print('Token monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_token.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_token.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_token,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Token Monsters")
#Ritual monsters
print('Ritual monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_ritual.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_ritual.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_ritual,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Ritual Monsters")
#Fusion monsters
print('Fusion monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_fusion.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_fusion.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_fusion,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Fusion Monsters")
#Synchro monsters
print('Synchro monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_synchro.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_synchro.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_synchro,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Synchro Monsters")
#Xyz monsters
print('Xyz monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_xyz.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_xyz.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_xyz,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Xyz Monsters")
#Pendulum monsters
print('Pendulum monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_pendulum.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_pendulum.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_pendulum,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Pendulum Monsters")
#Link monsters
print('Link monster attribute distribution (Change from Total)')
for i in range(len(Attribute_df.index)):
    print(str(Attribute_df.index[i])+': '+str(Attribute_df.Perc_is_link.values[i])+'%'
         + ' ('+str(Attribute_df.Delta_is_link.values[i])+'%)')

fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Attribute_df.index,y=Attribute_df.Att_is_link,palette=my_pal)
_=g.set(xlabel='Attribute',ylabel='Frequency',title="Attribute Distribution of Link Monsters")
#Make a dictionary of value counts for each type, grouped by sub-categories
d={}
for i in range(len(type_list)):
    d["Att_{0}".format(type_list[i])]=YGO_df.groupby(type_list[i]).Type.value_counts()[1]

#Convert to a dataframe,and add the Total columns
Type_df=pd.DataFrame(d)
Type_df['Total']=YGO_df.Type.value_counts()
Type_df['Total_Perc']=round(Type_df.Total/sum(Type_df.Total)*100,1)

Type_df.fillna(0,inplace=True)

#Add a % column for each
for i in range(len(type_list)):
    Type_df["Perc_{0}".format(type_list[i])]=round(Type_df.iloc[:,i]/np.nansum(
        Type_df.iloc[:,i])*100,1)
    
#Add a change relative to Total, called Delta
for i in range(len(type_list)):
    Type_df["Delta_{0}".format(type_list[i])]=round(Type_df["Perc_{0}".format(type_list[i])]
    -Type_df['Total_Perc'],1)
    
Type_df
#Type overall
print('Overall Type distribution')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Total_Perc.values[i])+'%')
    
#Attribute count
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Total)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of All Monsters")
#Normal
print('Normal Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_normal.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_normal.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_normal)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Normal Monsters")
#Effect
print('Effect Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_effect.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_effect.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_effect)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Effect Monsters")
#Flip
print('Flip Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_flip.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_flip.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_flip)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Flip Monsters")
#Gemini
print('Gemini Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_gemini.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_gemini.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_gemini)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Gemini Monsters")
#Union
print('Union Type distribution (Change from Total)')
for i in range(len(Type_df.index)):
    print(str(Type_df.index[i])+': '+str(Type_df.Perc_is_union.values[i])+'%'
         + ' ('+str(Type_df.Delta_is_union.values[i])+'%)')
    
fig, ax = plt.subplots(figsize=(12, 5))
plt.xticks(rotation=90)
g=sns.barplot(x=Type_df.index,y=Type_df.Att_is_union)
_=g.set(xlabel='Type',ylabel='Frequency',title="Type Distribution of Union Monsters")
