import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_column',None)

pd.set_option('display.max_row',None)
df = pd.read_excel('../input/league-of-legends-world-championship-2019/2019-summer-match-data-OraclesElixir-2019-11-10.xlsx')

df.head()
print('We have',df.shape[1],'attributes to work on.')
teamdf = df.loc[df['position']=='Team',:]

teamdf.head()
playersdf = df.loc[df['position']!='Team',:]

playersdf.head()
print(int((teamdf.shape[0]/2)),'matches played')
bans = teamdf[['ban1','ban2','ban3','ban4','ban5']].melt()

bans.head()
top5bans = bans['value'].value_counts().nlargest(5)

for i in range (0,top5bans.shape[0]):

    print(top5bans.index[i],'was banned',top5bans[i],'times, which is',str("%.2f" %((top5bans[i]/119)*100))+'% of the matches.')
top5picks = playersdf['champion'].value_counts().nlargest(5)

for i in range (0,top5picks.shape[0]):

    print(top5picks.index[i],'was picked',top5picks[i],'times, which is',str("%.2f" %((top5picks[i]/119)*100))+'% of the matches.')
top = playersdf.loc[playersdf['position']=='Top']

mid = playersdf.loc[playersdf['position']=='Middle']

bot = playersdf.loc[playersdf['position']=='ADC']

support = playersdf.loc[playersdf['position']=='Support']

jungle = playersdf.loc[playersdf['position']=='Jungle']
print('Top 5 picked champions in Top:')

print(top['champion'].value_counts().nlargest(5))
print('Top 5 picked champions in Mid:')

print(mid['champion'].value_counts().nlargest(5))
print('Top 5 picked champions in Bot:')

print(bot['champion'].value_counts().nlargest(5))
print('Top 5 picked champions in Support:')

print(support['champion'].value_counts().nlargest(5))
print('Top 5 picked champions in Jungle:')

print(jungle['champion'].value_counts().nlargest(5))
timedf = teamdf.loc[teamdf['side']=='Red'] #picking only one side of the match to get the times only once

timedf.head(3)
timedf['gamelength'].describe()
maxtime = teamdf.loc[teamdf['gamelength']==teamdf['gamelength'].max()]

mintime = teamdf.loc[teamdf['gamelength']==teamdf['gamelength'].min()]

print("Maximum Match Time is game number {} between {} and {}".format(maxtime['game'].tolist()[0],maxtime['team'].tolist()[0],maxtime['team'].tolist()[1]))

print("Minimum Match Time is game number {} between {} and {}".format(mintime['game'].tolist()[0],mintime['team'].tolist()[0],mintime['team'].tolist()[1]))
side = teamdf[['side','result']]

side.groupby('side')['result'].value_counts()
maxkills = teamdf.loc[teamdf['teamkills']==teamdf['teamkills'].max()]

minkills = teamdf.loc[teamdf['teamkills']==teamdf['teamkills'].min()]

print('Max kills for a team in a single game is {} for {} against {}'.format(maxkills['teamkills'].tolist()[0],maxkills['team'].tolist()[0],teamdf.loc[teamdf['gameid']==maxkills['gameid'].tolist()[0]]['team'].tolist()[0]))

print('Min kills for a team in a single game is {} for {} against {}'.format(minkills['teamkills'].tolist()[0],minkills['team'].tolist()[0],teamdf.loc[teamdf['gameid']==minkills['gameid'].tolist()[0]]['team'].tolist()[1]))
maxdeaths = teamdf.loc[teamdf['teamdeaths']==teamdf['teamdeaths'].max()]

mindeaths = teamdf.loc[teamdf['teamdeaths']==teamdf['teamdeaths'].min()]

print('Max deaths for a team in a single game is {} for {} against {}'.format(maxdeaths['teamdeaths'].tolist()[0],maxdeaths['team'].tolist()[0],teamdf.loc[teamdf['gameid']==maxdeaths['gameid'].tolist()[0]]['team'].tolist()[1]))

print('Min deaths for a team in a single game is {} for {} against {}'.format(mindeaths['teamdeaths'].tolist()[0],mindeaths['team'].tolist()[0],teamdf.loc[teamdf['gameid']==mindeaths['gameid'].tolist()[0]]['team'].tolist()[1]))
print("There was {} Double Kills".format(playersdf['doubles'].sum()))

print("There was {} Triple Kills".format(playersdf['triples'].sum()))

print("There was {} Quadra Kills".format(playersdf['quadras'].sum()))

print("There was {} Penta Kills".format(playersdf['pentas'].sum()))
player = playersdf[['player','doubles','triples','quadras','pentas','k','d','a','fb','fbvictim','fbassist','fbtime']]

playerdoubles = player.groupby('player')['doubles'].sum()

playertriples = player.groupby('player')['triples'].sum()

playerquadras = player.groupby('player')['quadras'].sum()

playerpentas = player.groupby('player')['pentas'].sum()

print("Max Double kills is {} by {}".format(playerdoubles.max(),playerdoubles.loc[playerdoubles == playerdoubles.max()].index[0]))

print("Max Triple kills is {} by {}".format(playertriples.max(),playertriples.loc[playertriples == playertriples.max()].index[0]))

print("Max Quadra kills is {} by {}".format(playerquadras.max(),playerquadras.loc[playerquadras == playerquadras.max()].index[0]))

print("Max Penta kills is {} by {}".format(playerpentas.max(),playerpentas.loc[playerpentas == playerpentas.max()].index[0]))
playerkills = player.groupby('player')['k'].sum()

playerdeaths = player.groupby('player')['d'].sum()

playerassists = player.groupby('player')['a'].sum()

print("Max kills is {} by {}".format(playerkills.max(),playerkills.loc[playerkills == playerkills.max()].index[0]))

print("Max deaths is {} by {}".format(playerdeaths.max(),playerdeaths.loc[playerdeaths == playerdeaths.max()].index[0]))

print("Max assists is {} by {}".format(playerassists.max(),playerassists.loc[playerassists == playerassists.max()].index[0]))

print("Min kills is {} by {}".format(playerkills.min(),playerkills.loc[playerkills == playerkills.min()].index[0]))

print("Min deaths is {} by {}".format(playerdeaths.min(),playerdeaths.loc[playerdeaths == playerdeaths.min()].index[0]))

print("Min assists is {} by {}".format(playerassists.min(),playerassists.loc[playerassists == playerassists.min()].index[0]))
firstbloodkill = player.groupby('player')['fb'].sum()

firstbloodassist = player.groupby('player')['fbassist'].sum()

firstblooddeath = player.groupby('player')['fbvictim'].sum()

print("Max First Blood Kills is {} by {}".format(firstbloodkill.max(),firstbloodkill.loc[firstbloodkill == firstbloodkill.max()].index[0]))

print("Max First Blood Assist is {} by {}".format(firstbloodassist.max(),firstbloodassist.loc[firstbloodassist == firstbloodassist.max()].index[0]))

print("Max Fist Blood Victim is {} by {}".format(firstblooddeath.max(),firstblooddeath.loc[firstblooddeath == firstblooddeath.max()].index[0]))
LowestFBt = player['fbtime'].min()

HighestFBt = player['fbtime'].max()

lowteam = teamdf.loc[teamdf['fbtime']==player['fbtime'].min()]['team'].tolist()

highteam = teamdf.loc[teamdf['fbtime']==player['fbtime'].max()]['team'].tolist()

lowkiller = playersdf.loc[(playersdf['fbtime']==player['fbtime'].min()) & (playersdf['fb']==1) ]['player'].tolist()[0]

lowvictim = playersdf.loc[(playersdf['fbtime']==player['fbtime'].min()) & (playersdf['fbvictim']==1)]['player'].tolist()[0]

highkiller = playersdf.loc[(playersdf['fbtime']==player['fbtime'].max()) & (playersdf['fb']==1) ]['player'].tolist()[0]

highvictim = playersdf.loc[(playersdf['fbtime']==player['fbtime'].max()) & (playersdf['fbvictim']==1)]['player'].tolist()[0]

print('Fastest First Blood happend after',"%.2f" %LowestFBt,'minutes in the match between {} and {}, {} was killed by {}'.format(lowteam[0],lowteam[1],lowvictim,lowkiller))

print('Slowest First Blood happend after',"%.2f" %HighestFBt,'minutes in the match between {} and {}, {} was killed by {}'.format(highteam[0],highteam[1],highvictim,highkiller))
kpmteamh = teamdf.loc[teamdf['kpm']==teamdf['kpm'].max()]['team'].tolist()[0]

kpmh = "%.2f" %teamdf.loc[teamdf['kpm']==teamdf['kpm'].max()]['kpm'].tolist()[0]

kpmteaml = teamdf.loc[teamdf['kpm']==teamdf['kpm'].min()]['team'].tolist()[0]

kpml = "%.2f" %teamdf.loc[teamdf['kpm']==teamdf['kpm'].min()]['kpm'].tolist()[0]

gameh = teamdf.loc[teamdf['kpm']==teamdf['kpm'].max()]['gameid'].tolist()[0]

gamel = teamdf.loc[teamdf['kpm']==teamdf['kpm'].min()]['gameid'].tolist()[0]

opkpmteamh = teamdf.loc[teamdf['gameid']==gameh]['team'].tolist()[1]

opkpmteaml = teamdf.loc[teamdf['gameid']==gamel]['team'].tolist()[1]

print('Highest Kill Per Minute Team is {} with {} KPM against {}.'.format(kpmteamh,kpmh,opkpmteamh))

print('Lowest Kill Per Minute Team is {} with {} KPM against {}.'.format(kpmteaml,kpml,opkpmteaml))

print('Average Kill Per Minute for teams in all matches is {} KPM.'.format("%.2f" %teamdf['kpm'].mean()))
kpmplayerh = playersdf.loc[playersdf['kpm']==playersdf['kpm'].max()]['player'].tolist()[0]

pkpmh = "%.2f" %playersdf.loc[playersdf['kpm']==playersdf['kpm'].max()]['kpm'].tolist()[0]

kpmplayerl = playersdf.loc[playersdf['kpm']==playersdf['kpm'].min()]['player'].tolist()[0]

pkpml = "%.2f" %playersdf.loc[playersdf['kpm']==playersdf['kpm'].min()]['kpm'].tolist()[0]

pgameh = playersdf.loc[playersdf['kpm']==playersdf['kpm'].max()]['gameid'].tolist()[0]

pgamel = playersdf.loc[playersdf['kpm']==playersdf['kpm'].min()]['gameid'].tolist()[0]

opkpmteamh = teamdf.loc[teamdf['gameid']==pgameh]['team'].tolist()[0]

opkpmteaml = teamdf.loc[teamdf['gameid']==pgamel]['team'].tolist()[1]

print("Highest Kill Per Minute Player is {} with {} KPM against {}.".format(kpmplayerh,pkpmh,opkpmteamh))

print("Lowest Kill Per Minute Player is {} with {} KPM against {}.".format(kpmplayerl,pkpml,opkpmteaml))

print('Average Kill Per Minute for players in all matches is {} KPM.'.format("%.2f" %playersdf['kpm'].mean()))
avgkpmt = teamdf.groupby('team')['kpm'].mean()

avgkpmp = playersdf.groupby('player')['kpm'].mean()

print('The Highest Average KPM Team is {} with {} KPM.'.format(avgkpmt.loc[avgkpmt == avgkpmt.max()].index.tolist()[0],"%.2f" %avgkpmt.max()))

print('The Lowest Average KPM Team is {} with {} KPM.'.format(avgkpmt.loc[avgkpmt == avgkpmt.min()].index.tolist()[0],"%.2f" %avgkpmt.min()))

print('The Highest Average KPM Player is {} with {} KPM.'.format(avgkpmp.loc[avgkpmp == avgkpmp.max()].index.tolist()[0],"%.2f" %avgkpmp.max()))

print('The Lowest Average KPM Player is {} with {} KPM.'.format(avgkpmp.loc[avgkpmp == avgkpmp.min()].index.tolist()[0],"%.2f" %avgkpmp.min()))
#Let me check that

playersdf.loc[playersdf['player']=='Lwx']['kpm'].mean()
rules = ['Top','Jungle','Middle','ADC','Support']

for i in rules:

    kda = playersdf.loc[playersdf['position']==i].groupby(['player'])['k','d','a'].sum()

    kda['kda'] = (kda['k']+kda['a'])/kda['d'] 

    print('{} has the highest KDA as {} with {} KDA.'.format(kda.loc[kda['kda']==kda['kda'].max()].index.tolist()[0],i,"%.2f" %kda['kda'].max()))
kda2 = playersdf.loc[playersdf['position']=='Top'].groupby(['player'])['k','d','a'].sum()

kda2['kda'] = (kda2['k']+kda2['a'])/kda2['d'] 

kda2.sort_values(by=['kda'],ascending=False).head()
playersdf.loc[playersdf['position']=='Top']['player'].unique()
playersdf.loc[(playersdf['position']=='Top') & (playersdf['player']=='ShowMaker')]
df.loc[(df['gameid']==1070445)]
rules = ['Top','Jungle','Middle','ADC','Support']

for i in rules:

    kda = playersdf.loc[playersdf['position']==i].groupby(['player'])['k','d','a'].sum()

    kda['kda'] = (kda['k']+kda['a'])/kda['d'] 

    print('{} has the lowest KDA as {} with {} KDA.'.format(kda.loc[kda['kda']==kda['kda'].min()].index.tolist()[0],i,"%.2f" %kda['kda'].min()))
teamdf.groupby('team')['k'].sum().nlargest(5)
teamdf.groupby('team')['d'].sum().nlargest(5)
champplays = playersdf.groupby('champion')['result'].count()

champplays = champplays.reset_index()



champwins = playersdf.loc[playersdf['result']==1].groupby('champion')['result'].count()

champwins = champwins.reset_index()



champplays.rename(columns={'result':'number of plays'},inplace=True)

champwins.rename(columns={'result':'number of wins'},inplace=True)



champwinrate = champplays.merge(champwins)

champwinrate['Win Rate'] = champwinrate['number of wins']/champwinrate['number of plays']

champwinrate.loc[champwinrate['Win Rate'] == champwinrate['Win Rate'].max()]
champwins.sort_values(by=['number of wins'],ascending=False).head()
playersdf.groupby('champion')['position'].value_counts()
DragonKilled = teamdf.groupby('team')['teamdragkills'].sum()

DragonKilled = DragonKilled.reset_index()

DragonKilled = DragonKilled.sort_values(by=['teamdragkills'],ascending=False)

DragonKilled.head()
DragonKilled.tail()
DrakeKilled = teamdf.groupby('team')['elementals'].sum()

DrakeKilled = DrakeKilled.reset_index()

DrakeKilled = DrakeKilled.sort_values(by=['elementals'],ascending=False)

DrakeKilled.head()
gamedragon = teamdf.groupby('gameid')['teamdragkills'].sum()

gamedragon = gamedragon.reset_index()

gamedragon = gamedragon.sort_values(by=['teamdragkills'],ascending=False)

gamedragon.head()
teamdf.loc[(teamdf['gameid']==1070555 )] 
teamdf.loc[(teamdf['gameid']==1071627 )]
teamdf.groupby('team')['teamdragkills'].sum().sum()
firedrakesKilled = teamdf.groupby('team')['firedrakes'].sum()

firedrakesKilled = firedrakesKilled.reset_index()

firedrakesKilled = firedrakesKilled.sort_values(by=['firedrakes'],ascending=False)

firedrakesKilled.head()
teamdf.groupby('team')['firedrakes'].sum().sum()
waterdrakesKilled = teamdf.groupby('team')['waterdrakes'].sum()

waterdrakesKilled = waterdrakesKilled.reset_index()

waterdrakesKilled = waterdrakesKilled.sort_values(by=['waterdrakes'],ascending=False)

waterdrakesKilled.head()
teamdf.groupby('team')['waterdrakes'].sum().sum()
earthdrakesKilled = teamdf.groupby('team')['earthdrakes'].sum()

earthdrakesKilled = earthdrakesKilled.reset_index()

earthdrakesKilled = earthdrakesKilled.sort_values(by=['earthdrakes'],ascending=False)

earthdrakesKilled.head()
teamdf.groupby('team')['earthdrakes'].sum().sum()
airdrakesKilled = teamdf.groupby('team')['airdrakes'].sum()

airdrakesKilled = airdrakesKilled.reset_index()

airdrakesKilled = airdrakesKilled.sort_values(by=['airdrakes'],ascending=False)

airdrakesKilled.head()

teamdf.groupby('team')['airdrakes'].sum().sum()
eldersKilled = teamdf.groupby('team')['elders'].sum()

eldersKilled = eldersKilled.reset_index()

eldersKilled = eldersKilled.sort_values(by=['elders'],ascending=False)

eldersKilled.head()
teamdf.groupby('team')['elders'].sum().sum()
teamdf.groupby('team')['teamdragkills'].sum().nlargest(5)
teamdf.loc[teamdf['gameid']==1070555]
maxdragkillsteam = teamdf.groupby(['gameid','team'])['teamdragkills'].sum()

maxdragkillsteam = maxdragkillsteam.reset_index()

maxdragkillsteam.loc[maxdragkillsteam['teamdragkills'] == maxdragkillsteam['teamdragkills'].max()]
teamdf.loc[teamdf['gameid']==1060790]
teamdf.loc[teamdf['gameid']==1071398]
teamdf.loc[teamdf['gameid']==1072193]
teamdf.groupby('team')['herald'].sum().nlargest(5)
heraldtime = teamdf.groupby('team')['heraldtime'].mean()

heraldtime = heraldtime.reset_index()

FastestHerald = heraldtime.loc[heraldtime['heraldtime']==heraldtime['heraldtime'].min()]

SlowestHerald = heraldtime.loc[heraldtime['heraldtime']==heraldtime['heraldtime'].max()]

print(FastestHerald)

print(SlowestHerald)
teamdf['heraldtime'].min()
teamdf.loc[teamdf['heraldtime'] == teamdf['heraldtime'].min()]
teamdf.groupby('team')['ft'].sum().nlargest(5)
ftwr = teamdf.groupby(['team','ft'])['result'].sum()

ftwr = ftwr.reset_index()

ftwr.head()
ftwr.groupby('ft')['result'].sum()
83/(36+83)
maxftt = teamdf['fttime'].max()

maxfttgameid = teamdf.loc[teamdf['fttime']==teamdf['fttime'].max()]['gameid'].tolist()[0]

maxfttteams = teamdf.loc[teamdf['gameid']==maxfttgameid]['team'].tolist()

maxfttteam = teamdf.loc[teamdf['fttime']==teamdf['fttime'].max()]['team'].tolist()[1]

print('Longest Time to kill the first tower was {} minutes in the game between {} and {} and was killed by {}.'.format("%.2f" %maxftt,maxfttteams[0],maxfttteams[1],maxfttteam))
minftt = teamdf['fttime'].min()

minfttgameid = teamdf.loc[teamdf['fttime']==teamdf['fttime'].min()]['gameid'].tolist()[0]

minfttteams = teamdf.loc[teamdf['gameid']==minfttgameid]['team'].tolist()

minfttteam = teamdf.loc[teamdf['fttime']==teamdf['fttime'].min()]['team'].tolist()[0]

print('Shortest Time to kill the first tower was {} minutes in the game between {} and {} and was killed by {}.'.format("%.2f" %minftt,minfttteams[0],minfttteams[1],minfttteam))
print("Average time taken to kill the first tower is","%.2f" %teamdf['fttime'].mean(),"minutes.")
teamdf.groupby('team')['firstmidouter'].sum().nlargest(5)
ftwr = teamdf.groupby(['team','firstmidouter'])['result'].sum()

ftwr = ftwr.reset_index()

ftwr.groupby('firstmidouter')['result'].sum()
90/119
teamdf.groupby('team')['firsttothreetowers'].sum().nlargest(5)
ftwr = teamdf.groupby(['team','firsttothreetowers'])['result'].sum()

ftwr = ftwr.reset_index()

ftwr.groupby('firsttothreetowers')['result'].sum()
96/119
teamdf.groupby('team')['teamtowerkills'].sum().nlargest(5)
teamdf.groupby('team')['fbaron'].sum().nlargest(5)
ftwr = teamdf.groupby(['team','fbaron'])['result'].sum()

ftwr = ftwr.reset_index()

ftwr.groupby('fbaron')['result'].sum()
101/119
maxfb = teamdf['fbarontime'].max()

maxfbgameid = teamdf.loc[teamdf['fbarontime']==teamdf['fbarontime'].max()]['gameid'].tolist()[0]

maxfbteams = teamdf.loc[teamdf['gameid']==maxfbgameid]['team'].tolist()

maxfbteam = teamdf.loc[teamdf['fbarontime']==teamdf['fbarontime'].max()]['team'].tolist()[0]

print('Longest Time to kill the first Baron was {} minutes in the game between {} and {} and was killed by {}.'.format("%.2f" %maxfb,maxfbteams[0],maxfbteams[1],maxfbteam))
minfb = teamdf['fbarontime'].min()

minfbgameid = teamdf.loc[teamdf['fbarontime']==teamdf['fbarontime'].min()]['gameid'].tolist()[0]

minfbteams = teamdf.loc[teamdf['gameid']==minfbgameid]['team'].tolist()

minfbteam = teamdf.loc[teamdf['fbarontime']==teamdf['fbarontime'].min()]['team'].tolist()[1]

print('Shortest Time to kill the first baron was {} minutes in the game between {} and {} and was killed by {}.'.format("%.2f" %minfb,minfbteams[0],minfbteams[1],minfbteam))
teamdf.groupby('team')['teambaronkills'].sum().nlargest(5)
teamdf.groupby('team')['dmgtochamps'].sum().nlargest(5)
teamdf.groupby('team')['dmgtochamps'].mean().nlargest(5)
teamdf.groupby('team')['dmgtochamps'].max().nlargest(5)
DamageteamGameID = teamdf.loc[teamdf['dmgtochamps']==teamdf['dmgtochamps'].max()]['gameid'].tolist()[0]

teamdf.loc[teamdf['gameid']==DamageteamGameID]
teamdf.groupby('gameid')['dmgtochamps'].sum().max()
playersdf.groupby('player')['dmgtochamps'].sum().nlargest(5)
playersdf.groupby('player')['dmgtochamps'].mean().nlargest(5)
playersdf.groupby(['player'])['dmgtochamps'].max().nlargest(5)
DamagePlayerGameID = playersdf.loc[playersdf['dmgtochamps']==playersdf['dmgtochamps'].max()]['gameid'].tolist()[0]

playersdf.loc[playersdf['gameid']==DamagePlayerGameID]
teamdf.groupby('team')['dmgtochampsperminute'].mean().nlargest(5)
playersdf.groupby('player')['dmgtochampsperminute'].mean().nlargest(5)
playersdf.groupby('player')['dmgshare'].mean().nlargest(5)
playersdf.groupby('player')['earnedgoldshare'].mean().nlargest(5)
ds = []

for i in playersdf['dmgshare'].tolist():

    try:

      ds.append(float(i))

    except:

      ds.append(float(0)) 

playersdf['damageshare'] = ds
playersdf[['player','dmgshare','damageshare']].head()
gs = []

for i in playersdf['earnedgoldshare'].tolist():

    try:

      gs.append(float(i))

    except:

      gs.append(float(0)) 

playersdf['goldshare'] = gs
playersdf[['player','earnedgoldshare','goldshare']].head()
playersdf.groupby('player')['damageshare'].mean().nlargest(5)
playersdf.groupby('player')['goldshare'].mean().nlargest(5)
teamdf.groupby('team')['wards'].sum().nlargest(5)
teamdf.groupby('team')['wards'].mean().nlargest(5)
playersdf.groupby('player')['wards'].sum().nlargest(5)
playersdf.groupby('player')['wards'].mean().nlargest(5)
teamdf.groupby('team')['wpm'].mean().nlargest(5)
playersdf.groupby('player')['wpm'].mean().nlargest(5)
playersdf.groupby('player')['wardshare'].mean().nlargest(5)
teamdf.groupby('team')['wardkills'].sum().nlargest(5)
playersdf.groupby('player')['wardkills'].sum().nlargest(5)
teamdf.groupby('team')['wcpm'].mean().nlargest(5)
teamdf.groupby('team')['visionwards'].sum().nlargest(5)
playersdf.groupby('player')['visionwards'].sum().nlargest(5)
teamdf.groupby('team')['totalgold'].sum().nlargest(5)
playersdf.groupby('player')['totalgold'].sum().nlargest(5)
teamdf.groupby('team')['totalgold'].mean().nlargest(5)
playersdf.groupby('player')['totalgold'].mean().nlargest(5)
teamdf.groupby('team')['earnedgpm'].mean().nlargest(5)
playersdf.groupby('player')['earnedgpm'].mean().nlargest(5)
teamdf.groupby('team')['minionkills'].sum().nlargest(5)
playersdf.groupby('player')['minionkills'].sum().nlargest(5)
playersdf.groupby('player')['monsterkills'].sum().nlargest(5)
teamdf.groupby('team')['cspm'].mean().nlargest(5)
playersdf.groupby('player')['cspm'].mean().nlargest(5)
teamdf.groupby('team')['csdat10'].mean().nlargest(5)
playersdf.groupby('player')['csdat10'].mean().nlargest(5)
teamdf.groupby('team')['gdat15'].mean().nlargest(5)
playersdf.groupby(['player'])['gdat15'].mean().nlargest(5)
teamdf.groupby('team')['xpdat10'].mean().nlargest(5)
playersdf.groupby('player')['xpdat10'].mean().nlargest(5)