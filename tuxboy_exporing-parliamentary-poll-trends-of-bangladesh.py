import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
from matplotlib import pyplot as plt

import os
data_1991 = pd.read_csv('../input/bangladesh-election-results/Bangladesh1991ElectionResults.csv')
data_1996 = pd.read_csv('../input/bangladesh-election-results/Bangladesh1996ElectionResults.csv')
data_2001 = pd.read_csv('../input/bangladesh-election-results/Bangladesh2001ElectionResults.csv')
data_2008 = pd.read_csv('../input/bangladesh-election-results/Bangladesh2008ElectionResults.csv')

data_1991['year'] = '1991'
data_1996['year'] = '1996'
data_2001['year'] = '2001'
data_2001.rename(columns={'fourPartyAlliancePercentage': 'bangladeshNationalPartyPercentage'}, inplace=True)
data_2008['year'] = '2008'

df = pd.concat([data_1991, data_1996, data_2001, data_2008], sort=False).reset_index()

replacements = {
    'Chadpur': 'Chandpur',
    'Brahmminbaria': 'Brahamanbaria',
    'Hobiganj': 'Habiganj',
    'Jhalakathi': 'Jhalokati',
    'Laxmipur': 'Lakshmipur',
    'Moulvibazar': 'Maulvibazar',
    'Netrokona': 'Netrakona',
    'Narshingdi': 'Narsingdi',
    'Panchaghar': 'Panchagarh',
    'Sirajgonj': 'Sirajganj',
    "Mymensingh-Netrokona": 'Netrakona',
    "MymensinghNetrokona": 'Netrakona',
    'Barisal-Pirojpur': 'Pirojpur',
    'BarisalPirojpur': 'Pirojpur',
    'Borguna': 'Barguna',
    'Brahmminbaria': 'Brahamanbaria',
    'Jhalokathi': 'Jhalokati',
    'Khagrachari': 'Khagracchari',
    'Panchaghar': 'Panchagarh',
    'Shatkhira': 'Satkhira',
    'Sunamgonj': 'Sunamganj',
    'Khagrachari': 'Khagracchari',
    'Brahmanbaria': 'Brahamanbaria'
}

for key, value in replacements.items():
    df['district'] = df['district'].replace(key, value)
    
bd_districts = gpd.read_file('../input/bangladesh-administrative-geojson/districts_amended.json')
registered_voters = df[['year', 'registeredVoters']].dropna().groupby(['year']).sum().reset_index()
voter_turnout = df[['year', 'voterTurnout']].dropna().groupby(['year']).mean().reset_index()

plt.subplot(2, 1, 2)
plt.title('Registered Voters')
plt.xlabel('Year')
plt.bar(registered_voters['year'], registered_voters['registeredVoters'])

plt.subplot(2, 2, 1)
plt.title('Voter Turnout (%)')
plt.xlabel('Year')
plt.bar(voter_turnout['year'], voter_turnout['voterTurnout'], color='green')

plt.subplot(2, 2, 2)
plt.title('Voters not Voting (%)')
plt.xlabel('Year')
plt.bar(voter_turnout['year'], voter_turnout['voterTurnout'].apply(lambda x: 100- x), color='red')

plt.tight_layout()
plt.show()

cols = ['year', 'awamiLeaguePercentage', 'bangladeshNationalPartyPercentage', 'jatiyaPartyPercentage']
percentages = df[cols].groupby(['year']).mean().reset_index()

plt.plot(percentages['year'], percentages['awamiLeaguePercentage'])
plt.plot(percentages['year'], percentages['bangladeshNationalPartyPercentage'])
plt.plot(percentages['year'], percentages['jatiyaPartyPercentage'])

plt.legend(['Awami League', 'Bangladesh National Party', 'Jatiya Party'], loc='upper left')
plt.xlabel('Year')
plt.ylabel('Average % of Votes across Seats')

plt.show()
seats_by_district = df[['year', 'district', 'constituency']] \
    .groupby(['year', 'district']) \
    .count() \
    .reset_index()
fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(20, 10), sharex=True, sharey=True)

seats_2008 = seats_by_district[seats_by_district['year'] == '2008']
merged = bd_districts.set_index('ADM2_EN').join(seats_by_district.set_index('district'))
ax.axis('off')
ax.set_title('Constituency in 2008')
merged.plot(column='constituency', cmap='Greens', linewidth=1.2, ax=ax, edgecolor='0.8', legend=True)

turnout = df[['year', 'district', 'voterTurnout']].dropna().groupby(['year', 'district']).mean().reset_index()
turnout_2008 = turnout[turnout['year'] == '2008']
merged = bd_districts.set_index('ADM2_EN').join(turnout_2008.set_index('district'))
ax2.axis('off')
ax2.set_title('Voter Turnout in 2008 (%)')
merged.plot(column='voterTurnout', cmap='YlOrBr', linewidth=1.2, ax=ax2, edgecolor='0.8', legend=True)