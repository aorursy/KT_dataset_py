import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib notebook

data = pd.read_csv('../input/closing_odds.csv.gz', compression='gzip', sep=',', quotechar='"')
print('Reading completed. Total rows {}'.format(len(data)))
#data.info()

#round((1/ data[['max_odds_home_win',]]).sum(axis=1).sort_values(),1).value_counts().sort_index()

# CONSENCUS PROBABILITY

# 1. MINIMUM CHANCE

data['min_ch_home'] = 1 / data.max_odds_home_win
data['min_ch_draw'] = 1 / data.max_odds_draw
data['min_ch_away'] = 1 / data.max_odds_away_win

#sum
data['sum_min_chances'] = data.min_ch_home + data.min_ch_draw + data.min_ch_away


# 2. DETECT FORKS

def fork(row):
    if min(row['n_odds_home_win'],row['n_odds_draw'],row['n_odds_away_win']) < 5:
        return 0
    elif row['sum_min_chances'] > 1:
        return 0
    else:
        return 1

data['fork'] = data.apply(fork,axis = 1)
print('Forks detection complete')

# 3. CONCENSUS PROBABILITY
#actually, we don't need this probability that much. but it will help us to make better split between wins bets
#data['p_cons_home'] = 1 / data.avg_odds_home_win
#data['p_cons_draw'] = 1 / data.avg_odds_draw
#data['p_cons_away'] = 1 / data.avg_odds_away_win
#data['p_cons_sum'] = data.p_cons_home + data.p_cons_draw + data.p_cons_away

# YOUR BET

data['bet_home'] = 100 * data['min_ch_home'] / data['sum_min_chances']
data['bet_draw'] = 100 * data['min_ch_draw'] / data['sum_min_chances']
data['bet_away'] = 100 * data['min_ch_away'] / data['sum_min_chances']
data['sum_bet'] = data.bet_home + data.bet_draw + data.bet_away
#WINNER DETECTION

def winner(row):
    if row['home_score'] > row['away_score']: return 1
    elif row['home_score'] == row['away_score']: return 2
    elif row['home_score'] < row['away_score']: return 3
    else: return -1
    
data['REAL_WINNER'] = data.apply(winner, axis = 1) # 1 -home, 2 - draw, 3 - away
#forks_df = data[data.fork == 1].copy()

#forks_df = forks_df[forks_df.league.str.contains('Ukraine')]

def profit(row):
    if row['fork'] == 0: return 0
    if   row['REAL_WINNER'] == 1: return row['bet_home'] * (row['max_odds_home_win']) - 100
    elif row['REAL_WINNER'] == 2: return row['bet_draw'] * (row['max_odds_draw']) - 100
    elif row['REAL_WINNER'] == 3: return row['bet_away'] * (row['max_odds_away_win']) - 100
    else: return '-1'
    
# costs are always 100$    
data['PROFIT'] = data.apply(profit,axis = 1)

#print('Total Profit: {:.0f}\nTotal Costs: {}'.format(data.PROFIT.sum(), len(data)*100))
print('Total Profit: {}'.format(data.PROFIT.sum()))

%matplotlib notebook


# amount of evernts with forks and without forks
# based on probability sum
fig3, ax3 = plt.subplots()


plot_ser = data.sum_min_chances.round(2).value_counts().sort_index().cumsum()
plot_ser = plot_ser / plot_ser.max() * 100
#fig1 = plt.figure()
plot_ser[plot_ser.index <= 1].plot(color = 'green', label = 'you win') # we win
plot_ser[plot_ser.index >= 1].plot(color = 'red', label = 'you loose') # we lose
plt.xlim(.8,1.2)
plt.ylim(0,)
plt.xlabel('Probability Sum')
plt.ylabel('% of total bets')

plt.axvline(x=1, color = 'black', ls = ':', alpha = 0.3)

win_line = plot_ser[plot_ser.index <= 1].max().round(2)
plt.axhline(y=win_line, color = 'black', ls = ':', alpha = 0.3)

plt.text(1.2, 20, ' {:.0f}% of bets with folks'.format(win_line), fontsize=10,horizontalalignment='right', fontstyle = 'italic')
plt.title('Cumulative distribution, %')

#plt.axhspan(0.25, 70, facecolor='0.5', alpha=0.5)
#plt.axvspan(1.25, 50, facecolor='#2ca02c', alpha=0.5)

plt.legend()
#%matplotlib notebook
fig2 = plt.figure()
data[data.fork == 1].PROFIT.round(0).map(int).value_counts().sort_index().plot.bar(color = 'r', alpha = 0.5, width = 1)

plt.xlim(0.5,10.5)
plt.ylabel('Amount of events')
plt.xlabel('Fork ROI, %')
plt.title('Distribution of forks by ROI, rounded to int')
# calculate amount of bets for qurtiles

cum_df = data[data.PROFIT > 0.01].PROFIT.sort_values().to_frame()
cum_df.reset_index(drop = True, inplace = True)
cum_df['PROFIT_CUM'] = cum_df['PROFIT'].cumsum().map(lambda x: round(x, 2))
cum_df['ROI'] = cum_df['PROFIT'].map(lambda x: round(x,2))
cum_df['PROCENTILE'] = cum_df['PROFIT_CUM'] / cum_df['PROFIT_CUM'].max() * 100

perc_dict = dict()
for perc in (25, 50 , 75, 100):
    temp_df = cum_df[cum_df.PROCENTILE >= perc]
    i = temp_df.head(1).index[0]
    max_roi = cum_df.loc[i,'ROI']
    bets = len(temp_df)
    profit = cum_df.loc[i,'PROFIT_CUM']
    print('ROI: {}, bets: {}. Profit: {}'.format(max_roi, bets, profit))
    perc_dict[perc] = (max_roi, profit)
    #print(max_roi)
    #print(i, max_roi)
#cum_df

cum_df.head(3)
#fig4, ax = plt.subplots(111)
#plt.gca()
#fig4 = plt.figure()
#.loc[[2.51, 6.46, 35.38, 1458.58]]

import math
#math.log(100, 10)
fig4, ax4 = plt.subplots()
plot_df = cum_df.copy()

plot_df['ROI'] = plot_df.ROI.map(lambda x: math.log(x, 10))
plot_df = plot_df.drop_duplicates('ROI', keep = 'last').set_index('ROI')
plot_df.index.name = 'ROI, %'
#fig4 = plt.figure()
plot_df.PROFIT_CUM.plot(ax = ax4, color = 'green', alpha = 1)

ax4.set_xticklabels([str(10**x)+'%' for x in ax4.get_xticks()])
ax4.set_ylabel("Profit, '000 $")
ax4.set_ylim(0,)
ax4.set_yticklabels([x // 1000 for x in ax4.get_yticks()])

for p, (roi, pr) in perc_dict.items():
    log_x = math.log(roi, 10)
    plt.axvline(x= log_x, color = 'black', alpha = 0.1, ls = '--')
    plt.axhline(y= pr, color = 'black', alpha = 0.1, ls = '--')
#plt.axvline(x=math.log(6.46,10), color = 'black', alpha = 0.3)

    plt.text(log_x, pr, '{}P. Roi: {:.1f}%. Profit: {:.0f}k $'.format(p, roi, pr/1000), fontsize=10,horizontalalignment='right', fontstyle = 'italic')

#ax4.set_ylim(1000,200000)
#ROI: 2.51, bets: 17507. Profit: 49716.77
#№ROI: 6.46, bets: 4491. Profit: 99432.21
#№ROI: 35.38, bets: 618. Profit: 149174.86
#ROI: 1458.58, bets: 1. Profit: 198858.41
            
plt.title('Cumulative Profit vs ROI on logarithm scale')    

#cum_df.ROI.map
    
#cum_df.loc[[2.51, 6.46, 35.38, 1458.58]]
#cum_df
