held_out = True # whether to use held out data
%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as  np

import tests
import os

def setWorkingDir():
    os.chdir("/kaggle/working/")

def setUtilDir():
    setWorkingDir()
    os.chdir("../input/analysisutils/")
    
def setDataDir():
    setWorkingDir()
    os.chdir("../input/cryptocurrency-market-dynamics/")

setUtilDir()    
import analysis_utils as utils
import importlib
importlib.reload(tests)
importlib.reload(utils)
setDataDir()
checks, interventions = utils.read_data(held_out)
checks = utils.expand_columns(checks, interventions)
daily_volume = utils.get_daily_volume(checks)
coin_data = utils.get_coin_data(checks)
print(checks.columns)
print(interventions.columns)
len(checks)
tests.print_descriptives(checks, daily_volume)
sns.set(context = 'paper', font_scale = 3, font='serif', style = 'white',  rc={"lines.linewidth": 2.5})
plt.plot(range(len(daily_volume)), daily_volume)
plt.xlabel('Day of Experiment')
plt.ylabel('Observed BTC Vol.')

setWorkingDir()
plt.savefig('btc-vol-over-time.jpg', bbox_inches = 'tight')
sns.set(context = 'paper', font_scale = 3, font='serif', style = 'white')
plt.hist(np.log(coin_data['tot']))
plt.xlabel('Coin log(Ave. Hourly BTC Vol.)')
plt.ylabel('Frequency')
plt.savefig('btc-vol-coin-hist.jpg', bbox_inches = 'tight')
sns.set(context = 'paper', font_scale = 3, font='serif', style = 'white')
plt.hist(np.log(coin_data['size']))
plt.xlabel('Coin log(Ave. BTC Trade Size)')
plt.ylabel('Frequency')
plt.savefig('trade-size-coin-hist.jpg', bbox_inches = 'tight')
sns.set(context = 'paper', font_scale = 3, font='serif', style = 'white')
plt.hist(coin_data['counts'], 20)
plt.xlabel('Estimated Average Number\n of Trades Per Hour (Per Coin)')
plt.ylabel('Frequency')
plt.savefig('num-trades-coin-hist.jpg', bbox_inches = 'tight')
tests.run_ttests(checks)
#results = tests.get_bootstrap_results(checks)

titles = {}
titles['buy'] = {'trade':'Last Observed Trade is a Buy',
                 'perc':'% Buy-Side Volume',
                 'null':'Some Trade Occurs'}
titles['sell'] = {'trade':'Last Observed Trade is a Sell',
                 'perc':'% Sell-Side Volume',
                 'null':'Some Trade Occurs'}

"""
for d in results:
    for a in ['buy','sell']:
        tests.violin(results[d][a], 
                     {1:'15 Minutes',2:'30 Minutes',3:'60 Minutes'}, 
                     'Time Since Intervention',
                     '(Treat. Prob.) - (Control Prob.)', 
                     title = 'Dependent Variable:\n' + titles[a][d], 
                     filebase = d + '-' + a)
"""

tests.run_regressions(checks)
tests.get_total_effect_size(checks, interventions)
tests.get_state_fractions(interventions)
sns.set(context = 'paper', font_scale = 2.75, font='serif', style = 'white',  rc={"lines.linewidth": 4})
plt.plot(checks.loc[(checks['monitor_num'] == 0)].groupby('hour').mean()['total_60'])
plt.xlabel('Hour')
plt.ylabel('Mean Trading Volume')
plt.savefig('hourly-volume.jpg', bbox_inches = 'tight')
sns.set(context = 'paper', font_scale = 2.75, font='serif', style = 'white',  rc={"lines.linewidth": 4})
plt.plot(checks.loc[(checks['monitor_num'] == 0)].groupby('day_of_week').mean()['total_60'])
plt.xlabel('Day of Week')
plt.ylabel('Mean Trading Volume')
plt.savefig('by-day-volume.jpg', bbox_inches = 'tight')
tests.bot_check(checks)
