import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics

## Purpose of this chart is to get an understanding of how predicted probabilities translate to actual wins
## As well as an understanding of the prediction distribution across percentages
## I've also added calcs for accuracy, auc & maximum miss

## NOTE - in order to run this need to have a completed model, then create a dataframe 
## that contains the predicted chance of winning & actual outcome for each game in the test set

## Create dataframe with predicted win % and actual result for each game from predicted test set
## I've used random numbers as a dummy, and included my actual results in the comments

predTestComp = pd.DataFrame({'pred' : np.random.uniform(0, 1, 20000),
                             'Win': np.random.randint(0, 2, 20000)})

## Round prediction % for ease of interpretation

predTestComp['predRound'] = np.round(predTestComp['pred'], decimals=2) 

## Find Difference in predictions vs actuals, as well as max miss & print results
## Also find number of "correct" predictions - above 50% and win / below 50% and loss
## Then print the results
    
predTestComp['diff'] = abs(predTestComp['Win'] - predTestComp['pred'])
maximum = np.round(max(predTestComp['diff']),decimals = 4)
def f(df3):
    if df3['diff'] < 0.5:
        val = 1
    else:
        val = 0
    return val

predTestComp['bin'] = predTestComp.apply(f, axis=1)
accuracy = np.round(sum(predTestComp['bin']) / len(predTestComp),decimals = 4)
print('The Biggest Upset Was', maximum)
print('Accuracy was', accuracy)
    
## Now Calc AUC
    
y = np.array(predTestComp['Win']+1)
pred = np.array(predTestComp['pred'])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
auc = np.round(metrics.auc(fpr, tpr),decimals=4)
print('AUC was', auc)
predTestComp.drop(['Type'],inplace=True,axis=1,errors='ignore')
predTestComp.drop(['id'],inplace=True,axis=1,errors='ignore')

## Group predictions by predicted win percentage, then find the total number of wins, games & average win percentage 
## For each rounded prediction value (0-100%)

grouped = predTestComp.groupby(['predRound'])['Win'].agg(['sum', 'count', 'mean']).reset_index()

## Create subplots that have win% on the x axis, and number of games on the secondary axis

fig, tsax = plt.subplots(figsize=(12,5))
barax = tsax.twinx()

## Create bar chart based on the count of games for each predicted percentage

barax.bar(grouped.index, grouped['count'], facecolor=(0.5, 0.5, 0.5), alpha=0.3) 

## Create line chart that shows the average win percentage by predicted percentage

fig.tight_layout()
tsax.plot(grouped.index, grouped['mean'], color = 'b')

## Set axis & data point labels as well as tick distribution

barax.set_ylabel('Number of Games')
barax.xaxis.tick_top()
tsax.set_ylabel('Win %')
tsax.set_xlabel('Predicted Win %')
tsax.set_xlim([0, 101])
tsax.set_ylim([0, 1])
plt.xticks(np.arange(0, 101, 10))
percListX = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
percListY = ['0%', '20%', '40%', '60%', '80%', '100%']
tsax.set_xticklabels(percListX)
#tsax.set_yticklabels(percListY)

## Put line graph in front of bar chart

tsax.set_zorder(barax.get_zorder()+1) 
tsax.patch.set_visible(False) # hide the 'canvas' 

## Create legend labels - necessary because it's a subplot

line_patch = mpatches.Patch(color='blue', label='Percentage of Games Won')
bar_patch = mpatches.Patch(color='gray', label='Number of Games')
plt.legend(handles=[line_patch, bar_patch], loc = 'upper center')
