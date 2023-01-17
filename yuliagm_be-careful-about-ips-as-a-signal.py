import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import gc
%matplotlib inline
input_path = '../input/'
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }

train = pd.read_csv(input_path+'train_sample.csv', dtype=dtypes)
train.head()
#convert to date/time
train['click_time'] = pd.to_datetime(train['click_time'])
train['attributed_time'] = pd.to_datetime(train['attributed_time'])

#extract hour as a feature
train['click_hour']=train['click_time'].dt.hour
def plotStrip(x, y, hue, figsize = (14, 9)):
    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y, \
             hue = hue, jitter = 0.4, marker = '.', \
             size = 4, palette = colours)
        ax.set_xlabel('')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['col1', 'col2'], bbox_to_anchor=(1, 1), \
               loc=2, borderaxespad=0, fontsize = 16);
    return ax
X = train
Y = X['is_attributed']
ax = plotStrip(X.click_hour, X.ip, Y)
ax.set_ylabel('ip', size = 16)
ax.set_title('IP (vertical), by HOUR(horizontal), split by converted or not(color)', size = 20);
del train
gc.collect()
total_rows = 18790470
sample_size = total_rows//120

def get_skiprows(total_rows, sample_size):
    inc = total_rows // sample_size
    return [row for row in range(1, total_rows) if row % inc != 0]

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        }

test = pd.read_csv(input_path+'test.csv',
                 skiprows=get_skiprows(total_rows,sample_size), dtype=dtypes)
test.head()
#convert to date/time
test['click_time'] = pd.to_datetime(test['click_time'])

#extract hour as a feature
test['click_hour']=test['click_time'].dt.hour
#dummy variable for hour color bands in test
test['band'] = np.where(test['click_hour']<=6, 0, \
                        np.where(test['click_hour']<=11, 1, \
                                np.where(test['click_hour']<=15, 2, 3)))
print(len(test))
X = test
Y = test['band']
ax = plotStrip(X.click_hour, X.ip, Y)
ax.set_ylabel('ip', size = 16)
ax.set_title('IP (vertical), by HOUR(horizontal), split by hour band', size = 20);
