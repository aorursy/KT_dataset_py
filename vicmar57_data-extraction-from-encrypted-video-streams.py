%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input/d_e_f_e_v_s/D_E_f_E_V_S"]).decode("utf8"))
import pandas as pd



path = r'../input//d_e_f_e_v_s/D_E_f_E_V_S/walking' 

df = pd.read_csv(path + "/1/1 vic.csv")

df.head()
df.info()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#to read multiple CSVs automatically

path = r'../input/d_e_f_e_v_s/D_E_f_E_V_S/frame_test.csv' 



labels_dfs = pd.read_csv(path)

plt.figure(figsize = (15,6))

legendTitles = []



labels_dfs.drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)



df = labels_dfs

df['key'] = np.array(df['Time']).astype(int)

df = df.groupby('key',axis=0).sum()



df['bytes'] = df['Length'].cumsum()  

avg_bps = "{:.1f}".format(df['Length'].sum() / (len(df['Length'])*1000))

print("average bitrate:" ,avg_bps)

df['rollLen'] = df['Length'].rolling(20).mean() #in bytes

df['rollLen'] = df['rollLen']/1000 #KBPS



plt.plot(range(len(df['rollLen'])), df['rollLen'])

title = "Bitrate (KBPS) vs time - average: " + avg_bps + " KB/S"

plt.title(title, fontsize=18)

plt.xlabel('Time (sec)', fontsize=18); plt.ylabel('KB', fontsize=16)

plt.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#to read multiple CSVs automatically

path = r'../input/d_e_f_e_v_s/D_E_f_E_V_S/motion increase test.csv' 





labels_dfs = pd.read_csv(path)

plt.figure(figsize=(15,6))

#plt.ylim((0,250))

legendTitles = []



labels_dfs.drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)



df = labels_dfs

df['key'] = np.array(df['Time']).astype(int)

df = df.groupby('key',axis=0).sum()



df['bytes'] = df['Length'].cumsum()  

avg_bps = "{:.1f}".format(df['Length'].sum() / (len(df['Length'])*1000))

print("average bitrate:" ,avg_bps)

df['rollLen'] = df['Length'].rolling(20).mean() #in bytes

df['rollLen'] = df['rollLen']/1000 #KBPS

plt.plot(range(len(df['rollLen'])), df['rollLen'])

title = "Bitrate (KBPS) vs time - average: " + avg_bps + " KB/S"

plt.title(title, fontsize=18)

plt.xlabel('Time (sec)', fontsize=18); plt.ylabel('KB', fontsize=16)

    

plt.show()
num_dfs_tot = 0

sec_to_cut = 15



#to read multiple CSVs automatically

import glob

path = r'../input/d_e_f_e_v_s/D_E_f_E_V_S/walking' 

allFolderPaths = glob.glob(path + "/*")

time_sliced = pd.DataFrame()

num_labels = len(allFolderPaths)

labels_dfs = [0]*num_labels



#read all CSVs for label i, and cut 15 sec of start and end of each video.

for i in range(num_labels) :

    labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]

    num_dfs_tot += len(labels_dfs[i])

    

    for j in range(len(labels_dfs[i])) :

        labels_dfs[i][j].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)

        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > sec_to_cut]

        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - sec_to_cut]

        labels_dfs[i][j]['Time'] -= sec_to_cut

        labels_dfs[i][j]['Label'] = i

        labels_dfs[i][j]['location'] = j

        

labels_dfs[2][1][:10]
labels_dfs[2][1].info()
import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from math import ceil

import timeit



start = timeit.default_timer() #to measure runtime

time_slice = 2; #in seconds

time_sliced = pd.DataFrame()



#read all CSVs for label i, and cut 15 sec of start and end of each video.

for i in range(num_labels) :

    for j in range(len(labels_dfs[i])) :

        

        df = labels_dfs[i][j] 

        df['TimeSlice'] = (np.array(labels_dfs[i][j]['Time'])/time_slice).astype(int)

        df = df.groupby('TimeSlice',axis=0 ,sort = 'False').agg(

        {'Time'     : ['count', 'std', 'mean'],

         'Length'   : ['mean', 'sum', 'std'],

         'Label'    : 'first',

         'location'    : 'first'}).fillna(0) 

        time_sliced = time_sliced.append(df)



time_sliced.columns = ["_".join(x) for x in df.columns.ravel()]



time_sliced.rename(columns={'Time_count':'pack_count',

                          'Label_first':'Label',

                          'location_first':'location'}, inplace=True)

time_sliced.head()
time_sliced['Label'].value_counts(normalize=True)
time_sliced.info()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from statistics import mean 



num_labels = 5;

#labels_dfs = [0]*num_labels

fileNames = [0]*num_labels



LAB = 0

TSUR = 1

VIC = 2



categories_or_singles = "cat"



#to read multiple CSVs automatically

import glob

path = r'../input/d_e_f_e_v_s/D_E_f_E_V_S/walking' 

allFolderPaths = glob.glob(path + "/*")



plt.figure(figsize=(20, 13))

legendTitles = []

ax = plt.gca() 

#max_spec_bitrate = [512]*1200

max_observed_bitrate = [458]*1200

avgs = [0]*num_labels



for i in range(num_labels) :

    #labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]

    fileNames[i] = [f for f in glob.glob(allFolderPaths[i] + "/*.csv")]

    plt.xlim((0,1200))

    color = next(ax._get_lines.prop_cycler)['color']

    avgs[i] = []



    for j in range(len(labels_dfs[i])) :

        #labels_dfs[i][j].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)

        #labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > 15]

        #labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - 15]

        #labels_dfs[i][j]['Time'] -= 15

        #labels_dfs[i][j]['Label'] = i

        #labels_dfs[i][j]['location'] = j

        

        short_fileName = fileNames[i][j].replace('../input/d_e_f_e_v_s/D_E_f_E_V_S/walking/', '')

        short_fileName = short_fileName.replace('.csv', '')

        legendTitles.append(short_fileName) #'vid ' + str(i) + '.' + str(j+1))

        df = labels_dfs[i][j]

        df['key'] = np.array(df['Time']).astype(int)

        df = df.groupby('key',axis=0).sum()

        

        df['bytes'] = df['Length'].cumsum()  

        df['BPS'] = df['bytes']/df['Time']

        bps = float("{:.1f}".format(df['Length'].sum() / (len(df['Length'])*1000)))

        avgs[i].append(bps)

        

        df['rollLen'] = df['Length'].rolling(20).mean()

        df['rollLen'] = df['rollLen']/1000

        

        if(categories_or_singles == "cat"):

            if j == VIC: #outlier room

                plt.plot(range(len(df['Length'])), df['rollLen'] , linestyle = '-', color = color, linewidth=4.0)

            else:    

                plt.plot(range(len(df['Length'])), df['rollLen'] , color = color)

        else:

            if j == VIC:

                plt.plot(range(len(df['Length'])), df['rollLen'], linestyle = '-', linewidth=4.0)

            else:  

                plt.plot(range(len(df['Length'])), df['rollLen'])

#            plt.ylim((0,350))

        

        plt.xlabel('Time (sec)', fontsize=16); plt.ylabel('KB', fontsize=16)

        



    avgs[i] = int(mean(avgs[i]))

    avgLine = [avgs[i]]*1200

    legendTitles.append("avg, label = " + str(i))

    plt.plot(range(len(avgLine)), avgLine)

#legendTitles.append("max_spec_bitrate") #'vid ' + str(i) + '.' + str(j+1))    

#plt.plot(range(len(max_spec_bitrate)), max_spec_bitrate)

#legendTitles.append("max_observed_bitrate") #'vid ' + str(i) + '.' + str(j+1))    

#plt.plot(range(len(max_observed_bitrate)), max_observed_bitrate)

plt.title("Videos' bitrates", fontsize=18)

plt.legend(legendTitles)

plt.show()

import itertools

from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(true_class, prediction,

                          normalize=True,

                          title='Confusion matrix',

                          cmap=plt.cm.YlOrRd):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    cm = confusion_matrix(true_class, prediction)

    classes = (np.sort(np.unique(true_class)))

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



   # print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    #plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

    plt.show()

    

    return
train_data = []

test_data = []

    

train_data.append(time_sliced[time_sliced['location'] == LAB]) 

train_data.append(time_sliced[time_sliced['location'] == TSUR])

test_data.append(time_sliced[time_sliced['location'] == VIC])



train_data  = pd.concat(train_data, axis=0) #first 80% of data for label i

test_data   = pd.concat(test_data, axis=0)   #last 20% of data for label i

train_data.drop('location', axis=1, inplace=True)

test_data.drop('location', axis=1, inplace=True)



#train_data = train_data.sample(frac=1).reset_index(drop=True) #mix the data!!!

#test_data  = test_data.sample(frac=1).reset_index(drop=True) #mix the data!!!



train_labels = train_data['Label'] #get labels

test_labels = test_data['Label'] #get labels

train_data = train_data.drop('Label',axis=1) #remove labels from data

test_data = test_data.drop('Label',axis=1) #remove labels from data    

    



#fit classifier

clf = RandomForestClassifier(n_estimators = 100, max_depth= 4)

clf.fit(train_data ,train_labels)

pred = clf.predict(test_data)





features = list(train_data.columns.values)

importances = clf.feature_importances_

indices = np.argsort(importances)



plt.figure(1,figsize = (10, 6))

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()





plt.figure(2, figsize = (7, 7))

plot_confusion_matrix(test_labels, pred)



acc_score = "{:.2f}".format(accuracy_score(test_labels , pred))

mean_absolute_err = "{:.3f}".format(mean_absolute_error(test_labels , pred))

print('time_slice:', time_slice, 'sec')

print ('accuracy_score:',acc_score)

print ('mean_absolute_error:', mean_absolute_err)
features
# Extract single tree

estimator = clf.estimators_[5]



from sklearn.tree import export_graphviz

# Export as dot file

export_graphviz(estimator, out_file='tree.dot',

                feature_names = features,

                class_names = list(map(str, range(0,5))),

                rounded = True, proportion = True,leaves_parallel = True,label = False, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')
import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from math import ceil

import timeit



start = timeit.default_timer() #to measure runtime



time_slices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]; #in seconds

num_labels = 5;

'''

num_dfs_tot = 0

labels_dfs = [0]*num_labels

'''

sec_to_cut = 15

acc_scores = []

maes = []



#to read multiple CSVs automatically

import glob

path = r'../input/d_e_f_e_v_s/D_E_f_E_V_S/sitting' 

allFolderPaths = glob.glob(path + "/*")



'''

#read all CSVs for label i, and cut 15 sec of start and end of each video.

for i in range(num_labels) :

    labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]

    for j in range(len(labels_dfs[i])) :

        labels_dfs[i][j].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)

        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > sec_to_cut]

        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - sec_to_cut]

        labels_dfs[i][j]['Time'] -= sec_to_cut

        labels_dfs[i][j]['Label'] = i

        labels_dfs[i][j]['location'] = j

'''

        

for s in range(len(time_slices)) :

    time_sliced = pd.DataFrame()

    for i in range(num_labels) :

        for j in range(len(labels_dfs[i])) :

            df = labels_dfs[i][j] 

            df['TimeSlice'] = (np.array(labels_dfs[i][j]['Time'])/time_slices[s]).astype(int)

            df = df.groupby('TimeSlice',axis=0 ,sort = 'False').agg(

            {'Time'     : ['count', 'std', 'mean'],

             'Length'   : ['mean', 'sum', 'std'],

             'Label'    : 'first',

             'location'    : 'first'}).fillna(0)

            time_sliced = time_sliced.append(df)

    

    time_sliced.columns = ["_".join(x) for x in df.columns.ravel()]

    time_sliced.rename(columns={'Time_count':'pack_count',

                          'Label_first':'Label',

                          'location_first':'location'}, inplace=True)

    

    train_data = []; test_data = []

        

    train_data.append(time_sliced[time_sliced['location'] == LAB])  #lab

    train_data.append(time_sliced[time_sliced['location'] == TSUR])  #tsur

    test_data.append(time_sliced[time_sliced['location'] == VIC])   #vic

    

    train_data  = pd.concat(train_data, axis=0) 

    test_data   = pd.concat(test_data, axis=0)   

    train_data.drop('location', axis=1, inplace=True)

    test_data.drop('location', axis=1, inplace=True)

    

    train_labels = train_data['Label'] #get labels

    test_labels = test_data['Label'] #get labels

    train_data = train_data.drop('Label',axis=1) #remove labels from data

    test_data = test_data.drop('Label',axis=1) #remove labels from data    

        

    

    clf = RandomForestClassifier(n_estimators = 100)

    clf.fit(train_data ,train_labels)

    pred = clf.predict(test_data)

    

    acc_score = "{:.2f}".format(accuracy_score(test_labels , pred))

    mean_absolute_err = "{:.3f}".format(mean_absolute_error(test_labels , pred))

    acc_scores.append(int(float(acc_score)*100))

    maes.append(float(float(mean_absolute_err)*100))

    #print ('slice size:', time_slices[s], 's,', 'acc_score:',int(float(acc_score)*100), '%, mean_abs_err:', mean_absolute_err, '\n')



stop = timeit.default_timer() #to measure runtime

print('RunTime:', "{:.1f}".format(stop - start), 'sec.')





plt.figure(3,figsize=(15,6))

plt.title('Mean abs. error(time slice size)',fontsize=16)

plt.xlabel('time slice size'); plt.ylabel('Mean abs. error * 100')

plt.xticks(np.arange(0,20), np.arange(0,20).astype(int))

plt.ylim([0,100])

plt.grid()

plt.scatter(time_slices,maes)

#plt.plot(time_slices,maes)



plt.figure(4,figsize=(15,6))

plt.title('accuracy score(time slice size)',fontsize=16)

plt.xlabel('time slice size'); plt.ylabel('accuracy score (%)')

plt.xticks(np.arange(0,20), np.arange(0,20).astype(int))

plt.grid()

plt.ylim([0,100])

plt.scatter(time_slices,acc_scores)

num_dfs_tot
import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from math import ceil

import timeit



time_slice = 10; #in seconds

num_labels = 5;

'''

num_dfs_tot = 0

labels_dfs = [0]*num_labels

'''



# Find the Time diff that shows when a frame ends



combined = pd.DataFrame()



seconds_in_DFs_list = [0]*num_dfs_tot #in a every single video dataframe

time_slice_list     = [0]*num_dfs_tot #initiallization

indexes_to_split_frame_by_list = [0]*num_dfs_tot #initiallization

time_sliced_df_list = []





start = timeit.default_timer() #to measure runtime



for k in range(num_labels) : #slice data frames by time slice seconds

    for i in list(range(len(labels_dfs[k]))) : #slice data frames by time slice seconds

        #"lose" non essential data

       #labels_dfs[k][i].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)

       #labels_dfs[k][i]['Time']  = labels_dfs[k][i]['Time'].astype('float') #change Time column data to int type 

       combined = combined.append(labels_dfs[k][i])

       

       seconds_in_DFs_list[i] = int(ceil(labels_dfs[k][i].at[len(labels_dfs[k][i])-1,'Time'])) #number of seconds in current df

       time_slice_list[i] = list(range(0,seconds_in_DFs_list[i],time_slice)) #list of times to slice by

       indexes_to_split_frame_by_list[i] = list(range(0,seconds_in_DFs_list[i],time_slice)) #initialization

       

       for j in list(range(len(time_slice_list[i]))): #calc indexes to slice by

           #find first index of labels_dfs[k][i]['Time'] the exceeds time_slice_list[i][j]

           indexes_to_split_frame_by_list[i][j] = np.argmax(labels_dfs[k][i]['Time'] > time_slice_list[i][j]) -1 #TODO: moved below.

                

       df_slice = pd.DataFrame() 

       

       for j in list(range(len(time_slice_list[i]))): #TODO: change for efficiency.

           #print(len(time_slice_list[i]))

           #indexes_to_split_frame_by_list[i][j] = np.argmax(labels_dfs[k][i]['Time'] > time_slice_list[i][j]) -1 TODO: doesnt work because we need j+1.

           if (j != len(time_slice_list[i])-1 and j != 0):  #not first or last index

               ind_start = indexes_to_split_frame_by_list[i][j] + 1

               ind_end = indexes_to_split_frame_by_list[i][j+1] + 1

           elif j == len(time_slice_list[i])-1:             #last index

               ind_start = indexes_to_split_frame_by_list[i][j] + 1

               ind_end = len(labels_dfs[k][i])

           else:                                            # j==0, first index

               ind_start = 0

               ind_end = indexes_to_split_frame_by_list[i][j+1]+1

           

           df_slice = labels_dfs[k][i][ind_start:ind_end]

           df_slice.loc[:,'Time'] = df_slice['Time'].diff().fillna(0) #calc time diffs, not time since begining of recording packets.

           time_sliced_df_list.append(df_slice)

         



combined.to_csv('combined.csv',index=False)





timeDiffCol = labels_dfs[1][1]['Time']

import matplotlib.pyplot as plt

plt.figure(1,figsize=(15,6))

plt.hist(timeDiffCol, bins = 30)

#plt.xlim(0,0.1)

plt.xlabel('TimeDiff', fontsize=16); plt.ylabel('Count', fontsize=16)

plt.title("Histogram of time diffs between two packets", fontsize = 20)



#%% Distinguish between the frames



labels_dfs[1][1]['frameNum'] = 0

j=0

timeDiffCol.reset_index(drop = True , inplace = True)

labels_dfs[1][1].reset_index(drop = True ,inplace = True)



import time

t0 = time.time()

for idx,row in labels_dfs[1][1].iterrows():

    if(timeDiffCol[idx] < 0.025) : 

        labels_dfs[1][1].at[idx,'frameNum'] = j

    else:

        j += 1

        labels_dfs[1][1].at[idx,'frameNum'] = j

print ('time:  ', time.time()-t0)





united = labels_dfs[1][1].groupby('frameNum',axis=0).sum()

united['Time'] = np.cumsum(united['Time'])



plt.figure(2, figsize=(15,6))

plt.title("Frame size(time)", fontsize = 20)

plt.xlabel('Time (sec)', fontsize=16); plt.ylabel('Frame size', fontsize=16)

plt.xlim(0,20)

plt.plot(united['Time'],united['Length'])



import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from math import ceil

import timeit

from skimage import filters

import matplotlib.pyplot as plt



thresholds = [0]*num_labels

combined_DF= pd.DataFrame()



#%%

timeDiffCol = pd.Series()

n=0



for i in range(num_labels) :

    thresholds[i] = [0]*len(labels_dfs[i])

    

    for j in range(len(labels_dfs[i])) :

        thresholds[i][j] = filters.threshold_otsu(labels_dfs[i][j]['Time'],nbins=30)

        df_size = len(labels_dfs[i][j])

        

        for idx,row in labels_dfs[i][j].iterrows():



            if(labels_dfs[i][j].at[idx,'Time'] < thresholds[i][j]) : 

                labels_dfs[i][j].at[idx,'frameNum'] = n

            else:

                n += 1

                labels_dfs[i][j].at[idx,'frameNum'] = n

                labels_dfs[i][j].at[idx,'Time diff pre'] = labels_dfs[i][j].at[idx,'Time']

        

        combined_DF = pd.concat([combined_DF , labels_dfs[i][j]])



united = combined_DF.groupby('frameNum' ,sort = 'False').agg({'Time': ['std' ,'mean'],

                                                                  'Length' : ['std' , 'mean' , 'sum'],

                                                                  'Label' : 'first',

                                                                  'Time diff pre':'first'}).fillna(0)

united.columns = ["".join(x) for x in united.columns.ravel()]



united.head(15)


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from statistics import mean 



num_labels = len(allFolderPaths);

labels_dfs = [0]*num_labels

fileNames = [0]*num_labels



categories_or_singles = "cat"



#to read multiple CSVs automatically

import glob

path = r'../input/d_e_f_e_v_s/D_E_f_E_V_S/sitting' 

allFolderPaths = glob.glob(path + "/*")



plt.figure(figsize=(20, 13))

legendTitles = []

ax = plt.gca() 

#max_spec_bitrate = [512]*1200

max_observed_bitrate = [458]*600

avgs = [0]*num_labels



for i in range(num_labels) :

    labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]

    fileNames[i] = [f for f in glob.glob(allFolderPaths[i] + "/*.csv")]

    plt.xlim((0,600))

    color = next(ax._get_lines.prop_cycler)['color']

    avgs[i] = []



#    legendTitles = []



    for j in range(len(labels_dfs[i])) :

        labels_dfs[i][j].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)

        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > 15]

        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - 15]

        labels_dfs[i][j]['Time'] -= 15

        labels_dfs[i][j]['Label'] = i

        labels_dfs[i][j]['location'] = j

        #print(str(i))

        short_fileName = fileNames[i][j].replace('../input/d_e_f_e_v_s/D_E_f_E_V_S/sitting/', '')

        

        short_fileName = short_fileName.replace('.csv', '')

        legendTitles.append(short_fileName) #'vid ' + str(i) + '.' + str(j+1))

        df = labels_dfs[i][j]

        df['key'] = np.array(df['Time']).astype(int)

        df = df.groupby('key',axis=0).sum()

        

        #VR_funcs.PlotXYZ(df['Time'], df['Length'], [1]*len(df['Time']))

        df['bytes'] = df['Length'].cumsum()  

        #df['timeSUM'] = df['Time'].cumsum()  

        df['BPS'] = df['bytes']/df['Time']

        bps = float("{:.1f}".format(df['Length'].sum() / (len(df['Length'])*1000)))

#        df['Length'].sum() / df['Time'].sum()

#        location = df['location'][0]

        avgs[i].append(bps)

        

        df['rollLen'] = df['Length'].rolling(20).mean()

        df['rollLen'] = df['rollLen']/1000

        

        if(categories_or_singles == "cat"):

            if j == 2:

                plt.plot(range(len(df['Length'])), df['rollLen'] , linestyle = '-', color = color, linewidth=4.0)

            else:    

                plt.plot(range(len(df['Length'])), df['rollLen'] , color = color)

        else:

            if j == 2:

                plt.plot(range(len(df['Length'])), df['rollLen'], linestyle = '-', linewidth=4.0)

            else:  

                plt.plot(range(len(df['Length'])), df['rollLen'])

#            plt.ylim((0,350))

        

        #legend = plt.legend(handles=[one, two, three], title="title", loc=4, fontsize='small', fancybox=True)

        plt.xlabel('Time (sec)', fontsize=16); plt.ylabel('KB', fontsize=16)

#        break

        



    avgs[i] = int(mean(avgs[i]))

    avgLine = [avgs[i]]*600

    legendTitles.append("avg, label = " + str(i))

    plt.plot(range(len(avgLine)), avgLine)

#legendTitles.append("max_spec_bitrate") #'vid ' + str(i) + '.' + str(j+1))    

#plt.plot(range(len(max_spec_bitrate)), max_spec_bitrate)

#legendTitles.append("max_observed_bitrate") #'vid ' + str(i) + '.' + str(j+1))    

#plt.plot(range(len(max_observed_bitrate)), max_observed_bitrate)

plt.title("Videos' bitrates", fontsize=18)

plt.legend(legendTitles)

plt.show()





start = timeit.default_timer() #to measure runtime



time_slices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]; #in seconds

acc_scores = []

maes = []

        

for s in range(len(time_slices)) :

    time_sliced = pd.DataFrame()

    for i in range(num_labels) :

        for j in range(len(labels_dfs[i])) :

            df = labels_dfs[i][j] 

            df['TimeSlice'] = (np.array(labels_dfs[i][j]['Time'])/time_slices[s]).astype(int)

            df = df.groupby('TimeSlice',axis=0 ,sort = 'False').agg(

            {'Time'     : ['count', 'std', 'mean'],

             'Length'   : ['mean', 'sum', 'std'],

             'Label'    : 'first',

             'location'    : 'first'}).fillna(0) 

            time_sliced = time_sliced.append(df)

    

    time_sliced.columns = ["_".join(x) for x in df.columns.ravel()]

    

    

    train_data = []; test_data = []

        

    train_data.append(time_sliced[time_sliced['location_first'] == LAB])  #lab

    train_data.append(time_sliced[time_sliced['location_first'] == TSUR])  #tsur

    test_data.append(time_sliced[time_sliced['location_first'] == VIC])   #vic

    

    train_data  = pd.concat(train_data, axis=0) 

    test_data   = pd.concat(test_data, axis=0)   

    train_data.drop('location_first', axis=1, inplace=True)

    test_data.drop('location_first', axis=1, inplace=True)

    

    train_labels = train_data['Label_first'] #get labels

    test_labels = test_data['Label_first'] #get labels

    train_data = train_data.drop('Label_first',axis=1) #remove labels from data

    test_data = test_data.drop('Label_first',axis=1) #remove labels from data    

        

    

    clf = RandomForestClassifier(n_estimators = 100)

    clf.fit(train_data ,train_labels)

    pred = clf.predict(test_data)

    

    acc_score = "{:.2f}".format(accuracy_score(test_labels , pred))

    mean_absolute_err = "{:.3f}".format(mean_absolute_error(test_labels , pred))

    acc_scores.append(int(float(acc_score)*100))

    maes.append(float(float(mean_absolute_err)*100))

    print('slice size:', time_slices[s], 'sec.')

    #print ('acc_score:',int(float(acc_score)*100), '%, mean_abs_err:', mean_absolute_err, '\n')

    



stop = timeit.default_timer() #to measure runtime

print('RunTime:', "{:.1f}".format(stop - start), 'sec.')





plt.figure(3,figsize=(15,6))

plt.title('Mean abs. error(time slice size)',fontsize=16)

plt.xlabel('time slice size'); plt.ylabel('Mean abs. error * 100')

plt.xticks(np.arange(0,20), np.arange(0,20).astype(int))

plt.ylim([0,200])

plt.grid()

plt.scatter(time_slices,maes)

#plt.plot(time_slices,maes)



plt.figure(4,figsize=(15,6))

plt.title('accuracy score(time slice size)',fontsize=16)

plt.xlabel('time slice size'); plt.ylabel('accuracy score (%)')

plt.xticks(np.arange(0,20), np.arange(0,20).astype(int))

plt.grid()

plt.ylim([0,100])

plt.scatter(time_slices,acc_scores)

#plt.plot(time_slices,acc_scores)
