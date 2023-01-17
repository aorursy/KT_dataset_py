# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#../input/test.csv

#../input/train.csv



data_train = pd.read_csv('../input/train.csv')

data_train.head()
# Print out all the feature names

list(data_train.columns.values)
# cat1 - cat116 & cont1 - cont14 (and the "loss" column)



# check what kind of data for each category

print (data_train.cat1.head(1))

print (data_train.cat116.head(1))

print (data_train.cont1.head(1))

print (data_train.cont14.head(1))
# For testing syntax - delete this cell

x = "cat1"

for test in data_train[x]:

    pass

    #print (test)
# Check Disparity of Categories between cat1 - cat116

for x in range(1,117):

    disparity = []

    cat = "cat"

    proper_string = (str(cat)+str(x))

    for test in data_train[proper_string]:

        if test in disparity:

            pass

        else:

            disparity.append(test)

    print ("Len of List For: ",proper_string,"is ", len(disparity))        

            

        
# From prior cell, note that cat1-cat72 are binary, then disparity increases as we increment up the 

# list of features



# Now lets print cat116 again for good measure

print (data_train.cat116)



# Do we want to convert these letters to numbers and run some linear regressions?

# The categories do not seem to increment (i.e. the 27th catergory would be aa, then ab, then ac...etc)

# So the question becomes what is the importance of this lettering convention

# also, is it the same amongst the other columns of data?

# For now we will skip this puzzle, and move on to the numeric data
# Check Disparity of Categories between cont1 - cont14

for x in range(1,15):

    disparity = []

    cat = "cont"

    proper_string = (str(cat)+str(x))

    for test in data_train[proper_string]:

        if test in disparity:

            pass

        else:

            disparity.append(test)

    print ("Len of List For: ",proper_string,"is ", len(disparity))    

    

    # I would not have thought there would be so little disparity, considering this data is 

    # six digits. Might be a programming rounding error, need to dig deeper in next cell
# cont 3 had the least disparity, so lets examine it



print (data_train['cont2'])



#for x in data_train['cont2']:

    #print (x)

    

    #### Seems to pass the smell test 
# Now lets print a linear regression



import seaborn as sns

sns.jointplot(data_train['cont1'], data_train['loss'], kind='reg', size=6)
# ok not much to discern from cont1...lets look at the loss data a little



print (data_train['loss'].mean(), "Mean")

print (data_train['loss'].median(), "Median")

print (data_train['loss'].max(), "Maximum")

print (data_train['loss'].min(), "Minimum")
import numpy as np

# Heat Maps?

# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..cont14 into deciles

data_train['cont14_decile'] = pd.qcut(data_train['cont14'], 10, labels=False) + 1

#data_train.head()



from collections import defaultdict

#BE_ME_Portfolio = [x for x in np.arange(1, 10, 1)]

#ME_Portfolio = [x for x in np.arange(1, 10, 1)]

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont14_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the monthly return values

#monthly_returns = defaultdict(dict)

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

#for decile_BE in BE_ME_Portfolio:

    for decile_cont14 in Cont14_Data:

    #for decile_ME in ME_Portfolio:    

        #Portfolio = results.loc[(results['BE_ME_Decile'] == decile_BE) & (results['ME_Decile'] == decile_ME)]

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont14_decile'] == decile_cont14)]

        #monthly_return = np.mean(Portfolio['Month'])

        Average_Loss = np.sum(New_Frame['loss'])

        #monthly_returns[decile_BE][decile_ME] = monthly_return

        Claimed_Loss_Average[decile_loss][decile_cont14] = Average_Loss

        

#monthly_returns = pd.DataFrame(monthly_returns)

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

#monthly_returns.index.name = "ME Decile"

Claimed_Loss_Average.index.name = "cont14 decile"

#monthly_returns.columns.name = "BE_ME Decile"

Claimed_Loss_Average.columns.name = "Loss Decile"

#print monthly_returns.head()



import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont14 impact on Claim Losses ")

    pyplot.colorbar(axim)

    #savefig('sample.png')

    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print monthly_returns



### Note that I used the sum of the losses claimed in the respective deciles



# Conclusion - It looks like the tails of cont14 cause the most variance in claims
import numpy as np

# Heat Maps?

# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont1_decile'] = pd.qcut(data_train['cont1'], 10, labels=False) + 1

#data_train.head()



from collections import defaultdict

#BE_ME_Portfolio = [x for x in np.arange(1, 10, 1)]

#ME_Portfolio = [x for x in np.arange(1, 10, 1)]

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont1_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the monthly return values

#monthly_returns = defaultdict(dict)

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

#for decile_BE in BE_ME_Portfolio:

    for decile_cont1 in Cont1_Data:

    #for decile_ME in ME_Portfolio:    

        #Portfolio = results.loc[(results['BE_ME_Decile'] == decile_BE) & (results['ME_Decile'] == decile_ME)]

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont1_decile'] == decile_cont1)]

        #monthly_return = np.mean(Portfolio['Month'])

        Average_Loss = np.sum(New_Frame['loss'])

        #monthly_returns[decile_BE][decile_ME] = monthly_return

        Claimed_Loss_Average[decile_loss][decile_cont1] = Average_Loss

        

#monthly_returns = pd.DataFrame(monthly_returns)

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

#monthly_returns.index.name = "ME Decile"

Claimed_Loss_Average.index.name = "cont1 decile"

#monthly_returns.columns.name = "BE_ME Decile"

Claimed_Loss_Average.columns.name = "Loss Decile"

#print monthly_returns.head()



import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont1 impact on Claim Losses ")

    pyplot.colorbar(axim)

    #savefig('sample.png')

    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print monthly_returns



### Note that I used the sum of the losses claimed in the respective deciles



# Conclusion - It looks like the tails of cont14 cause the most variance in claims
import numpy as np

# Heat Maps?

# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont2_decile'] = pd.qcut(data_train['cont2'], 10, labels=False) + 1

#data_train.head()



from collections import defaultdict

#BE_ME_Portfolio = [x for x in np.arange(1, 10, 1)]

#ME_Portfolio = [x for x in np.arange(1, 10, 1)]

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont2_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the monthly return values

#monthly_returns = defaultdict(dict)

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

#for decile_BE in BE_ME_Portfolio:

    for decile_cont2 in Cont2_Data:

    #for decile_ME in ME_Portfolio:    

        #Portfolio = results.loc[(results['BE_ME_Decile'] == decile_BE) & (results['ME_Decile'] == decile_ME)]

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont2_decile'] == decile_cont2)]

        #monthly_return = np.mean(Portfolio['Month'])

        Average_Loss = np.sum(New_Frame['loss'])

        #monthly_returns[decile_BE][decile_ME] = monthly_return

        Claimed_Loss_Average[decile_loss][decile_cont1] = Average_Loss

        

#monthly_returns = pd.DataFrame(monthly_returns)

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

#monthly_returns.index.name = "ME Decile"

Claimed_Loss_Average.index.name = "cont2 decile"

#monthly_returns.columns.name = "BE_ME Decile"

Claimed_Loss_Average.columns.name = "Loss Decile"

#print monthly_returns.head()



import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont2 impact on Claim Losses ")

    pyplot.colorbar(axim)

    #savefig('sample.png')

    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print monthly_returns



### Note that I used the sum of the losses claimed in the respective deciles



# Conclusion - It looks like the tails of cont14 cause the most variance in claims
import numpy as np

# Heat Maps?

# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont3_decile'] = pd.qcut(data_train['cont3'], 10, labels=False) + 1

#data_train.head()



from collections import defaultdict

#BE_ME_Portfolio = [x for x in np.arange(1, 10, 1)]

#ME_Portfolio = [x for x in np.arange(1, 10, 1)]

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont3_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the monthly return values

#monthly_returns = defaultdict(dict)

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

#for decile_BE in BE_ME_Portfolio:

    for decile_cont3 in Cont3_Data:

    #for decile_ME in ME_Portfolio:    

        #Portfolio = results.loc[(results['BE_ME_Decile'] == decile_BE) & (results['ME_Decile'] == decile_ME)]

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont3_decile'] == decile_cont3)]

        #monthly_return = np.mean(Portfolio['Month'])

        Average_Loss = np.sum(New_Frame['loss'])

        #monthly_returns[decile_BE][decile_ME] = monthly_return

        Claimed_Loss_Average[decile_loss][decile_cont1] = Average_Loss

        

#monthly_returns = pd.DataFrame(monthly_returns)

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

#monthly_returns.index.name = "ME Decile"

Claimed_Loss_Average.index.name = "cont3 decile"

#monthly_returns.columns.name = "BE_ME Decile"

Claimed_Loss_Average.columns.name = "Loss Decile"

#print monthly_returns.head()



import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont3 impact on Claim Losses ")

    pyplot.colorbar(axim)

    #savefig('sample.png')

    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print monthly_returns



### Note that I used the sum of the losses claimed in the respective deciles



import numpy as np

# Heat Maps?

# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont4_decile'] = pd.qcut(data_train['cont4'], 10, labels=False) + 1

#data_train.head()



from collections import defaultdict

#BE_ME_Portfolio = [x for x in np.arange(1, 10, 1)]

#ME_Portfolio = [x for x in np.arange(1, 10, 1)]

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont4_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the monthly return values

#monthly_returns = defaultdict(dict)

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

#for decile_BE in BE_ME_Portfolio:

    for decile_cont4 in Cont4_Data:

    #for decile_ME in ME_Portfolio:    

        #Portfolio = results.loc[(results['BE_ME_Decile'] == decile_BE) & (results['ME_Decile'] == decile_ME)]

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont4_decile'] == decile_cont4)]

        #monthly_return = np.mean(Portfolio['Month'])

        Average_Loss = np.sum(New_Frame['loss'])

        #monthly_returns[decile_BE][decile_ME] = monthly_return

        Claimed_Loss_Average[decile_loss][decile_cont1] = Average_Loss

        

#monthly_returns = pd.DataFrame(monthly_returns)

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

#monthly_returns.index.name = "ME Decile"

Claimed_Loss_Average.index.name = "cont4 decile"

#monthly_returns.columns.name = "BE_ME Decile"

Claimed_Loss_Average.columns.name = "Loss Decile"

#print monthly_returns.head()



import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont4 impact on Claim Losses ")

    pyplot.colorbar(axim)

    #savefig('sample.png')

    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print monthly_returns



### Note that I used the sum of the losses claimed in the respective deciles





from collections import defaultdict

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont5_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the loss claim values

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

    for decile_cont5 in Cont5_Data:

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont5_decile'] == decile_cont4)]

        Average_Loss = np.sum(New_Frame['loss'])

        Claimed_Loss_Average[decile_loss][decile_cont1] = Average_Loss

        

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

Claimed_Loss_Average.index.name = "cont4 decile"

Claimed_Loss_Average.columns.name = "Loss Decile"





import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont4 impact on Claim Losses ")

    pyplot.colorbar(axim)



    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print Claimed_Loss_Average



### Note that I used the sum of the losses claimed in the respective deciles
# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont6_decile'] = pd.qcut(data_train['cont6'], 10, labels=False) + 1



from collections import defaultdict

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont6_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the loss claim values

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

    for decile_cont6 in Cont6_Data:

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont6_decile'] == decile_cont6)]

        Average_Loss = np.sum(New_Frame['loss'])

        Claimed_Loss_Average[decile_loss][decile_cont6] = Average_Loss # Source of error (fix this in prior cells)

        

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

Claimed_Loss_Average.index.name = "cont6 decile"

Claimed_Loss_Average.columns.name = "Loss Decile"





import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont6 impact on Claim Losses ")

    pyplot.colorbar(axim)



    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print Claimed_Loss_Average



### Note that I used the sum of the losses claimed in the respective deciles

# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont7_decile'] = pd.qcut(data_train['cont7'], 10, labels=False) + 1



from collections import defaultdict

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont7_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the loss claim values

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

    for decile_cont7 in Cont7_Data:

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont7_decile'] == decile_cont7)]

        Average_Loss = np.sum(New_Frame['loss'])

        Claimed_Loss_Average[decile_loss][decile_cont7] = Average_Loss # Source of error (fix this in prior cells)

        

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

Claimed_Loss_Average.index.name = "cont7 decile"

Claimed_Loss_Average.columns.name = "Loss Decile"





import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont7 impact on Claim Losses ")

    pyplot.colorbar(axim)



    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print Claimed_Loss_Average



### Note that I used the sum of the losses claimed in the respective deciles

# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont8_decile'] = pd.qcut(data_train['cont8'], 10, labels=False) + 1



from collections import defaultdict

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont8_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the loss claim values

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

    for decile_cont8 in Cont8_Data:

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont8_decile'] == decile_cont8)]

        Average_Loss = np.sum(New_Frame['loss'])

        Claimed_Loss_Average[decile_loss][decile_cont8] = Average_Loss # Source of error (fix this in prior cells)

        

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

Claimed_Loss_Average.index.name = "cont8 decile"

Claimed_Loss_Average.columns.name = "Loss Decile"





import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont8 impact on Claim Losses ")

    pyplot.colorbar(axim)



    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print Claimed_Loss_Average



### Note that I used the sum of the losses claimed in the respective deciles
# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont9_decile'] = pd.qcut(data_train['cont9'], 10, labels=False) + 1



from collections import defaultdict

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont9_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the loss claim values

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

    for decile_cont9 in Cont9_Data:

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont9_decile'] == decile_cont9)]

        Average_Loss = np.sum(New_Frame['loss'])

        Claimed_Loss_Average[decile_loss][decile_cont9] = Average_Loss # Source of error (fix this in prior cells)

        

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

Claimed_Loss_Average.index.name = "cont9 decile"

Claimed_Loss_Average.columns.name = "Loss Decile"





import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont9 impact on Claim Losses ")

    pyplot.colorbar(axim)



    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print Claimed_Loss_Average



### Note that I used the sum of the losses claimed in the respective deciles
# First split the loss data into deciles

data_train['loss_decile'] = pd.qcut(data_train['loss'], 10, labels=False) + 1

# Now split ..continous data into deciles

data_train['cont10_decile'] = pd.qcut(data_train['cont10'], 10, labels=False) + 1



from collections import defaultdict

Loss_Data = [x for x in np.arange(1, 10, 1)]

Cont10_Data = [x for x in np.arange(1, 10, 1)]



#: Create a dictionary to hold all the loss claim values

Claimed_Loss_Average = defaultdict(dict)



for decile_loss in Loss_Data:

    for decile_cont10 in Cont10_Data:

        New_Frame = data_train.loc[(data_train['loss_decile'] == decile_loss) & (data_train['cont10_decile'] == decile_cont10)]

        Average_Loss = np.sum(New_Frame['loss'])

        Claimed_Loss_Average[decile_loss][decile_cont10] = Average_Loss # Source of error (fix this in prior cells)

        

Claimed_Loss_Average = pd.DataFrame(Claimed_Loss_Average)

Claimed_Loss_Average.index.name = "cont10 decile"

Claimed_Loss_Average.columns.name = "Loss Decile"





import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_xticks(np.arange(len(df.columns)))

    ax.set_xticklabels(list(df.columns))

    ax.set_ylabel(df.index.name)

    ax.set_yticks(np.arange(len(df.index)))

    ax.set_yticklabels(list(df.index))

    ax.set_title("Cont10 impact on Claim Losses ")

    pyplot.colorbar(axim)



    

#: Plot our heatmap

heat_map(Claimed_Loss_Average)

#print Claimed_Loss_Average



### Note that I used the sum of the losses claimed in the respective deciles
# Now we want to test all the data at once, a neural network is probably a good idea



# to do for NN



# Convert "cat" data to numbers









# Convert loss data to 0-1

# Fix bug in tensorflow

# Data sets

IRIS_TRAINING = "../input/train.csv"

IRIS_TEST = "../input/test.csv"



with open("../input/test.csv",'r') as f:

    with open("updated_test.csv",'w') as f1:

        f.next() # skip header line

        for line in f:

            f1.write(line)

            

            
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import tensorflow as tf

import numpy as np



# Data sets

IRIS_TRAINING = "../input/train.csv"

IRIS_TEST = "../input/test.csv"



# Load datasets.

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,

                                                                  target_dtype=np.int,

                                                                  features_dtype=np.float32,

                                                                  target_column=-1)

test_set     = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST,

                                                                  target_dtype=np.int,

                                                                  features_dtype=np.float32,

                                                                  target_column=-1)
# Specify that all features have real-value data

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=132)]



# Build 3 layer DNN with 10, 20, 10 units respectively.

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,

                                            hidden_units=[10, 20, 10],

                                            n_classes=131,

                                            model_dir="/tmp/iris_model")



# Fit model.

classifier.fit(x=training_set.data, 

               y=training_set.target, 

               steps=2000)



# Evaluate accuracy.

accuracy_score = classifier.evaluate(x=test_set.data,

                                     y=test_set.target)["accuracy"]

print('Accuracy: {0:f}'.format(accuracy_score))



# Classify two new flower samples.

#new_samples = np.array(

#    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)

#y = classifier.predict(new_samples)

#print('Predictions: {}'.format(str(y)))