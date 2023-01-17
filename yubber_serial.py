# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra. very fast. written in cpp

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# as is alias – numpy.function() becomes np.function()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# You thought I was a Kaggle kernel, but it was actually me, a Jupyter notebook!
['cereal.csv']
# numpy crash course

namaste = np.array(range(5)) # run first block of code before this. this produces a static array



print(namaste[::-1]) # note that numpy arrays do not have typical list methods - they can't be changed



print(namaste*2) # adding or multiplying arrays maps Arr[i] * Arrr[i]

print([69,420]*2) # adding or multiplying lists concats them



print("\n")



print(np.array([0,"69",420, False])) # np arrays coerce all list elems to be same data type. strings are easiest to convert to

print(np.array([False, 69]))

print(np.array([False,True])+2) # test with + and *



print(np.arange(0,420,69)) # np's range
# table demo



table = [ # i hope you like matrices

    [1,2,3],

    [69,420,0],

    [345,76,876],

    [0,0,0]

]



print(table,"\n")



neo = np.matrix(table)



print(neo/69, "\n") # using operators on matrices also maps them to each elem instead of concat



print(neo.reshape(12,1),"\n") # reshaping into non-factors gives error



print(neo.reshape(6,-1),"\n") # negative tells np to divide appropriately and use clean value (ie unknown dimension). can only specify 1. -0 evaluates to 0
# pretty picture



# pic = np.matrix([

#     [list([0,255,255]),list([0,0,0]),list([255,255,255])],

#     [list([255,0,69]),list([69,69,69]),list([69,69,0])]

# ]) # hast thou no heart for rgb values | would you like the tuple or the list



pak = np.matrix([ [1,2,3,4,5,6,6,7,8,9,65,876,5,0] ]).reshape(-1, 2)



pic = np.matrix([

    [[x for x in (69,420)],2,3],

    [4,5,6]

])



print(pic,pak,sep="\n\n\n") # the universe is a matrix we are all matrices we are all matrices



print(pic.reshape(-1,1)) # gimp ain't got shit on me



print(pic.shape) # attr returns (rows, columns)
aie = np.array(["hue", "heu", "jejeje", "pelo"])



print(aie == "heu", aie != "hue", (aie == "jejeje") | (aie == "pelo"), "", sep="\n\n") # since python obeys pemdas, it evaluates innermost brackets first (ie it evals jejeje & pelo before the bitwise |)



print(aie[(aie == "pelo") | (aie == "heu")],"\n\n")



print(aie[False], aie[True], np.array([1,2,3,4,4,5,69,0,45,420]).sum(), sep="\n\n") # apparently 001 bad



print(np.array([1,2,3,4,5,6]).std(axis=0)) # sum is sum of elems on given axis. std is standard deviation you perv. **1/3 is cbrt.



'''

[[1],[2],[3]] + [[1,2,3]] broadcasts to 1+1, 1+2, 1+3 | 2+1, 2+2, 2+3 | 3+1 3+2 3+3

np will try to duplicate rows/columns to match but it will not rotate or reshape.

ie: if it shares the same row/col count OR one matrix is thin (1 row/col), it can be broadcast

'''

# alright younglings it's DATASET TIME



data = pd.read_csv("../input/cereal.csv")



obesity = data["calories"] >= 100 # cereal with calories > 100. data["calories"] >= 100 returns bools



print( "mean sugar val of cereal with >=100 cal: ", data["sugars"][obesity].mean() )



diabete = data["sugars"] 



# qu'est la céréal avec le plus/moins de calorie? honhon ich parle fransais wie wie



print("max sugar: ", data["name"][diabete == diabete.max()]) # or use the inbuilt python max() | note that 2 cereals share max



print("max cals: ", data["name"][ data["calories"] == data["calories"].max() ]) # print cereal name w max cals



print("",help(data.head),data.head(),data.tail(),sep="\n\n") # head is top of dataset, tail is bottom



# panda. panda. panda. panda. panda.



print(data.iloc[0],"loc:",data.loc[69],"",sep="\n\n") # there's no column with 0; data[0] looks for keys inside columns.

# iloc returns 1st row + keys, referencing row order as indices (1st row is 0 even if its index != 0). loc references row # (actual number used to label row) 



# data.index(list) replaces data[0]'s index with list[0]'s number value. don't mess with index willy-nilly though

# the fancy printed stuff is a pandas series (i.e. pandas' version of np array). a 2d series is a dataframe.



# to get cereal name smacks' data:

print(data.loc[ data["name"] == "Total Corn Flakes" ] )



# print("",data.iloc[data[42:69:2]],sep="\n\n") # supports slices



print(data.describe()) # percentage values are mean of percentile. only numerical values can be statesticled.



data.loc[data["protein"]>3] # -1 means data missing. NaN will fuck up ML models. workarounds include replacing value with mean of stuff made by same producer/similar products, etc. 



data.replace(-1,np.nan) # replace all -1 vals with nanna



data.loc[:,["carbo","potassium","sugars","weight"]] # data.loc[:] gives all columns. data.loc[0:30] gives first 31. data.loc[0:68,["a","b","c"]] gives first 70 columns' a, b and c values.



# you can pop from dicts with the key or .drop([key],axis=1). axis=1 overrides default axis
searcher = ["nuts","corn","fruit","bran","cinnamon","apple","almond","cocoa","date","walnut","oat","honey","bleach","wheat","raisin","rice","oatmeal"]



def checkgred(name):

    giver = []

    for i in searcher:

        if name.lower().find(i) != -1:

            giver.append(i)

    if len(giver) > 0:

        return giver



data["ingredients"] = data["name"].apply(checkgred) # .apply(fun) executes a function as if it were a child of data["name"]



print(data)