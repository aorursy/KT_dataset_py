# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



props = pd.read_csv(r"../input/properties_2016.csv")  #The properties dataset

train = pd.read_csv(r"../input/train_2016_v2.csv")   # The parcelid's with their outcomes



# Any results you write to the current directory are saved as output.
# Params: 

#  - colobj Column 

#  - verbose - prints info messages when verbose > 0

# Return value:

#  - Returns new optimised type (if any found), or original type



def get_min_memory_type(colobj, verbose = 0):

    orig_type = colobj.dtype

    new_types = ["uint8", "int8", "uint16", "int16", 'float16', "uint32", "int32", 'float32']

    if (orig_type not in new_types and orig_type not in ["int64", "uint64", "float64"]):

        if (verbose > 0):

            print("Type {} of column {} is unsupported".format(orig_type, colobj.name))

        return orig_type

        

    for new_type in new_types:

        yes_min = colobj.min() == colobj.min().astype(new_type).astype(orig_type)

        yes_max = colobj.max() == colobj.max().astype(new_type).astype(orig_type)

        if (yes_min and yes_max): 

            return new_type



    if (verbose > 0):

        print("No optimisation found for {} column of {} type".format(colobj.name, orig_type))

    return orig_type

# Params: 

#  - df Dataframe

#  - exclude - exclude columns from transformation

#  - verbose - prints info messages when verbose > 0

# Return value:

#    - Returns percentage memory use changes

#         Negative values mean memory use increase by mem_usg_diff_prc % after transformation

#         Positive values mean memory use improved by mem_usg_diff_prc % after transformation



def reduce_mem_usage(df, exclude = [], verbose = 0):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    if (verbose > 0):

        print("Memory usage before :",start_mem_usg," MB")

    

    for colname in df.columns:

        if (colname not in exclude):

            colobj = getattr(df, colname)

            new_type = get_min_memory_type(colobj, verbose)

            if (new_type != colobj.dtype):

                if (verbose > 0):

                    print("Converting {} column from {} to {}".format(colname, colobj.dtype, new_type))

                df[colname] = df[colname].astype(new_type)



    end_mem_usg = df.memory_usage().sum() / 1024**2 

    

    if (verbose > 0):

        print("Memory usage after :",end_mem_usg," MB")

       

    mem_usg_diff_prc = float((end_mem_usg-start_mem_usg))*100/start_mem_usg

    

    if (verbose > 0):

        if (end_mem_usg < start_mem_usg):

            print("Memory gain: {0:0.2f}%".format(-mem_usg_diff_prc))

        else:

            print("Memory loss: {0:0.2f}%".format(mem_usg_diff_prc))

  

    return -mem_usg_diff_prc

   

# Example of use



reduce_mem_usage(train, ["logerror"], 1)
# Safe to run multiple times

reduce_mem_usage(train, ["logerror"], 1)
def fillna_mean(df, cols):

    for col in cols:

        mean_values = df[[col]].mean(axis=0)

        print(col, mean_values[col])

        df[col].fillna(mean_values[col], inplace=True)

# fill in n/a values with means

fillna_mean(train, ["logerror"])
# Verify train again

reduce_mem_usage(train, [], 1)
# Fill n/a values for part of the columns

fillna_mean(props, ["airconditioningtypeid", "architecturalstyletypeid", 

                  "basementsqft", "bathroomcnt", "bedroomcnt", 

                  "buildingclasstypeid", "buildingqualitytypeid", 

                  "calculatedbathnbr", "decktypeid", "finishedfloor1squarefeet"])
# Optimise part of the props where columns have no n/a values



reduce_mem_usage(props[[ "airconditioningtypeid", "architecturalstyletypeid", 

                       "basementsqft", "bathroomcnt", "bedroomcnt", 

                       "buildingclasstypeid", "buildingqualitytypeid", 

                       "calculatedbathnbr", "decktypeid", 

                       "finishedfloor1squarefeet"]], [], 1)