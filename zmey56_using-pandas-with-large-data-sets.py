import pandas as pd



gl = pd.read_csv('../input/game_logs.csv', low_memory=False)

gl.head()
# set the memory_usage parameter to 'deep' to get an accurate number



gl.info(memory_usage='deep')
# the average memory usage for data type

for dtype in ['float','int','object']:

    selected_dtype = gl.select_dtypes(include=[dtype])

    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()

    mean_usage_mb = mean_usage_b / 1024 ** 2

    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
# the minimum and maximum values for each integer subtype



import numpy as np

int_types = ["uint8", "int8", "int16"]

for it in int_types:

    print(np.iinfo(it))


# select only the integer columns

# optimize the types 

# compare the memory usage





def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # исходим из предположения о том, что если это не DataFrame, то это Series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # преобразуем байты в мегабайты

    return "{:03.2f} MB".format(usage_mb)



gl_int = gl.select_dtypes(include=['int'])

converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')



print(mem_usage(gl_int))

print(mem_usage(converted_int))



compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)

compare_ints.columns = ['before','after']

compare_ints.apply(pd.Series.value_counts)
# with float columns



gl_float = gl.select_dtypes(include=['float'])

converted_float = gl_float.apply(pd.to_numeric,downcast='float')



print(mem_usage(gl_float))

print(mem_usage(converted_float))



compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)

compare_floats.columns = ['before','after']

compare_floats.apply(pd.Series.value_counts)
# create a copy of original dataframe

# assign optimized numeric columns in place of the originals

# see what overall memory usage is now



optimized_gl = gl.copy()



optimized_gl[converted_int.columns] = converted_int

optimized_gl[converted_float.columns] = converted_float



print(mem_usage(gl))

print(mem_usage(optimized_gl))
# looking at individual strings

# looking items in a pandas series



from sys import getsizeof



s1 = 'working out'

s2 = 'memory usage for'

s3 = 'strings in python is fun!'

s4 = 'strings in python is fun!'



for s in [s1, s2, s3, s4]:

    print(getsizeof(s))
obj_series = pd.Series(['working out',

                          'memory usage for',

                          'strings in python is fun!',

                          'strings in python is fun!'])

obj_series.apply(getsizeof)
# the number of unique values of each of object types



gl_obj = gl.select_dtypes(include=['object']).copy()

gl_obj.describe()
# convert to categorical



dow = gl_obj.day_of_week

print(dow.head())



dow_cat = dow.astype('category')

print(dow_cat.head())
# return the integer values the category type uses to represent each value



dow_cat.head().cat.codes
# memory usage for this column before and after converting



print(mem_usage(dow))

print(mem_usage(dow_cat))
# iterate over each object column

# if the number of unique values is less than 50% - convert it to the category type



converted_obj = pd.DataFrame()



for col in gl_obj.columns:

    num_unique_values = len(gl_obj[col].unique())

    num_total_values = len(gl_obj[col])

    if num_unique_values / num_total_values < 0.5:

        converted_obj.loc[:,col] = gl_obj[col].astype('category')

    else:

        converted_obj.loc[:,col] = gl_obj[col]
print(mem_usage(gl_obj))

print(mem_usage(converted_obj))



compare_obj = pd.concat([gl_obj.dtypes,converted_obj.dtypes],axis=1)

compare_obj.columns = ['before','after']

compare_obj.apply(pd.Series.value_counts)
# combine this with the rest 



optimized_gl[converted_obj.columns] = converted_obj



mem_usage(optimized_gl)
# use datetime for the first column of data set



date = optimized_gl.date

print(mem_usage(date))

date.head()
# convert 



optimized_gl['date'] = pd.to_datetime(date,format='%Y%m%d')



print(mem_usage(optimized_gl))

optimized_gl.date.head()
# apply memory-saving techniques when can't even create the dataframe in the first place

# specify the optimal column types when read the data set in





dtypes = optimized_gl.drop('date',axis=1).dtypes



dtypes_col = dtypes.index

dtypes_type = [i.name for i in dtypes.values]



column_types = dict(zip(dtypes_col, dtypes_type))



# rather than print all 161 items, we'll

# sample 10 key/value pairs from the dict

# and print it nicely using prettyprint



preview = first2pairs = {key:value for key,value in list(column_types.items())[:10]}

import pprint

pp = pp = pprint.PrettyPrinter(indent=4)

pp.pprint(preview)

#  to read in the data with the correct types in a few lines



read_and_optimized = pd.read_csv('../input/game_logs.csv',dtype=column_types,parse_dates=['date'],infer_datetime_format=True)



print(mem_usage(read_and_optimized))

read_and_optimized.head()
import matplotlib.pyplot as plt
# the distribution of game days.



optimized_gl['year'] = optimized_gl.date.dt.year

games_per_day = optimized_gl.pivot_table(index='year',columns='day_of_week',values='date',aggfunc=len)

games_per_day = games_per_day.divide(games_per_day.sum(axis=1),axis=0)



ax = games_per_day.plot(kind='area',stacked='true')

ax.legend(loc='upper right')

ax.set_ylim(0,1)

plt.show()
# how game length has varied over the years



game_lengths = optimized_gl.pivot_table(index='year', values='length_minutes')

game_lengths.reset_index().plot.scatter('year','length_minutes')

plt.show()