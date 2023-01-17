import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print(pd.__version__)
print(pd.show_versions(as_json=True))
a_list = list('abcdefg')

np_array = np.arange(1,10)

dict = {'a': 0 , 'b' : 1, 'c' : 2, 'd' : 3}
print(a_list)

print(np_array)

print(dict)
list_series = pd.Series(a_list)

npa_series = pd.Series(np_array)

dict_series = pd.Series(dict)
print(list_series)

print(npa_series)

print(dict_series)
pdser = pd.Series(dict_series)

print(type(pdser))
dfser = pd.DataFrame(pdser)

print(type(dfser))

print(dfser.shape)

print(dfser)
dfser.reset_index()
dir(pd)
print(pd.Index.__doc__)
help(pd.Index)
ser_lower = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))

ser_upper = pd.Series(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

ser_letterindex = pd.Series(np.arange(26))

# This can be done using either the dictionary argument to the DataFrame method, 

# or the list of Series objects to the concat method



#df_alphabet = pd.DataFrame({'lower': ser_lower, 'upper' : ser_upper, 'slno' : ser_letterindex})



df_alphabet = pd.concat([ser_lower, ser_upper, ser_letterindex], axis = 1)



df_alphabet.rename(columns={0: "Lower", 1: "Upper", 2: "letterIndex"}, inplace=True)

df_alphabet.rename_axis("alphabet")



df_alphabet
oneseries = pd.Series([1,3,5,2,4,8])

twoseries = pd.Series([2,1,5,7,0,9])



#Elements NOT common between arrays

s1 = oneseries[~oneseries.isin(twoseries)]



#Elements Common between arrays

s2 = oneseries[oneseries.isin(twoseries)]

print("Elements NOT common with their indices")

print(s1)

print("Elements common with their indices")

print(s2)
one_not2 = oneseries[~oneseries.isin(twoseries)]

two_not1 = twoseries[~twoseries.isin(oneseries)]



one_not2.append(two_not1, ignore_index = True)
series_union = pd.Series(np.union1d(oneseries, twoseries))

series_intersection = pd.Series(np.intersect1d(oneseries, twoseries))



print(series_union)

print(series_intersection)



series_uncommon = series_union[~series_union.isin(series_intersection)]

print(series_uncommon)
state = np.random.RandomState(25)

print(type(state))

randomseries = pd.Series(state.normal(5,10,25))

print(randomseries)

print(type(state.normal(5,10,25)))

print(state.normal(5,10,25))
#using pandas describe

print(randomseries.describe())



#using numpy percentile

print(np.percentile(randomseries, q=[25,50,75,100]))



#finding any percentile in the series using numpy

print(np.percentile(randomseries, q  = [66]))
frequencyseries = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))

print(frequencyseries)



print("value counts :\n" , frequencyseries.value_counts())

print(frequencyseries.value_counts())