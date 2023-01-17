import numpy as np # importing numpy

import pandas as pd # importing pandas



play_store = pd.read_csv('../input/googleplaystore.csv') # loading data set

print(play_store.shape)

play_store.head(10)
# facilitating the column names

new_columns = []

old_columns = play_store.columns



for col in old_columns:

    if col == 'Size':

        new_columns.append('_size')

    else:

        new_columns.append(col.lower().replace(' ', '_'))

        

play_store.columns = new_columns



print('{}\n{}\n\n{}\n{}'.format('Old Columns', old_columns, 'New Columns', new_columns))

play_store.sample(5)
# arranging line 10472

print('{}\n{}\n'.format('BEFORE', play_store.iloc[10472]))



for i in range((play_store.shape[1] - 1), 0, -1):

    play_store.iloc[10472, i] = play_store.iloc[10472, (i - 1)]

    play_store.iloc[10472, 1] = np.nan

    

print('{}\n{}'.format('AFTER', play_store.iloc[10472]))
print('BEFORE')

for i in [2, 3, 4, 5, 7]:

    column = play_store.iloc[:, i]

    print('{}\n{}'.format(column.name, column.head(5)))



play_store.rating = play_store.rating.astype('float32')

play_store.reviews = play_store.reviews.astype('int32')



def only_num(x):

    num = ''

    dot = w = -1



    for i in range(0, len(x)):

        if x[i] == '.':

            zero_left = True

            

            for ii in range(++i, len(x)):

                if x[ii].isnumeric() and x[ii] != '0':

                    zero_left = False

                    w = ii - i

                    

            if zero_left:

                break

            else:

                num = num + x[i]

                dot = i

            

        elif x[i].isnumeric():

            num = num + x[i]



    if num == '':

        num = '0'



    return num, dot, w

def numeralizing(x):

    if x == 'Varies with device':

        num = '-1'

    else:

        zero = ''

        rmv_dot = False

        z = w = 0

    

        if x.find('k') >= 0:

            zero = '000'

            rmv_dot = True

        elif x.find('M') >= 0:

            zero = '000000'

            rmv_dot = True

    

        num, z, w = only_num(x)

        

        if rmv_dot and z >= 0:

            num = num[0:z] + num[(z + 1):len(num)]

            

        zero = zero[0:(len(zero) - w)]

        num = num + zero



    return num



play_store._size = play_store._size.apply(numeralizing)

play_store._size = play_store._size.astype('int32')



play_store.installs = play_store.installs.apply(numeralizing)

play_store.installs = play_store.installs.astype('int32')



play_store.price = play_store.price.apply(numeralizing)

play_store.price = play_store.price.astype('float32')



print('\nAFTER')

for i in [2, 3, 4, 5, 7]:

    column = play_store.iloc[:, i]

    print('{}\n{}'.format(column.name, column.head(5)))

    

play_store.sample(5)
play_store.category = play_store.category.astype('category')

play_store.type = play_store.type.astype('category')

play_store.content_rating = pd.Categorical(play_store.content_rating,

                                           categories = ['Unrated',  'Everyone', 'Everyone 10+', 'Teen''Mature 17+', 'Adults only 18+'])

play_store.genres = play_store.genres.astype('category')

play_store.android_ver = play_store.android_ver.astype('category')
print('{}\n{}'.format('BEFORE', play_store.last_updated.head(5)))



play_store.last_updated = pd.to_datetime(play_store.last_updated.values)



print('\n{}\n{}'.format('AFTER', play_store.last_updated.head(5)))
play_store.head(10)