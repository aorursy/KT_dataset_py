# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import random as rand

import matplotlib.pyplot as plt

%matplotlib inline

import sys

if sys.version_info[0] < 3: 

    from StringIO import StringIO

else:

    from io import StringIO







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

import subprocess as sbps

file_name_list = check_output(["ls", "../input"]).decode("utf8")

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# p = sbps.Popen(['ls', '-lh', '../input'], stdout=sbps.PIPE)

print(

    sbps.check_output(['ls', '-lh', '../input']).decode('utf8')    ,

)
def get_file_head(file_name, n_head):

    return sbps.check_output(

        ['head', '-{}'.format(n_head), '../input/{}'.format(file_name)]

    )



def get_file_length(file_name):

    p = sbps.Popen(

        ['cat', '../input/{}'.format(file_name)],

        stdout=sbps.PIPE

    )

    return sbps.check_output(

        ['wc', '-l'],

        stdin=p.stdout

    )

    
for file_name in file_name_list.split('\n')[:-1]:

    print(

        '-------------{}-------------'.format(file_name)

    )

    print(

        '{}'.format(

            get_file_head(file_name, n_head=5)

        ).split('\\n')

    )

    print(

        'file_length: {}'.format(

            get_file_length(file_name)

        )

    )
tl_destinations = 62107

tl_test = 2528244

tl_train = 37670294



# lets count sample size needed with confidence level 0.99

# and margin of error = 0.01

ts_train = 16580

ts_test = 16480

ts_destinations = 13092
def create_file_buffer_from_file(file_name, n_sample, total_lines):

    # open file handler

    file_path = '../input/{}'.format(file_name)

    f = open(file_path)

    file_buffer = list()

    rows = np.sort(

        np.random.randint(

            1, 

            tl_train,

            size=n_sample

        )

    )

    # get the file header

    f.seek(0)

    header_line = f.readline()

    file_buffer.append(header_line)

    for row in rows:

        f.seek(row)

        f.readline() #discard - bound to be partial line

        file_buffer.append(

            f.readline()

        )

    

    f.close()

    return StringIO('\n'.join(file_buffer))

    

def get_random_sample_from_file(file_name, n_sample, total_lines):    

    

    return pd.read_csv(

        create_file_buffer_from_file(

            file_name, 

            n_sample, 

            total_lines

        )

    )

    
_df_train = get_random_sample_from_file('train.csv', ts_train, tl_train)

_df_train.head()
_df_train.describe()
_df_train.columns
_df_train.groupby('is_booking')['date_time'].count()
_df_train['dt'] = pd.to_datetime(_df_train['date_time'])

_df_train['dt'].dtype
_df_train['date_time'].dtype