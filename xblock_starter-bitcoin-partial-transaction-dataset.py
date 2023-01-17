# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import matplotlib.pyplot as plt
# #  Store the label information as label_address_dict, key: label, value: addresses belonging to the label

# domain = os.path.abspath('../input/bitcoin-partial-transaction-dataset/label')

# label_address_dict = {}  # store the label addresses



# for labeled_file_name in os.listdir('../input/bitcoin-partial-transaction-dataset/label'):

#     filepath = os.path.join(domain, labeled_file_name)               # obtain the filename

#     label = labeled_file_name.split('.')[0].split('-')[0]            # obtain the label

#     if label not in label_address_dict:

#         label_address_dict[label] = []

#     with open(filepath, 'r') as f:

#         lines = f.readlines()

#         for line in lines:

#             labeled_address = line.rstrip('\n')

#             label_address_dict[label].append(labeled_address)



# print('labels: ' + str(list(label_address_dict.keys())))

# print('an example address belonging to BitcoinFog: ' + label_address_dict['BitcoinFog'][0])

# nums = []

# for key in label_address_dict.keys():

#     nums.append(len(label_address_dict[key]))

# plt.bar(list(label_address_dict.keys()), nums)

# plt.ylabel('Number')

# plt.title('Number of addresses belonging to the label')

# for x,y in enumerate(nums):

#     plt.text(x,y,'%s' %round(y,1),ha='center')

# plt.show()