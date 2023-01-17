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
n = int(input())

m = int(input())



is_multiple(n, m)



def is_multiple(n, m):

    if n % m == 0:

        print('True')

        return True

    else:

        print('False')

        return False
[ 2 ** i for i in range(0, 9)]
nums = [1, 2, 3, 5, 7]

determine_distinct(nums)



def determine_distinct(nums):

    for a in range(0, len(nums) - 1):

        for b in range(a + 1, len(nums)):

            if nums[a] == nums[b]:

                print('False')

                return False

    print('True')

    return True
n = int(input())

harmonic_gen(n)



def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1/i

        yield h