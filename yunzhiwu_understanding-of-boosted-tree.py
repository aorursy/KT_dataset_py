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
all_data = pd.read_csv("../input/winemag-data_first150k.csv")

all_data[:5]
pd.DataFrame.describe(all_data)
all_data = all_data.dropna(subset=['price'])  # drop NaN in column price

n1000 = all_data.sample(n=1000)

train_set = n1000.sample(frac=0.7)

test_set = n1000.loc[~n1000.index.isin(train_set.index)]
import seaborn as sns

sns.distplot(train_set["points"], kde = False)
train_set.country.value_counts()
train_set.variety.value_counts()
lambd = 0.05  # just try this value

gamma = 0.05  # just try this value



w = [0.5, 0.5, 0.5, 0.5] # initialize weight to each leaf, T = 4



price_caret = pd.Series([0] * 700)      # initialize output 0

train_set = train_set.assign(price_caret=price_caret.values)



iteration = 0
def get_leaf_index(points, country, variety):

    if points < 87:

        if country == "US":

            index = 0

        else:

            index = 1

    else:

        if variety == "Chardonnay":

            index = 2

        else:

            index = 3

    return index
# one iteration, use y_caret(t-1) to calculate w(t) and Obj(t)

def iter_calc_weight_obj():

    global iteration

    global train_set

    iteration = iteration + 1

    

    G = [0, 0, 0, 0]

    H = [0, 0, 0, 0]

    for index, wine in train_set.iterrows():

            j = get_leaf_index(wine["points"], wine["country"], wine["variety"])

            G[j] = G[j] + 2 * (wine["price_caret"] - wine["price"])  # 1st-order partial derivatives

            H[j] = H[j] + 2  # 2nd-order partial derivatives



    # get the optimal weight

    for j in range(0, len(w)):

        w[j] = -(G[j] / (H[j] + lambd))



    print("weight for each leaves for iteration {0}:".format(iteration))

    print(w)



    # the resulting objective value

    obj = 0

    for j in range(0, len(w)):

        obj = obj + G[j] * G[j] / (H[j] + lambd)



    obj = - obj / 2 + gamma * 4



    print("resulting objective value for iteration {0}:".format(iteration))

    print(obj)

    print("")



    # update the price_caret

    for index, wine in train_set.iterrows():

        leaf_index = get_leaf_index(wine["points"], wine["country"], wine["variety"])

        train_set.set_value(index,'price_caret',w[leaf_index])
iter_calc_weight_obj()

iter_calc_weight_obj()
iter_calc_weight_obj()

iter_calc_weight_obj()
iter_calc_weight_obj()

iter_calc_weight_obj()