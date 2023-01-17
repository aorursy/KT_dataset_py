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
# test instance 1

# weights = [1,2,3,4]

# values = [2,4,4,5]

# C=5



#test instance 2

weights = [1,2,5,6,7]

values = [1,6,18,22,28]

C=11



pick_value = [ [0]*(C+1) for i in range(len(weights)+1)]  #this table store the total value, initialize all to 0

pick_up = [ [[]]*(C+1) for i in range(len(weights)+1)]  #this table store the pick up infomation, initialize all to empty list



#review these two values

for i in range(len(weights)+1):

    print(pick_value[i])

    

for i in range(len(weights)+1):

    print(pick_up[i])

    

for row in range(1,len(weights)+1):

    item_num = row-1  #row means the row num of the table. which starts from 1. however, item num starts from 0.

    print(item_num,"----")

    for col in range(1,C+1):

        print(row,"x",col)

        weight = col

        if(weights[item_num]>weight):

            pick_value[row][col] = pick_value[row-1][col]

            print("total weight %d is less than the weight %d of item %d, give up"%(weight,weights[item_num],item_num))

        else:

#             pick_value[row][col] = max(pick_value[rol-1][col],pick_value[rol-1][int(col-weights[item_num])]+values[item_num])

            print("total weight %d is more than the weight %d of item %d, try have"%(col,weights[item_num],item_num))

            if(pick_value[row-1][col] < pick_value[row-1][int(col-weights[item_num])]+values[item_num]):

                pick_value[row][col] = pick_value[row-1][int(col-weights[item_num])]+values[item_num]

                print("[%d,%d] %d add new item value %d = %d is more than the previous value [%d,%d] %d, pick up"%(row-1,int(col-weights[item_num]),pick_value[row-1][int(col-weights[item_num])],values[item_num],pick_value[row][col],row-1,col,pick_value[row-1][col]))

                pick_up[row][col] = pick_up[row-1][int(col-weights[item_num])] + [item_num]

            else:

                print("[%d,%d] %d add new item value %d = %d is less than the previous value [%d,%d] %d, not worth it"%(row-1,int(col-weights[item_num]),pick_value[row-1][int(col-weights[item_num])],values[item_num],pick_value[row-1][int(col-weights[item_num])]+values[item_num],row-1,col,pick_value[row-1][col]))

                pick_value[row][col] = pick_value[row-1][col]

                pick_up[row][col] = pick_up[row-1][col]



for i in range(len(weights)+1):

    print(pick_value[i])



for i in range(len(weights)+1):

    print(pick_up[i])

    

print(pick_up[-1][-1])
file_path="../input/Knapsack_Instances.csv"

knapsack_instances = pd.read_csv(file_path, header = None, skip_blank_lines= False)

knapsack_instances
weights = list(knapsack_instances.loc[0])

values = list(knapsack_instances.loc[1])



for i in range(len(weights)):

    print(i,weights[i],values[i])
def items_pick(weights, values, C):

    pick_value = [ [0]*(C+1) for i in range(len(weights)+1)]

    pick_up = [ [[]]*(C+1) for i in range(len(weights)+1)]

    for row in range(1,len(weights)+1):

        item_num = row-1

        for col in range(1,C+1):

            weight = col

            if(weights[item_num]>weight):

                pick_value[row][col] = pick_value[row-1][col]

                pick_up[row][col] = pick_up[row-1][col]

            else:

                if(pick_value[row-1][col] < pick_value[row-1][int(col-weights[item_num])]+values[item_num]):

                    pick_value[row][col] = pick_value[row-1][int(col-weights[item_num])]+values[item_num]

                    pick_up[row][col] = pick_up[row-1][int(col-weights[item_num])] + [item_num]

                else:

                    pick_value[row][col] = pick_value[row-1][col]

                    pick_up[row][col] = pick_up[row-1][col]



    return pick_up[-1][-1]





items_pick(weights,values,C=110)
def apply_on_C(C):

    #apply items_pick on all knapsack instances from csv file

    for i in range(0,1500,3):

        weights = list(knapsack_instances.loc[i])

        values = list(knapsack_instances.loc[i+1])

        result = items_pick(weights,values,C)

#         print(result)

#         print([1 if x in result else 0 for x in range(50) ])

        knapsack_instances.loc[i+2] = [1 if x in result else 0 for x in range(50) ]  #write the result to the third line of each instance

        

    return knapsack_instances
knapsack_instances = apply_on_C(100)

knapsack_instances
knap_sizes = np.random.uniform(100,150,500)

knap_sizes

print(sum(knap_sizes)/len(knap_sizes))
def Evaluate_Sol(results):

    current_obj = 0

    for i in range(0,1500,3):

        sizes = list(results.loc[i, 0:49])

        rewards = list(results.loc[i+1, 0:49])

        solution = [bool(i) for i in list(results.loc[i+2, 0:49])]

        C = knap_sizes[int(i/3)]

#         print(C)

#         print(sizes)

#         print(solution)

        total_size = np.dot(sizes,solution)

        total_reward = np.dot(rewards,solution)

#         print(total_size)

        if total_size <= C:

            current_obj+=total_reward

        else:

#             print("total_size %d > C %d, drop" % (total_size,C))

            pass

    return current_obj
final_results = Evaluate_Sol(knapsack_instances)

final_results
knapsack_instances = apply_on_C(100)

knapsack_instances
result = Evaluate_Sol(knapsack_instances)

print(result)
knapsack_instances = apply_on_C(120)

knapsack_instances
result = Evaluate_Sol(knapsack_instances)

print(result)
C_results = {}

for i in range(100,150+1):

    knapsack_instances = apply_on_C(i)

    result = Evaluate_Sol(knapsack_instances)

    print(i,":", result)

    C_results[i] = result

    

C_results