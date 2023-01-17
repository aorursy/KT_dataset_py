# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
child_prefs = pd.read_csv('../input/child_wishlist.csv', header=None)

child_prefs = child_prefs.drop(0, axis=1).values



gift_prefs = pd.read_csv('../input/gift_goodkids.csv', header=None)

gift_prefs = gift_prefs.drop(0, axis=1).values
n_children = 1000000 # n children to give

n_gift_type = 1000 # n types of gifts available

n_gift_quantity = 1000 # each type of gifts are limited to this quantity

n_child_pref = 10 # number of gifts a child ranks

n_gift_pref = 1000 # number of children a gift ranks

twins = 4000

ratio_gift_happiness = 2

ratio_child_happiness = 2
def avg_normalized_happiness(pred, child_prefs, gift_prefs):

    

    # check if number of each gift exceeds n_gift_quantity

    gift_counts = Counter(elem[1] for elem in pred)

    for count in gift_counts.values():

        assert count <= n_gift_quantity

                

    # check if twins have the same gift

    for t1 in range(0,twins,2):

        twin1 = pred[t1]

        twin2 = pred[t1+1]

        assert twin1[1] == twin2[1]

    

    max_child_happiness = n_child_pref * ratio_child_happiness

    max_gift_happiness = n_gift_pref * ratio_gift_happiness

    total_child_happiness = 0

    total_gift_happiness = np.zeros(n_gift_type)

    

    for row in tqdm.tqdm(pred):

        child_id = row[0]

        gift_id = row[1]

        

        # check if child_id and gift_id exist

        assert child_id < n_children

        assert gift_id < n_gift_type

        assert child_id >= 0 

        assert gift_id >= 0



        child_happiness = (n_child_pref - np.where(child_prefs[child_id]==gift_id)[0]) * ratio_child_happiness

        if not child_happiness:

            child_happiness = -1



        gift_happiness = ( n_gift_pref - np.where(gift_prefs[gift_id]==child_id)[0]) * ratio_gift_happiness

        if not gift_happiness:

            gift_happiness = -1



        total_child_happiness += child_happiness

        total_gift_happiness[gift_id] += gift_happiness

    

    # print(max_child_happiness, max_gift_happiness

    print('normalized child happiness=',float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) , \

        ', normalized gift happiness',np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity))

    return float(total_child_happiness)/(float(n_children)*float(max_child_happiness)) + np.mean(total_gift_happiness) / float(max_gift_happiness*n_gift_quantity)
def pick_first_choice(child_pref, avail_gifts):

    

    # preference list (of remaining available gifts)

    overlap = set(child_pref) & set(avail_gifts)

    child_pref_available = [x for x in child_pref if x in overlap] # preserves pref order

    

    try: # first pick on the list

        return child_pref_available[0]

    except: # if prefered gifts aren't available, pick first available

        return avail_gifts[0]
gift_matches = []

gift_counter = np.zeros(n_gift_type)



for child in tqdm.tqdm(range(n_children)):



    if child < twins:

        if child % 2 == 0: # twin 1

            avail_gifts = np.where(gift_counter < n_gift_quantity-1)[0]

            chosen_gift = pick_first_choice(child_prefs[child], avail_gifts)

        else:

            # chosen_gift = chosen_gift # pick same as twin 1

            pass

        

    else: # not twins

        avail_gifts = np.where(gift_counter < n_gift_quantity)[0]

        chosen_gift = pick_first_choice(child_prefs[child], avail_gifts)



    gift_counter[chosen_gift] += 1

    gift_matches.append((child, chosen_gift))
avg_normalized_happiness(gift_matches, child_prefs, gift_prefs)
p = pd.DataFrame(gift_matches, columns=['ChildId', 'GiftId']).set_index('ChildId')

p.to_csv('nice_inversion_benchmark.csv')