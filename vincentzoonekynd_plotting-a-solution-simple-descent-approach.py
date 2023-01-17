import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

from collections import Counter

from tqdm import tqdm, tqdm_notebook
d = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv')

sample_submission = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv')



N_DAYS = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125



family_size_dict = d[['n_people']].to_dict()['n_people']

cols = [f'choice_{i}' for i in range(10)]

choice_dict = d[cols].to_dict()

days = list(range(N_DAYS,0,-1))
## Faster cost function, from https://www.kaggle.com/nickel/250x-faster-cost-function-with-numba-jit



from numba import njit



prediction = sample_submission['assigned_day'].values

desired = d.values[:, 1:-1]

family_size = d.n_people.values

penalties = np.asarray([

    [

        0,

        50,

        50 + 9 * n,

        100 + 9 * n,

        200 + 9 * n,

        200 + 18 * n,

        300 + 18 * n,

        300 + 36 * n,

        400 + 36 * n,

        500 + 36 * n + 199 * n,

        500 + 36 * n + 398 * n

    ] for n in range(family_size.max() + 1)

])



@njit()

def jited_cost(prediction, desired, family_size, penalties):

    N_DAYS = 100

    MAX_OCCUPANCY = 300

    MIN_OCCUPANCY = 125

    penalty = 0

    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)

    for i in range(len(prediction)):

        n = family_size[i]

        pred = prediction[i]

        n_choice = 0

        for j in range(len(desired[i])):

            if desired[i, j] == pred:

                break

            else:

                n_choice += 1

        

        daily_occupancy[pred - 1] += n

        penalty += penalties[n, n_choice]



    accounting_cost = 0

    n_out_of_range = 0

    constraint = 0

    daily_occupancy[-1] = daily_occupancy[-2]

    for day in range(N_DAYS):

        n_next = daily_occupancy[day + 1]

        n = daily_occupancy[day]

        n_out_of_range += (n > MAX_OCCUPANCY) or (n < MIN_OCCUPANCY)

        constraint += 1_000_000 * max( 0, n - MAX_OCCUPANCY, MIN_OCCUPANCY - n )

        diff = abs(n - n_next)

        accounting_cost += max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))



    total = penalty + accounting_cost + constraint

    return np.asarray([total, penalty, n_out_of_range * 1_000_000, accounting_cost, n_out_of_range])
def plot_occupancy(prediction):

    occupancy = {k:0 for k in days}

    for family, day in enumerate(prediction):

        occupancy[day] += family_size_dict[family]    

    x = occupancy.keys()

    y = occupancy.values()

    z = [ 'C0' if u >= MIN_OCCUPANCY and u <= MAX_OCCUPANCY else 'C1' for u in y ]

    fig, ax = plt.subplots(figsize=(20,5))

    ax.axhline(MIN_OCCUPANCY, color='black')

    ax.axhline(MAX_OCCUPANCY, color='black')

    ax.bar(x, y, color=z)

    ax.set_ylabel( "Occupancy" )

    ax.set_xlabel( "Day" )

    plt.show()
def plot_choices(prediction):

    m = np.zeros( (11,len(prediction)) )

    for family, day in enumerate(prediction):

        choices = [ f"choice_{i}" for i in range(10) ]

        choices = [ choice_dict[c][family] for c in choices ]

        i = np.where( [ day == c for c in choices + [day] ] )[0][0]

        m[i,family] = 1

        

    # Re-order the families by day

    i = np.argsort(prediction)

    m = m[ : , i]

    

    fig, ax = plt.subplots( figsize=(20,4))

    ax.pcolor(m)

    ax.set_yticks(np.arange(m.shape[0]) + 0.5, minor=False)

    ax.set_yticklabels( list(range(10)) + ['other'], minor=False)

    ax.set_xlabel("Family (in chronological order)") 

    ax.set_ylabel("Choice")

    plt.show()
def get_accounting_cost(prediction, desired, family_size, penalties):

    N_DAYS = 100

    MAX_OCCUPANCY = 300

    MIN_OCCUPANCY = 125

    penalty = 0

    

    daily_occupancy = np.zeros(N_DAYS + 1, dtype=np.int64)

    for i in range(len(prediction)):

        daily_occupancy[prediction[i] - 1] += family_size[i]



    accounting_costs = np.zeros(N_DAYS)

    daily_occupancy[-1] = daily_occupancy[-2]

    for day in range(N_DAYS):

        n_next = daily_occupancy[day + 1]

        n = daily_occupancy[day]

        diff = abs(n - n_next)

        accounting_costs[day] = max(0, (n-125.0) / 400.0 * n**(0.5 + diff / 50.0))



    return accounting_costs



def plot_accounting_costs(prediction):

    accounting_costs = get_accounting_cost(prediction, desired, family_size, penalties)

    fig, ax = plt.subplots( figsize=(20,4))

    ax.bar( range(N_DAYS), accounting_costs )

    ax.set_xlabel("Day")

    ax.set_ylabel("Accounting cost")

    plt.show()
# Random, uniform solution

print( jited_cost(sample_submission['assigned_day'].values, desired, family_size, penalties)[0] )

plot_occupancy(sample_submission['assigned_day'])

plot_choices(sample_submission['assigned_day'])

plot_accounting_costs(sample_submission['assigned_day'])
# Grant everyone their first wish



cols = [f'choice_{i}' for i in range(10)]

choices = d[cols].T.to_dict()

choices = { k: list(v.values()) for k,v in choices.items() }

solution = np.array( [ choices[i][0] for i in range( sample_submission.shape[0] ) ] )



print( jited_cost(solution, desired, family_size, penalties)[0] )

plot_occupancy(solution)

plot_choices(solution)

#plot_accounting_costs(solution) # Too high
best = sample_submission['assigned_day'].values

best_cost = jited_cost(best, desired, family_size, penalties)[0]
N = 100_000   # Increase this

with tqdm_notebook(total=N) as pbar:

    for _ in range(N):

        candidate = best.copy()



        # Pick a neighbouring solution, for various notions of "neighbourhood":

        # - Reassign a family, but only among its choices

        # - Reassign a family, anywhere

        # - Swap two families 

        

        p = [.98, .01, .01] # Play with those probabilities

        which = np.random.choice(3, p=p)



        if which == 0:

            # Re-assign one family, but only among its choices

            i = np.random.choice( len(candidate) )

            k = np.random.choice( choices[i] )

            candidate[i] = k



        if which == 1:

            # Re-assign one family, anywhere

            i = np.random.choice( len(candidate) )

            k = np.random.choice( N_DAYS ) + 1

            candidate[i] = k



        if which == 2:

            # Swap two families

            i1 = np.random.choice( len(candidate) )

            i2 = np.random.choice( len(candidate)-1 )

            if i2 >= i1:

                i2 = i2 + 1

            k = candidate[i1]

            candidate[i1] = candidate[i2]

            candidate[i2] = k



        #cost = cost_function(candidate)

        cost = jited_cost(candidate, desired, family_size, penalties)

        if cost[0] < best_cost:

            best = candidate

            best_cost = cost[0]

            pbar.set_description(f"{round(cost[0])} ({int(cost[4])}) {which}")    

        pbar.update(1)
print( jited_cost(best, desired, family_size, penalties)[0] )

plot_occupancy(best)

plot_choices(best)

plot_accounting_costs(best)