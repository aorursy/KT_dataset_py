from itertools import product



import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

from numba import njit, prange



import santa_s_2019_faster_cost_function_24_s as utils

from santa_s_2019_faster_cost_function_24_s import build_cost_function
# Load Data

base_path = '/kaggle/input/santa-workshop-tour-2019/'

sub_path = '/kaggle/input/santa-ip/'

data = pd.read_csv(base_path + 'family_data.csv', index_col='family_id')

submission = pd.read_csv(sub_path + 'submission.csv', index_col='family_id')



# Build your "cost_function"

cost_function = build_cost_function(data)



# Run it on default submission file

original = submission['assigned_day'].values

original_score = cost_function(original)



%timeit cost_function(original)
def stochastic_product_search(top_k, fam_size, original, choice_matrix, 

                              disable_tqdm=False, verbose=10000,

                              n_iter=500, random_state=2019):

    """

    original (np.array): The original day assignments.

    

    At every iterations, randomly sample fam_size families. Then, given their top_k

    choices, compute the Cartesian product of the families' choices, and compute the

    score for each of those top_k^fam_size products.

    """

    

    best = original.copy()

    best_score = cost_function(best)

    

    np.random.seed(random_state)



    for i in tqdm(range(n_iter), disable=disable_tqdm):

        fam_indices = np.random.choice(range(choice_matrix.shape[0]), size=fam_size)

        changes = np.array(list(product(*choice_matrix[fam_indices, :top_k].tolist())))



        for change in changes:

            new = best.copy()

            new[fam_indices] = change



            new_score = cost_function(new)



            if new_score < best_score:

                best_score = new_score

                best = new

        

        if new_score < best_score:

            best_score = new_score

            best = new

    

        if verbose and i % verbose == 0:

            print(f"Iteration #{i}: Best score is {best_score:.2f}")

    

    print(f"Final best score is {best_score:.2f}")

    return best
choice_matrix = data.loc[:, 'choice_0': 'choice_9'].values
best = stochastic_product_search(

    choice_matrix=choice_matrix, 

    top_k=5,

    fam_size=5, 

    original=original, 

    n_iter=20000,

    disable_tqdm=False,

    verbose=2000

)
best = stochastic_product_search(

    choice_matrix=choice_matrix, 

    top_k=2,

    fam_size=12, 

    original=best, 

    n_iter=50000, 

    disable_tqdm=True,

    verbose=5000

)
best = stochastic_product_search(

    choice_matrix=choice_matrix, 

    top_k=8,

    fam_size=3, 

    original=best, 

    n_iter=50000, 

    disable_tqdm=False,

    verbose=5000

)
submission['assigned_day'] = best

final_score = cost_function(best)

submission.to_csv(f'submission_{final_score}.csv')