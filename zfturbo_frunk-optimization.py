import ast
import numba
import string

import numpy as np
import pandas as pd

from itertools import combinations, permutations, product
from math import sqrt, factorial
from sklearn.neighbors import KDTree
from sympy import isprime, primerange
from tqdm import tqdm_notebook
K = 4#!
def frunkopt_func_generator(K):
    K = K - 1 # Dont look like that, K-Opt means, 
              # that we will remove K links between 2K points 
              # p1 - p2, p3 - p4, ..., p2K-1 - p2K
              # so we will get K-1 unmoved segments
              # p2..p3, p4..p5, ..., p2K-2..p2K-1
    letters = np.array(list(string.ascii_uppercase[:K]))
    i = 0
    tab = '    '
    function_name = f'move_{K}opt'
    function_strings = []
    function_strings.append("@numba.jit('void(i8[:], i8[:], i8)', nopython=True, parallel=False)")
    function_strings.append(f"def {function_name}(path, idx, move_type):")
    function_strings.append("{}pslice = slice(idx[0]+1, idx[-1]+1)".format(tab))
    function_strings.append("{}{} = {}".format(tab, ', '.join(letters), ', '.join([f'path[idx[{t}]+1:idx[{t+1}]+1]' for t in np.arange(K)])))
    function_strings.append("{}{} = {}".format(tab, ', '.join(string.ascii_lowercase[:K]), '[::-1], '.join(letters) + '[::-1]'))
    for p in permutations(np.arange(K)):
        for r in product([0, 1], repeat=K):
            out_arr = [letters[j].lower() if r[j] else letters[j] for j in p]
            function_strings.append('{}{} move_type == {}:'.format(tab, 'elif' if i else 'if', i))
            function_strings.append("{}path[pslice] = np.concatenate(({}))".format(tab * 2, ', '.join(out_arr)))
            i += 1
    return function_name, """{}""".format('\n'.join(function_strings))
frunktion_name, frunktion_body = frunkopt_func_generator(K)
%%time
exec(compile(ast.parse(frunktion_body), '<string>', mode='exec'))
frunkopt_move = locals()[frunktion_name]
cities = pd.read_csv(
    '../input/traveling-santa-2018-prime-paths/cities.csv', 
    index_col=['CityId'])

XY = np.stack(
    (
        cities.X.astype(np.float32), 
        cities.Y.astype(np.float32)
    ), 
    axis=1)

is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)
@numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
def cities_dist(id_from, id_to, offset):
    xy_from, xy_to = XY[id_from], XY[id_to]
    dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
    distance = sqrt(dx * dx + dy * dy)
    if offset % 10 == 9 and is_not_prime[id_from]:
        return 1.1 * distance
    return distance

@numba.jit('f8(i8[:], i8)', nopython=True, parallel=False)
def chunk_score(chunk, offset):
    pure_distance, penalty = 0.0, 0.0
    penalty_modulo = 9 - offset % 10
    for path_index in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[path_index], chunk[path_index+1]
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        pure_distance += distance
        if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
            penalty += distance
    return pure_distance + 0.1 * penalty


@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def path_score(path):
    return chunk_score(path, 0)

def path_score_full(path):
    pure_distance, penalty = 0.0, 0.0
    penalty_modulo = 9
    for path_index in numba.prange(path.shape[0] - 1):
        id_from, id_to = path[path_index], path[path_index + 1]
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        pure_distance += distance
        if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
            penalty += distance
    return (
        round(pure_distance, 4), 
        round(0.1 * penalty, 4), 
        round(pure_distance + 0.1 * penalty, 4)
    )
N_NEIGHBORS = 2*K + 2
RADIUS = 3*K + 3
kdt = KDTree(XY[1:])
neighbors_N = kdt.query(XY[1:], N_NEIGHBORS, return_distance=False)[:, K:]
neighbors_R = kdt.query_radius(XY[1:], RADIUS, count_only=False, return_distance=False)
neighbors = set()

for city_id in tqdm_notebook(cities.index[1:]):
    for neib_triplet in combinations(neighbors_N[city_id - 1] + 1, K):
        neighbors.add(tuple(sorted(neib_triplet)))
        
    for neib_triplet in combinations(neighbors_R[city_id - 1][:N_NEIGHBORS-K] + 1, K):
        neighbors.add(tuple(sorted(neib_triplet)))
    
print(f'{len(neighbors)} cities {K}-neighbors are selected.')
@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def sum_distance(ids):
    res = 0
    for i in numba.prange(len(ids)):
        for j in numba.prange(i + 1, len(ids)):
            res += cities_dist(ids[i], ids[j], 0)
    return res
neighbors = np.array(list(neighbors))
distances = np.array(list(map(sum_distance, tqdm_notebook(neighbors))))
order = distances.argsort()
neighbors = neighbors[order]
initial_path = pd.read_csv('../input/not-a-5-and-5-halves-opt-0efc12/1515559.6779840707.csv').Path.values
path = initial_path.copy()
path_index = np.argsort(path[:-1])
initial_score = total_score = path_score(path)

print(path_score_full(path))
print(f'Total score is {path_score(path):.2f}.')
frunkopt_moves_count = factorial(K - 1) * 2 ** (K - 1)
print(frunkopt_moves_count)
runs = 2
neighbors_len = len(neighbors)
cases_improved = np.zeros((runs, frunkopt_moves_count))
def make_submission(name, path):
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)
for run in np.arange(1, runs + 1):
    magic_number = 4223 * K * N_NEIGHBORS * RADIUS // run
    print(f'Run #{run} | {magic_number}')
    for step, ids in tqdm_notebook(enumerate(neighbors[:magic_number], 1), total=magic_number):
        if step % 10 ** 6 == 0:
            new_total_score = path_score(path)
            last_improvement = total_score - new_total_score
            print(f"score {new_total_score:.2f} | last 10^6 {last_improvement:.2f} | total {initial_score - new_total_score:.2f}.")
            print(cases_improved[run - 1])
            total_score = new_total_score
            if last_improvement > K:
                make_submission(f'frunkopt_{path_score(path):.4f}', path)

        idx = sorted(path_index[ids])
        new_idx = idx - idx[0]

        pslice = slice(idx[0], idx[-1] + 2)
        chunk = path[pslice]
        best_score = chunk_score(chunk, idx[0])
        best_move = -1
        
        for move_type in numba.prange(1, frunkopt_moves_count): # since move_type == 0 will not change chunk
            new_chunk = chunk.copy()
            frunkopt_move(new_chunk, new_idx, move_type)
            new_score = chunk_score(new_chunk, idx[0])
            if new_score < best_score:
                best_score = new_score
                best_move = move_type
                best_chunk = new_chunk.copy()
        
        if best_move > -1:
            path[pslice] = best_chunk
            path_index = np.argsort(path[:-1])
            cases_improved[run - 1, best_move] += 1
print(cases_improved)
total_score = path_score(path)
print(f'Total improvement | {initial_score - total_score:.2f}')
print('Final scores |', path_score_full(path))
make_submission(f'final_frunkopt_{path_score(path):.4f}', path)