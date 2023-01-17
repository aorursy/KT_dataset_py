import time
import pandas as pd
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from itertools import chain
from itertools import combinations
from itertools import permutations
from itertools import product
from sklearn.neighbors import NearestNeighbors
def initial():
    df = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv')
    df['Z'] = 1 + .1 * ~df['CityId'].apply(sp.isprime)
    data = df[['X', 'Y', 'Z']].values
    tour = np.loadtxt('../input/traveling-santa-lkh-solution/pure1502650.csv', 
                      skiprows=1, dtype=int)
    return tour, data

tour, data = initial()
data[:5]
tour[:5]
def distance(tour, data, pen=9):
    xy, z = np.hsplit(data[tour], [2])
    dist = np.hypot(*(xy[:-1] - xy[1:]).T)
    dist[pen::10] *= z[:-1][pen::10].flat
    return dist

dist = distance(tour, data)
dist
distance(tour[3:9], data, pen = 9 - 3 % 10) == dist[3:8]
f'Initial Score: {np.sum(dist):.2f}'
def candidates(data, opt, ext):
    nns = NearestNeighbors(n_neighbors=opt + ext).fit(data[:, :2])
    kne = nns.kneighbors(data[:, :2], return_distance=False)
    np.random.shuffle(kne)
    cand = set()
    for i in kne:
        for j in combinations(i[1:], opt - 1):
            cand.add(tuple(sorted((i[0],) + j)))
    return cand
list(candidates(data, opt=3, ext=0))[:5]
len(candidates(data, opt=2, ext=0))
len(candidates(data, opt=2, ext=1))
def alternatives(tour, cuts, fil):
    edges = [tuple(x) for x in np.split(tour, cuts)[1:-1]]
    a, b = tour[cuts[0] - 1], tour[cuts[-1]]
    alter = set()
    for i in set(product(*zip(edges, [x[::-1] for x in edges]))):
        for j in permutations(i):
            if not fil or all(x != y for x, y in zip(edges, j)):
                alter.add(tuple(chain((a,), *j, (b,))))
    alter.discard(tuple(chain((a,), *edges, (b,))))
    return alter
# edges
tour[2:5], tour[5:7]
# a, b
tour[1] , tour[7]
alternatives(tour, cuts = [2,5,7], fil=False)
alternatives(tour, cuts = [2,5,7], fil=True)
def submit(tour):
    np.savetxt('submission.csv', tour, fmt='%d', header='Path', comments='')
def kopt(opt, ext, fil):
    tour, data = initial()
    sequ = 1 + np.argsort(tour[1:])
    dist = distance(tour, data)
    print(f'opt:{opt} & ext:{ext} & fil:{fil} ...')
    cand = candidates(data, opt, ext)
    print(f' Initial Score:\t{np.sum(dist):0.2f}')
    for c in cand:
        cuts = sorted(sequ[j] for j in c)
        alter = alternatives(tour, cuts, fil)
        if not alter:
            continue
        atour, pen = np.array(list(alter)), 9 - (cuts[0] - 1) % 10
        adist = np.array([distance(x, data, pen) for x in atour])
        if np.any(np.sum(adist, 1) < np.sum(dist[cuts[0] - 1:cuts[-1]])):
            arg = np.argmin(np.sum(adist, 1))
            dist[cuts[0] - 1:cuts[-1]] = adist[arg]
            tour[cuts[0]:cuts[-1]] = atour[arg][1:-1]
            sequ[atour[arg][1:-1]] = range(cuts[0], cuts[-1])
    print(f' Final Score:\t{np.sum(dist):0.2f}')
    submit(tour)
t0 = time.time()
kopt(opt=2, ext=0, fil=False)
print(f'Time:\t{time.time()-t0:.2f}s')
t0 = time.time()
kopt(opt=2, ext=1, fil=False)
print(f'Time:\t{time.time()-t0:.2f}s')
t0 = time.time()
kopt(opt=3, ext=0, fil=False)
print(f'Time:\t{time.time()-t0:.2f}s')
t0 = time.time()
kopt(opt=4, ext=0, fil=True)
print(f'Time:\t{time.time()-t0:.2f}s')
def graph():
    tour, data = initial()
    xy = data[tour][:, :2]
    segm = np.hstack((xy[:-1], xy[1:])).reshape(-1, 2, 2)
    lc = mcoll.LineCollection(segments=segm,
                              array=np.linspace(0, 1, len(segm)),
                              cmap=plt.get_cmap('Spectral'),
                              lw=.9)
    fig, ax = plt.subplots(figsize=(10,8))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.add_collection(lc)
    ax.plot(*xy.T, lw=.3, c='black')
    plt.show()
    
graph()