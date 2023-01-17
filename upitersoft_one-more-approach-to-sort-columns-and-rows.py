from six.moves import cPickle as pickle
import bz2

def loadPickleBZ(pickle_file):
    with bz2.BZ2File(pickle_file, 'r') as f:
        loadedData = pickle.load(f)
        return loadedData

def savePickleBZ(pickle_file, data):
    with bz2.BZ2File(pickle_file, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return
import time
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

data_path = "../input/"

print(os.listdir(data_path))

def loadData(fname):
    data = pd.read_csv(data_path+fname)
    names = data.columns.get_values()
    ids = data.values[:, 0]
    target = data.values[:, 1:2]
    values = data.values[:, 2:]
    return (names, ids, np.array(target, dtype=np.double), np.array(values, dtype=np.double))

loadtime = [-time.time()]
(names, ids, target, values) = loadData('santander-value-prediction-challenge/train.csv')
loadtime[0] += time.time()

target = target.reshape((-1,))

print('Loading Train done.', loadtime)
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import lightgbm as lgb
from sklearn.model_selection import KFold
from time import time
import scipy.misc
try:
    before = loadPickleBZ(data_path+'one-more-approach-to-sort-columns-and-rows/before.pbz')
    beforeM = before - np.transpose(before)
    print('before matrix loaded')
except:
    before = np.zeros((values.shape[1], values.shape[1]), dtype=np.int32)
    print('before matrix NOT loaded!')

sets = [[set()] for i in range(values.shape[0])]
# try:
#     sets = loadPickleBZ(data_path+'one-more-approach-to-sort-columns-and-rows/sets.pbz')
#     print('row sets loaded!')
# except:
#     print('row sets NOT loaded!')
## update before matrix
def setBefore(bef, aft):
    for b in bef:
        before[b, aft] += 1
    return

def valrow(n):
    return values[n]

_vset = []
## value set for row n
def vset(n):
    while len(_vset) <= n:
        _vset.append( None )

    if _vset[n] is None:
        vs = set(valrow(n))
        vs.remove(0)
        _vset[n] = vs

    return _vset[n]

## column indices from set S in row k
def indices(S, k):
    vals = valrow(k)
    idx = []
    for num in S:
        idx.extend(list(np.where(vals == num)[0]))
    return idx

used = np.zeros(values.shape[0], np.bool)


for i in range(values.shape[0]):
    ## remove early exit for better sorting matrix
#     if i > 100:
#         break
    if i % 10 == 1:
        print(i, end=' ')
#     else:
        continue

    if sets[i] is not None:
        sets[i][0] = sets[i][0] | {i}
        
    for j in range(values.shape[0]):
        if i == j:
            continue
            
        A = vset(i)
        B = vset(j)

        I = A & B

        ## calculate column order if rows share 50% of values
        if (len(I) > (len(A) + len(B))/2*0.5):
            
            ## add to sets if rows share 60% of values
            if (len(I) > (len(A) + len(B))/2*0.60):
                if sets[i] is not None:
                    sets[i][0] = sets[i][0] | {j}
                    sets[j] = None

            B = B - I
            A = A - I

            m = 0
            if target[i] in I:
                m += 1
            if target[j] in I:
                m += 2
            if target[i] in B:
                m += 4
            if target[j] in A:
                m += 8

            U = indices(A, i)
            V = indices(I, i)
            nbef = 0
            ntotal = 1e-30
            for u in U:
                for v in V:
                    ntotal += 1
                    if before[u, v] - before[v, u]>0:
                        nbef += 1
            befA = nbef/ntotal

            U = indices(B, j)
            V = indices(I, j)
            nbef = 0
            ntotal = 1e-30
            for u in U:
                for v in V:
                    ntotal += 1
                    if before[u, v] - before[v, u]>0:
                        nbef += 1
            befB = nbef/ntotal

#             print(i, j, '-', [m], '-', '0<1' if befA < befB else '1>0', target[i], target[j], '-', A, B)

            if m == 9 or m == 8:
#                 print('i in I, j in A')

                bef = indices(A, i)
                aft = indices(I, i)
                setBefore(bef, aft)

                bef = indices(I, j)
                aft = indices(B, j)
                setBefore(bef, aft)

            if m == 6 or m == 4:
#                 print('j in I, i in B')

                bef = indices(B, j)
                aft = indices(I, j)
                setBefore(bef, aft)

                bef = indices(I, i)
                aft = indices(A, i)
                setBefore(bef, aft)
                
print('Rows compare done')
import matplotlib.pyplot as plt
%matplotlib inline

savePickleBZ('before.pbz', before)
savePickleBZ('sets.pbz', sets)

def showBefore(before):
    before = np.array(before, dtype=np.float)
    before = before / (before + np.transpose(np.copy(before)) + 1e-30)
    before *= 255
    before = np.array(before, dtype=np.uint8)
    plt.figure(figsize=(16, 16))
    plt.imshow(before)
    
showBefore(before)
# show uncertain comparisions
beforeM = before - np.transpose(before)
bef_unc = (np.abs(beforeM) / (before + np.transpose(before) + 1e-30) < 0.3) * (beforeM != 0)
showBefore(bef_unc)
from functools import cmp_to_key

beforeM = before - np.transpose(before)

## Find out which set stands before another
def compareSets(A, B, i, j):
    I = A & B

    B = B - I
    A = A - I

    befA = 0
    befB = 0

    # if (target[i] in B) and (target[j] not in A):
    #     befB = 1
    # if (target[i] not in B) and (target[j] in A):
    #     befA = 1

    if befA == befB:
        U = indices(A, i)
        V = indices(I, i)
        nbef = 0
        ntotal = 1e-30
        for u in U:
            ntotal += len(V)
            nbef += (beforeM[u, V] > 0).sum()
        befA = nbef / ntotal

        U = indices(B, j)
        V = indices(I, j)
        nbef = 0
        ntotal = 1e-30
        for u in U:
            ntotal += len(V)
            nbef += (beforeM[u, V] > 0).sum()
        befB = nbef / ntotal


    if befA == befB:
        U = indices(I, i)

        nbef = 0
        ntotal = 1e-30
        for z in A|I:
            V = indices({z}, i)
            for v in V:
                ntotal += len(U)
                nbef += (beforeM[U, v] > 0).sum()
        befB = nbef / ntotal

        U = indices(I, j)

        nbef = 0
        ntotal = 1e-30
        for z in B|I:
            V = indices({z}, j)
            for v in V:
                ntotal += len(U)
                nbef += (beforeM[U, v] > 0).sum()
        befA = nbef / ntotal

    return A, B, I, befA, befB

if True:
    targetFound = [0,0,1e-30]
    sum_sle = [0,0]
    orders = []

    for st in sets:
        if st is None:
            continue
        order = list(st[0])
        orders.append(order)

    for ordI, order in enumerate(orders):
        if ordI > 400:
            break
        if len(order)<=1:
            continue
        print('\n-----------------\nloop: ', ordI, '/', len(orders), len(order), 'found', targetFound[0] / targetFound[2], targetFound[1] / targetFound[2])

        def sortOrder():
            print('sorting... ', len(order), end=' ')

            comparisions = [0]
            def rowCompare(i, j):
                comparisions[0] += 1
                if comparisions[0] % 500 == 0:
                    print('cmps =', comparisions[0], end=' ')
                    
                A = vset(i)
                B = vset(j)
                A, B, I, befA, befB  = compareSets(A, B, i, j)
                return befA - befB
            
            if len(order) > 50:
                order.sort(key = cmp_to_key(rowCompare))            
            else:
                ## Bubble sort is not a best choise here...
                for oi in range(len(order)):
                    print(oi, end=' ')
                    for oj in range(oi+1, len(order)):

                        i = order[oi]
                        j = order[oj]

                        A = vset(i)
                        B = vset(j)

                        A, B, I, befA, befB  = compareSets(A, B, i, j)
                        comparisions[0] += 1

                        if befA > befB:
                            order[oi] = j
                            order[oj] = i

                        if befA == befB and np.random.random()<0.5:
                            order[oi] = j
                            order[oj] = i
                            
            print('comparisions: ', comparisions[0])
                            

        sortOrder()
        resort = False
        for k in range(2):
            if resort:
                sortOrder()
            resort = False
            for oi in range(len(order)-1):
                def ijAB(oi, oip1 = None):
                    if oip1 is None:
                        oip1 = oi+1
                    i = order[oi]
                    j = order[oip1]
                    A = vset(i)
                    B = vset(j)
                    return i, j, A, B


                i, j, A, B = ijAB(oi)
                A, B, I, befA, befB = compareSets(A, B, i, j)

                if befA < befB:
                    targetFound[2] += 1
                    if target[i] in I:
                        targetFound[0] += 1
                    if target[i] in B:
                        targetFound[1] += 1
                else:
                    resort = True

                def whre(k):
                    c = 'X'
                    if target[k] in A:
                        c = 'A'
                    if target[k] in B:
                        c = 'B'
                    if target[k] in I:
                        c = 'I'
                    return c

#                 print(i, j, '-', '0=0' if befA == befB else '0<1' if befA < befB else '1>0', whre(i)+whre(j), '\t', target[i], target[j]) ## '-', A, B, I

            print('\tfound target in I {} or B {} set'.format(targetFound[0] / targetFound[2], targetFound[1] / targetFound[2]))

            if resort:
                print('\t------ Resort')
                continue
            else:
                break
    
    print('Finished')
