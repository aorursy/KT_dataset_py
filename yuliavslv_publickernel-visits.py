import numpy

import pandas

from sklearn.utils.extmath import weighted_mode



inp = pandas.read_csv('../input/train.csv')

table = inp.get('visits')
num_week = 157

array = list()

for i in range(num_week):

    tmp = [j for j in range(7 * i + 1, 7 * i + 8)]

    array.append(tmp)      
delta = 1

sum = 0

for i in range(num_week):

    for j in range(7):

        array[i][j] = ((i + 1) / num_week) ** delta

        sum += array[i][j]

for i in range(num_week):

    for j in range(7):

        array[i][j] = array[i][j] / sum
visits, weight = list(), list()

for i in table:

    tmp1, tmp2 = list(), list()

    vis = i[1:].split(' ')

    for j in vis:

        tmp1.append((int(j) - 1) % 7 + 1)

        tmp2.append(array[(int(j) - 1) // 7][(int(j) - 1) % 7])

    visits.append(tmp1)

    weight.append(tmp2)
res = list()

for i in range(len(visits)):

    w = weighted_mode(visits[i], weight[i])

    res.append(' ' + str(int(w[0][0])))
solution = pandas.DataFrame(columns = ['id', 'nextvisit'])

solution['id'] = inp.get('id')

solution['nextvisit'] = res

solution.to_csv('solution.csv', index = False, sep = ',')