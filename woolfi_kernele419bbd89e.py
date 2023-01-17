import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



print("*** data reading ***")



testX = [[0] * 8 for i in range(300000)]

Y = []



train = pd.read_csv('../input/train.csv')

train = train.values



for j in range(len(train)):

    vizits = np.array(list(map(int, train[j][1].split())))



    for i in range(0, vizits.size):

        m = vizits[i] % 7

        testX[j][(7 if m == 0 else m)] += (vizits[i] + 1) / 1100
print("*** predicting ***")



for j in range(len(testX)):

    max_ = testX[j][0] 

    imax = 0;

    

    for i in range(len(testX[j])):

        if testX[j][i] > max_:

            max_ = testX[j][i]

            imax = i

    

    Y += [imax]
with open('answers.csv', mode='w') as file:

    file.write('id,nextvisit\n')

        

    for i in range(len(Y)):

        file.write('{0}, {1}\n'.format(i + 1, Y[i]))