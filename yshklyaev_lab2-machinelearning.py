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
import numpy as np





eps = 0.05

eps_2 = 0.05



def functionTwo(w,b,a):

    return (np.log(np.exp(w+b)/np.exp(a)))



def function(w ,b ):

    return (np.log(1 / (1 + np.exp(w + b))))



def diff_dw2(w, b,a):

    return (((functionTwo(w + eps, b,a) - functionTwo(w, b,a)) / eps))





def diff_db2(w, b, a):

    return ((functionTwo(w, b + eps,a) - functionTwo(w, b, a)) / eps)

def diff_dw(w, b):

    return ((function(w + eps, b) - function(w, b)) / eps)





def diff_db(w, b):

    return ((function(w, b + eps) - function(w, b)) / eps)







def grad(w, b ):

    return np.sqrt((w ** 2) + (b ** 2))



def GradientDownTwo(x, w, b, a, h):

    iter = 0

    w = w*x

    base_w = w

    base_b = b

    test_dw = diff_dw(base_w, base_b)

    test_db = diff_db(base_w, base_b)

    print('start x1: ', w, 'x2: ', b, 'func_x: ', functionTwo(base_w, base_b ,a), 'diff_w: ', test_dw, 'diff_b: ',

          test_db)

    print()

    while (iter < 10):

        print('step: ', iter)

        dw = diff_dw2(w, b, a)

        db = diff_db2(w, b, a)

        print('dw: ', dw)

        print('db: ', db)

        gradient = grad(dw, db)

        print('grad:  ', gradient)

        if (abs(gradient) < eps):

            print('final: ', w, b, functionTwo(w, b, a))

            return functionTwo(w, b, a)

        else:

            w = w - h * dw

            b = b - h * db

            print("w: ", w, "b: ", b)

            print("function: ", functionTwo(w, b,a))



        if (functionTwo(w, b,a) - functionTwo(base_w, base_b,a) < 0):

            if (((abs(w - base_w) and abs(b - base_b)) < eps_2) and (

                    functionTwo(w, b,a) - functionTwo(base_w, base_b,a))< eps_2):

                print('final: ', w, b, functionTwo(w, b,a))

                return functionTwo(w, b, a)

        else:

            print('h ', h / 2)

            h = h / 2

            w = w - h * dw

            b = b - h * db

        iter += 1



    return functionTwo(w, b, a)



def GradientDown(x, w, b,  h):

    iter = 0

    w = w*x

    base_w = w

    base_b = b

    test_dw = diff_dw(base_w, base_b)

    test_db = diff_db(base_w, base_b)

    print('start x1: ', w, 'x2: ', b, 'func_x: ', function(base_w, base_b), 'diff_w: ', test_dw, 'diff_b: ',

          test_db)

    print()

    while (iter < 10):

        print('step: ', iter)

        dw = diff_dw(w, b)

        db = diff_db(w, b)

        print('dw: ', dw)

        print('db: ', db)

        gradient = grad(dw, db)

        print('grad:  ', gradient)

        if (abs(gradient) < eps):

            print('final: ', w, b, function(w, b))

            break

        else:

            w = w - h * dw

            b = b - h * db

            print("w: ", w, "b: ", b)

            print("function: ", function(w, b))



        if (function(w, b) - function(base_w, base_b) < 0):

            if (((abs(w - base_w) and abs(b - base_b)) < eps_2) and (

                    function(w, b) - function(base_w, base_b))< eps_2):

                print('final: ', w, b, function(w, b))

                print()

        else:

            print('h ', h / 2)

            h = h / 2

            w = w - h * dw

            b = b - h * db

        iter += 1



    return function(w, b)



def binary_logistic_regression():

    w = np.linspace(0,1,10)

    w = np.resize(w, (w.size,1))



    x = np.array([1,2,3,4,5,1,2,3,4,5])

    print(x)

    h = 0.1

    b = 6

    q = []

    temp = 0.0

    iter = 0

    for i in range(w.size):

        iter+=1

        print('iteration',iter)



        print('temp',temp)

        q.append(GradientDown( x[i],w[i], b,  h))



    print('q',q)

    print(-1/w.size*np.sum(q))



def non_binary_logistic_regression():

    x = np.array([1, 2, 3])

    x = np.resize(x,(x.size,1))

    b = np.array([3,5,1,4])

    b = np.resize(b,(b.size,1))

    w = np.linspace(0,1,(x.size*b.size)).reshape(x.size,b.size)



    q = []

    d = []

    h = 0.1

    for j in range(b.size):

        for i in range(x.size):

            d.append(x[i]*w[i][j] + b[j])

    a = np.array(d)

    a = np.resize(a,(x.size,b.size))

    for j in range(b.size):

        for i in range(x.size):

            q.append(GradientDownTwo(x[i], w[i][j], b[j], a[i][j], h))

        print(q)

        print(-1/len(q)*(np.sum(q)))





def main():

    #binary_logistic_regression()

    non_binary_logistic_regression()



if __name__ == '__main__':

    main()