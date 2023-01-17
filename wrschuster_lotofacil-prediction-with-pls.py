# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
hf = pd.read_excel('/kaggle/input/brazilian-lottery-loto-fcil/LotoFacil_DataSet.xlsx')
hf.tail()
hf = hf.set_index('Concurso')
hf.tail()
hf.info()
#Importing the funcions to perform PLS Regression on our data set



from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



import matplotlib.pyplot as plt
X = hf.iloc[: , 16:41] # X containing just the "C" - Cumulative sum of the numbers

y = hf['R5'] # y containing the target cumulative sum for the nest draw for the number 3
n_comp = np.arange(1,26,1)



score = []

for n in n_comp:

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    reg = PLSRegression(n_components= n)

    reg.fit(X_train, y_train)

    sc = r2_score(y_test, reg.predict(X_test))

    score.append(sc)



plt.xlabel('Número de componentes')

plt.ylabel('Score')

plt.plot(n_comp, score)
X_count = hf[['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 

           'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25']]

targets = ['R1','R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12','R13', 'R14', 

           'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25']
#Modelling the answer of the dataset for each target (R's) using 25 components



for n in targets:

    X_train, X_test, y_train, y_test = train_test_split (X_count, hf[n], random_state = 0)

    reg = PLSRegression(n_components = 25)

    fit = reg.fit(X_train, y_train)

    sc = r2_score(y_test, reg.predict(X_test))

    mse = mean_squared_error(y_test, reg.predict(X_test))

    print('Score {}: {}' .format(n, sc))

    print('MSE {}: {}' .format(n, mse))
X_ = X_count.iloc[0:-1, :] # Draws 0 up to 2027

X_pred = X_count.iloc[2027:] # Draw 2028 to be predicted
#Predicting the answer for the selected draw and saving in 'result.xlsx' file

result = []

for r in targets:

    X_train, X_test, y_train, y_test = train_test_split(X_count, hf[r], random_state = 0)

    pls = PLSRegression(n_components = 25)

    pls.fit(X_train, y_train)

    prediction = pls.predict(X_pred)[0]

    result.append(prediction[0])

final = pd.DataFrame(result, index = targets, columns = ['Predicted'])
final # Predicted cumulative sum with number of the draw 2028. This are the cumulative that should happen in the draw 2029.
# Now, let's check what are the numbers to be played.

X_pred_answer = hf.iloc[2027: ,41:]
X_transposed = X_pred_answer.T
result = X_transposed.merge(final, how = 'inner', right_index = True, left_index = True)
result
result['play'] = round(result['Predicted'] - result[2028])
result