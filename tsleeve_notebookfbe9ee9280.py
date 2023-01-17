# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

import json



"""

An example of a dish

{

"id": 47028,

"cuisine": "japanese",

"ingredients": [

  "melted butter",

  "matcha green tea powder",

  "white sugar",

  "milk",

  "all-purpose flour",

  "eggs",

  "salt",

  "baking powder",

  "chopped walnuts"

  ]

}

"""



# preprocess data

with open('../input/train.json') as f:

    trainData = json.load(f)

    allCuisines = {} # map from cuisine name to number

    allCuisinesList = []

    allIngredients = {} # map from ingredient name to number

    cntIngr = 0 # counter for ingredients

    for dish in trainData:

        # processing cuisine

        cuisine = dish['cuisine']

        if cuisine not in allCuisines:

            allCuisines[cuisine] = len(allCuisinesList)

            allCuisinesList.append(cuisine)

        dish['cuisine'] = allCuisines[cuisine]

        # processing ingredients

        ingredients = dish['ingredients']

        for idx, ingr in enumerate(ingredients):

            if ingr not in allIngredients:

                allIngredients[ingr] = cntIngr

                cntIngr += 1

            ingredients[idx] = allIngredients[ingr]

    trainDataMatrix = []

    for dish in trainData:

        row = [0] * cntIngr

        for ingr in dish['ingredients']:

            row[ingr] = 1

        trainDataMatrix.append(row)



with open('../input/test.json') as f:

    testData = json.load(f)

    testDataMatrix = []

    for dish in testData:

        ingredients = [ingr for ingr in dish['ingredients'] if ingr in allIngredients]

        row = [0] * cntIngr

        for ingr in ingredients:

            row[allIngredients[ingr]] = 1

        testDataMatrix.append(row)



# Plug in algorithm here

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(trainDataMatrix, [dish['cuisine'] for dish in trainData])

result = map(lambda i: allCuisinesList[i], clf.predict(testDataMatrix))



# Output in csv for submission on Kaggle

import csv

with open('submission.csv', 'wb') as f:

    writer = csv.writer(f)

    writer.writerow(('id', 'cuisine'))

    for i, ingr in zip([dish['id'] for dish in testData], result):

        writer.writerow((i, ingr))

    