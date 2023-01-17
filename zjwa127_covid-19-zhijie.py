# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import pandas as pd

from tqdm import tqdm

import re

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

style.use("ggplot")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
docs = []

dirs = ["biorxiv_medrxiv","comm_use_subset","noncomm_use_subset","custom_license"]

for d in dirs:

    for file in tqdm(os.listdir(f"/kaggle/input/CORD-19-research-challenge/{d}/{d}")):

        file_path = f"/kaggle/input/CORD-19-research-challenge/{d}/{d}/{file}"

        j = json.load(open(file_path, "rb"))

        title = j['metadata']['title']

        try:

            abstract = j['abstract'][0]

        except:

            abstract=""



        full_text = ""

        for text in j['body_text']:

            full_text += text['text']+'\n\n'

        

        docs.append([title,abstract,full_text])

df = pd.DataFrame(docs, columns=['title','abstract','full_text'])

incubation = df[df['full_text'].str.contains('incubation')]

#print(incubation.head())

incubation_times = []

texts = incubation['full_text'].values

for t in texts:

    for sentence in t.split(". "):

        if "incubation" in sentence:

            single_day = re.findall(r" \d{1,2} day",sentence)

            if len(single_day) == 1:

                #print(single_day[0])

                #print(sentence)

                num = single_day[0].split(" ")

                incubation_times.append(float(num[1]))

                #print()

                #print()
plt.hist(incubation_times, bins=50)

print(f"The mean projected incubation time is {np.mean(incubation_times)} days")

print(f"Went through {len(incubation_times)} articles.")

plt.ylabel("bin counts")

plt.xlabel("incubation time (days)")

plt.show()