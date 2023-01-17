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
import json

with open('/kaggle/input/resume-entities-for-ner/Entity Recognition in Resumes.json','r') as j_file:

    data = [json.loads(f_line) for f_line in j_file.readlines()]



resumes = [resume['content'] for resume in data]

resume_df = pd.DataFrame(resumes, columns=['resume_text'])

resume_df.head()