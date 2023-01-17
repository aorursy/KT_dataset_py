from IPython.display import HTML

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
import tensorflow as tf
questions = pd.read_csv('../input/Questions.csv', encoding='latin1',nrows=1000)

answers = pd.read_csv('../input/Answers.csv', encoding='latin1',nrows=1000)
len(questions)
questions.head()
question=questions.ix[0]
HTML(question.Body)
a=answers[answers.ParentId==469]


HTML(a.Body[0])