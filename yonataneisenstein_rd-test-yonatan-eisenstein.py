import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df1 = pd.read_fwf('/kaggle/input/rd-test/RD test 3.txt')
df1
name1 = df1['CURRICULUM VITAE'][1]
name1
import re

name1 = re.sub(r'^.*?S', 'S', name1)
name1
phone1 = df1['CURRICULUM VITAE'][8]
phone1
phone1 = ''.join(x for x in phone1 if x.isdigit() or x == '+')
phone1
Email1 = None
website1 = df1['CURRICULUM VITAE'][9]
website1
website1 = re.sub(r'^.*?w', 'w', website1)
website1
date_of_birth1 = df1['CURRICULUM VITAE'][4]
date_of_birth1
date_of_birth1 = re.sub(r'^.*?2', '2', date_of_birth1)
date_of_birth1
output1 = pd.DataFrame({'name':[name1], 'phone':[phone1], 'Email': [Email1], 'website': [website1], 'date_of_birth': [date_of_birth1]})
output1
output1 = output1.T
output1
df2 = pd.read_fwf('/kaggle/input/rd-test/RD test 4.txt')
df2
name2 = df2.columns[0]
name2
phone2 = df2['James Smith'][6]

phone2 = ''.join(x for x in phone2 if x.isdigit() or x == '+')
phone2
Email2 = df2['James Smith'][4]
Email2
website2 = df2['Unnamed: 2'][5]
website2 = re.sub(r'^.*?w', 'w', website2)
website2
date_of_birth2 = None
output2 = pd.DataFrame({'name':[name2], 'phone':[phone2], 'Email': [Email2], 'website': [website2], 'date_of_birth': [date_of_birth2]})
output2
output2 = output2.T
output2
!jupyter nbconvert --to pdf /kaggle/input/rd-test-turn-this-into-a-pdf/__notebook__.ipynb --output /kaggle/working/output.pdf