# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
#question 1
def search_matchs(text):
    pattern='ab*'
    if re.search(pattern,text) :
            return True
    else :
            return False 
print(search_matchs("abbbb"))
#question 2
text = "BonjourMrwalid"
m=re.findall('[A-Z][a-z]+',text)
m
#question 3
text = "ab"
if re.search('a.*b$',text) :
    print(True)
else :
    print(False)
#question 4
text = "abbzab"
if re.search('\Bz\B',text) :
    print(True)
else :
    print(False)
#question 5
text = 'she was \nexiting*film.'
print(re.split('; |, |\*|\n',text))
#question 6
text = "Clearly, he has no excusement for such behavior."
for m in re.finditer(r"\w+ly", text):
    print('%d-%d: %s' % (m.start(), m.end(), m.group(0)))
################
text2 = "C'est terriblement cher pour un si petit tableau"
for m in re.finditer(r"\w+ment", text2):
    print('%d-%d: %s' % (m.start(), m.end(), m.group(0)))


text = '''Central design committee session 05/02/2022 6:30 pm Th 9/19 LAB: Serial encoding (Section 2.2)
    There will be another one on December 15th for those who are unable to make it today.
    Workbook 3 (Minimum Wage): due Wednesday09/2018 11:59pm
    He will be flying in Sep 15,2020
    We expect to deliver this between late Sep 17,2020 2021 and early 2022.'''
x=re.findall(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{2},\d{4})|(\d{2}/\d{2}/\d{4})', text)
x
