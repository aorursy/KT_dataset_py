# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import HTML
from IPython.display import display
offenders=pd.read_csv("../input/sex-offenders.csv")
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this Jupyter notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
def between(value,start,end):                                   #States whether a certain value is between two other values, exclusive and inclusive.
    return value>start and value<=end 

def convert_to_inches(dataframe):                               #Converts the foot'inch'' measure into just inch''. This will help to get the BMI.
    return (int)((dataframe/100)*12)+(dataframe%100)            

def bmi(weight,height):                                         #Calculates a person's BMI, by doing (lb/in^2)*703. 
    for i in range(len(height)):
        yield ((weight[i] / (height[i] ** 2)) * 703)
        
def get_region(adress):                                         #Gets the region of an adress given. Since they all have 5 digits, a space, then |N|S|W|E|, regex wasn't necessary.  
    mat=adress[6]
    regions = {"E":"East","N":"North","S":"South","W":"West"}
    return regions[mat]

def age_range(value):                                           #returns the block (age range) a certain age is.
    if value<18:
        return '17-'
    elif between(value,17,30):
        return '18~29'
    elif between(value,30,40):
        return '30~39'
    elif between(value,40,50):
        return '40~49'
    elif between(value,50,60):
        return '50~59'
    else:
        return '60+'

def get_bmi_range(value):                                     #same as age_range, but with BMI. the blocks' names and range are different. 
    if value<18.5:
        return 'underweight'
    elif between(value,18.5,25.0):
        return 'normal weight'
    elif between(value,25.0,30.0):
        return 'overweight'
    else:
        return 'obese'


offenders['AGE_RANGE']=[age_range(i) for i in offenders['AGE']]
offenders['AGE_RANGE'].value_counts().plot.bar(title = "Age range of sex offenders", figsize=(15,10))
offenders['HEIGHT_IN_INCHES']=pd.DataFrame(data=[(float)(convert_to_inches(i)) for i in offenders['HEIGHT']])
offenders['BMI'] = pd.DataFrame(data=bmi(offenders['WEIGHT'],offenders['HEIGHT_IN_INCHES'])) 
offenders['BMI_RANGE']=[get_bmi_range(value) for value in offenders['BMI']]
offenders['BMI_RANGE'].value_counts().plot.bar(title="BMI status of sex offenders", figsize=(15,10))
offenders['REGION']=[get_region(add) for add in offenders['BLOCK']]
offenders['REGION'].value_counts().plot.bar(title="Numbers of sex offenders by region",figsize=(15,10))
offenders['RACE'].value_counts().plot.pie(figsize=(20,15))
