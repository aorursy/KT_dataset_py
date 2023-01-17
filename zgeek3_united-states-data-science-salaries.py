# Javascript on/off code copied from: 
# https://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer
# To present a cleaner notebook

from IPython.display import HTML

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
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
# Turning off auto scrolling per:  https://stackoverflow.com/questions/36757301/disable-ipython-notebook-autoscrolling

HTML('''<script>
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
</script>''')

# Import statements for all of the packages needed

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import textwrap
% matplotlib inline

import warnings
warnings.filterwarnings('ignore')


# Setting up colors for bar charts

sns.set_palette(sns.color_palette("Set2")+sns.color_palette("Paired"))
# Changing the salary ranges more relevant to US market and smaller number
# so the info willl show up better in charts

def newranges(test):
    
    if test == "What is your current yearly compensation (approximate $USD)?":
        new = "(Updated) What is your current yearly compensation (approximate $USD)?"
    if test != "What is your current yearly compensation (approximate $USD)?":
        
        if test != "500,000+":
            top=int(test.split('-',)[1].replace(',',''))
        if test == "500,000+":
            top=500000        
        if top <= 50000:
            new = '0-50,000'
        elif top <= 100000:
            new = '50,000-100,000'
        elif top <= 150000:
            new = '100,000-150,000'
        elif top <= 200000:
            new = '150,000-200,000'
        elif top >= 200000:
            new = '200,000+'
            
    return new

# Trim the data reduces the data based on a specific question's desired answer 
# Want to keep.

# Keep = YES - keep only rows with specific column value
# Keep = NO - remove rows with column value 
# Keep = NAN - drop rows with NaN values
# Keep = NOANDCLEAN - remove rows with specific value and clean rows with NAN

# This will be used to filter down to just Survey respondants from the 
# United States and only ones who answered the salary question.

def trimthedata(keep,question,answer,multiplechoiceframe,freeformotherframe):   
    
    # Trimming multipleChoiceResponses frame mcr

    mcr = multiplechoiceframe
    ffr = freeformotherframe
    ffr.head()

   
    mcrquestions = mcr.loc[[0]]
    mcrquestions.head()
 
    
    # Keeping just rows where the question has a certain answer
    
    if keep == 'YES':
        mcranswers = mcr[mcr[question]==answer]
 
        
    # Removing rows where the question has a certain answer
    
    if keep == 'NO' or keep=='NOANDCLEAN':
        mcranswers = mcr[mcr[question]!=answer]
        mcranswers=mcranswers.drop(mcranswers.index[[0]])
        
    # Removing rows where the question has NaN value
    
    if keep == 'NOANDCLEAN':
        mcranswers = mcranswers.dropna(subset=[question])  
    
    # Getting values to adjust ffr dataframe as well
    
    uslist=list(mcranswers.index.values)

    mcr = mcrquestions.append(mcranswers)


    # Addusting freeFormResponses.csv ffr to match
    
    ffrquestions = ffr.loc[[0]]

    ffranswers = ffr.loc[uslist]
    ffranswers.head()

    ffr = ffrquestions.append(ffranswers)
    ffr.head(5)
    
    # Returning the two cleaned up dataframes

    return mcr,ffr
    


# Creating a bar chart of just the counts for salary range

def basicsalaryrangechart(newyearlyranges,multiplechoiceframe,showwork):
    
    x=0
    mcr = multiplechoiceframe
    
    question = mcr['Q9'][0]
    
    if showwork == "YES":

        print("---------------------------------------------------------------")
        print("Study of: " + question)
        print("---------------------------------------------------------------")

    mcr=mcr.drop(mcr.index[[0]])
   
    values=pd.value_counts(mcr['Q9'])

   
    newse = pd.Series()
    
    # Reordering the series to match the numerical values

    for range in newyearlyranges:
        x=0

        while x < len(values):
            if values.index[x] == range:
            
                newse.set_value(range,values[x])
            x=x+1
    if showwork == "YES":
        print(newse)
        
    # Plotting bar chart of salary ranges
    
    newse.plot.bar(figsize=(10,7),title=question)
    plt.xlabel('Salary Range')
    plt.ylabel('Total Number')
    plt.show()
    
# Creating bar graph based on multiplechoice answers

def getmultiplechoiceanswers(newyearlyranges,multiplechoiceframe,questiona,questionb,showwork): 
    
        if showwork == "YES":
            print("Question - " + questiona + ":  " + multiplechoiceframe[questiona][0])
            print("-----------------------------------------------------------------")
        
        # Creating a pivot table 
        
        counts = multiplechoiceframe[1:].groupby([questionb, questiona]).size().reset_index(name="Count")
        new_columns=['salary']
        two_columns=counts[questiona].unique()
        for c in two_columns:
            new_columns.append(c)

        new_test_pivot = counts.pivot(index=questionb,columns=questiona,values='Count')

        new_df= pd.DataFrame(columns=new_columns)

        x=0

        for range in newyearlyranges:
            new_df = new_df.append({'salary':range},ignore_index=True)
        
              
        for range in newyearlyranges:
            for nc in two_columns:
                new_df[nc][x]=new_test_pivot.loc[range,nc]
                if np.isnan(new_df[nc][x]):
                    new_df[nc][x]=0
            x=x+1
        
        
        new_df=new_df.set_index('salary')
        if showwork=="YES":
            print(new_df)
        
        new_df.loc[:,new_df.columns].plot.bar(stacked=True, figsize=(10,7),title = textwrap.fill(multiplechoiceframe[questiona][0]))
        plt.xlabel('Salary Range')
        plt.ylabel('Total Number')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
        

# Creating bar charts for select any and % questions


def getselectanyanswers(newyearlyranges,mainschemaframe,multiplechoiceframe,freeformotherframe,questiona,questionb,showwork):
    
    
    framecolumns = multiplechoiceframe.columns
    
    choices = []

    if showwork == "YES":
        print("Question - " + questiona + ":  " + mainschemaframe[questiona][0])
        print("-----------------------------------------------------------------")
        

        
    
    w=1
    if w==1:
    
    #if "Select all that apply" in mainschemaframe[questiona][0]:
        parts = [s for s in framecolumns if questiona + '_Part_' in s and 'OTHER' not in s and 'TEXT' not in s]

      
        for part in parts:
            
            if "Select all that apply" in multiplechoiceframe[part][0]:
                if "Selected Choice - " in multiplechoiceframe[part].unique()[0]:
                    choices.append(multiplechoiceframe[part].unique()[0].split("Selected Choice - ")[1])
                if "Selected Choice - " not in multiplechoiceframe[part].unique()[0]:
                    choices.append(multiplechoiceframe[part].unique()[0].split("(Select all that apply) - ")[1])
            if "Answers must add up to 100%" in multiplechoiceframe[part][0]:
                choices.append(multiplechoiceframe[part].unique()[0].split("(Answers must add up to 100%) - ")[1])
        
        newyearlyranges = ['0-50,000','50,000-100,000','100,000-150,000','150,000-200,000','200,000+']
    
        new_df= pd.DataFrame(columns=choices,index=newyearlyranges)


        x=0
        p=0
        for range in newyearlyranges:
            y=0

            for part in parts:
                temp_df = mcr[mcr['Q9']==range]

                if "Select all that apply" in multiplechoiceframe[part][0]:

                    value = temp_df[part].count()

                    numberofrespondents = temp_df.shape[0]

                    new_df[choices[y]][range]=value/numberofrespondents
                    if np.isnan(new_df[choices[y]][range]):
                        new_df[choices[y]][range]=0
                if "Answers must add up to 100%" in multiplechoiceframe[part][0]:
                    mci=list(temp_df.index.values)
                    values = temp_df.loc[mci[1]:, part].dropna().tolist()
                    values = [float(i) for i in values]
                    myarray = np.asarray(values)
                    new_df[choices[y]][range]=round(myarray.mean(),2)
                    if np.isnan(new_df[choices[y]][range]):
                        new_df[choices[y]][range]=0
                    
                                
                if questiona != 'Q35' and questiona + "_OTHER_TEXT" in freeformotherframe.columns and "Answers must add up to 100%" in multiplechoiceframe[part][0]:
                    
                    # Getting values to adjust ffr dataframe as well
                    ffr_column = questiona + "_OTHER_TEXT"
                    if p==0:
                        new_df["Other"] = np.nan
                        p=p+1
    
                    tlist=list(temp_df.index.values)
                    tlist = [float(i) for i in tlist]
                    tarray = np.asarray(tlist)
               
                    temp_ffr = freeformotherframe.loc[tarray]
                    

                    ffri=list(temp_ffr.index.values)
                    values = temp_ffr.loc[ffri[1]:, ffr_column].dropna().tolist()
                    values = [float(i) for i in values]
                    myarray = np.asarray(values)
                    
                    new_df["Other"][range]=round(myarray.mean(),2)
                    if np.isnan(new_df["Other"][range]):
                        new_df["Other"][range]=0
    
                    
                
                y=y+1
            x=x+1
        if showwork == "YES":
            print(new_df)
        
        
        
    
    dtitle = str(mainschemaframe[questiona][0]) 
    dtitle=textwrap.fill(dtitle, 60)
        
  
    if "Select all that apply" in mainschemaframe[questiona][0]:
        new_df.loc[:,new_df.columns].plot.bar(stacked=False, figsize=(10,7),title=dtitle + "\nPercent Selected")
        
    if "100%" in mainschemaframe[questiona][0]:
        new_df.loc[:,new_df.columns].plot.bar(stacked=True, figsize=(10,7),title=dtitle)
        

    plt.xlabel('Salary range')
    plt.ylabel('Percentage')

    
         
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    

# Load in Schema Data

ss = pd.read_csv('../input/SurveySchema.csv',low_memory=False)
questions=ss.columns

# Load other data and reduce down to desired data

# Loading in the data provided

mcr = pd.read_csv('../input/multipleChoiceResponses.csv',low_memory=False)
mcresponses = mcr.columns
originalresponses = mcr.shape[0]-1

ffr = pd.read_csv('../input/freeFormResponses.csv',low_memory=False)
ffrcolumns = ffr.columns

# Trimming down to just US data

mcr,ffr = trimthedata('YES','Q3','United States of America',mcr,ffr)

unitedstatesresponses = mcr.shape[0]-1

# Gettting rid of rows where the respondents did not disclose salary

mcr,ffr = trimthedata('NOANDCLEAN','Q9','I do not wish to disclose my approximate yearly compensation',mcr,ffr)

mcr.head()

# Creating a new column that updates the salary ranges as appendix to original

newyearlyranges = ['0-50,000','50,000-100,000','100,000-150,000','150,000-200,000','200,000+']

mcr['A1'] = mcr.apply(lambda row: newranges(row.Q9), axis=1)

# Replacing the original salaries with the new salaries in the original spot

mcr=mcr.rename(index=str, columns={"Q9": "Original_Q9"})
mcr=mcr.rename(index=str, columns={"A1": "Q9"})

print("Subset of the original data:")
print("----------------------------")
print("  Original responses:  " + str(originalresponses))
print("  Responses from US respondents:  " + str(unitedstatesresponses))
print("  Responses that included salary information:  " + str(mcr.shape[0]-1))

print("CHART 1")
basicsalaryrangechart(newyearlyranges,mcr,"NO")
print("CHART 2")
getmultiplechoiceanswers(newyearlyranges,mcr,'Q1','Q9','NO')
print("CHART 3")
getmultiplechoiceanswers(newyearlyranges,mcr,'Q2','Q9','NO')
print("CHART 4")
getmultiplechoiceanswers(newyearlyranges,mcr,'Q4','Q9','NO')
print("CHART 5")
getmultiplechoiceanswers(newyearlyranges,mcr,'Q5','Q9','NO')
print("CHART 6")
getmultiplechoiceanswers(newyearlyranges,mcr,'Q6','Q9','NO')
print('CHART 7')
getmultiplechoiceanswers(newyearlyranges,mcr,'Q8','Q9','NO')
print('CHART 8')
getmultiplechoiceanswers(newyearlyranges,mcr,'Q26','Q9','NO')

print('CHART 9')
getselectanyanswers(newyearlyranges,ss,mcr,ffr,"Q11",'Q9','NO')
print('CHART 10')
getselectanyanswers(newyearlyranges,ss,mcr,ffr,"Q34",'Q9','NO')
print('CHART 11')
getselectanyanswers(newyearlyranges,ss,mcr,ffr,"Q16",'Q9','NO')
print('CHART 12')
getmultiplechoiceanswers(newyearlyranges,mcr,'Q23','Q9','NO')
print('CHART 13')
getmultiplechoiceanswers(newyearlyranges,mcr,'Q25','Q9','NO')


# Get a clear list of questions and how many people answered them.
# For multiple choice, how many people of each answer.
# For answers that add up to 100% get the average for each value (note there appears to be some bad data)
# For free form, the question, the number of answers and a text cloud



qdict = {}

x=1
print("Title:  " + questions[0])
print('---------------------------------------------------------------')

# Creating a dictionary of questions to reorder.  Note the question # should be replaced by Q# 
# when merging to other info

while x != (len(questions)-1):
    qdict[int(questions[x][1:])] = ss[questions[x]][0]
    x=x+1
time = questions[x] + ': ' + ss[questions[x]][0]
 
    
basicsalaryrangechart(newyearlyranges,mcr,"YES")    
x=1
while x != (len(questions)-1):

    cn = 'Q' + str(x)
    
    parts = [cn]
    
    parts = parts + [s for s in mcresponses if cn + '_Part_' in s and 'TEXT' not in s]
    
    # Two of the questions are text answers only
    if cn != 'Q39' and cn != 'Q41': 
    
        if cn =='Q12':
            getmultiplechoiceanswers(newyearlyranges,mcr,'Q12_MULTIPLE_CHOICE','Q9','YES')

        if len(parts)==1 and cn != 'Q9' and cn != 'Q12' and cn != 'Q3':
            getmultiplechoiceanswers(newyearlyranges,mcr,parts[0],'Q9','YES')
        
        if len(parts) > 1:
            getselectanyanswers(newyearlyranges,ss,mcr,ffr,parts[0],'Q9','YES')
        print("\n")
                   
    x=x+1
    print('\n')
print(time)


