import pandas as pd

# the spreadsheets is imported but it contains multiple sheets and these sheets have unused rows at the top
data_file = '../input/Edexcel-data-set-Issue-1.xls'

# this creates a list (or array) of the worksheet names from the spreadsheet
spreadsheet = ['Camborne May-Oct 1987',
              'Camborne May-Oct 2015',
              'Heathrow May-Oct 1987',
              'Heathrow May-Oct 2015',
              'Hurn May-Oct 1987',
              'Hurn May-Oct 2015',
              'Leeming May-Oct 1987',
              'Leeming May-Oct 2015',
              'Leuchars May-Oct 1987',
              'Leuchars May-Oct 2015',
              'Beijing May-Oct 1987',
              'Beijing May-Oct 2015',
              'Jacksonville May-Oct 1987',
              'Jacksonville May-Oct 2015',
              'Perth May-Oct 1987',
              'Perth May-Oct 2015']


# this "for" loop goes through each named worksheet in the spreadsheet one at a time 
for sheet in spreadsheet:
    # importing the data from the worksheet to "dataset", the headers are not in the top row so they aren't imported
    dataset = pd.read_excel (data_file, sheet_name=sheet, header=None) 
    # Removes the first five rows with blurb at the top
    dataset = dataset.drop([0,1,2,3,4,5], axis=0) 
    
    # Rename the first three columns
    dataset.rename(columns={0: 'Date',
                            1: 'Daily Mean Temperature',
                            2: 'Daily Total Rainfall',}, inplace=True)
    
    # replace 'tr' with '0.025' in rainfall
    dataset['Daily Total Rainfall'] = dataset['Daily Total Rainfall'].replace({'tr': 0.025})
    dataset['Daily Total Rainfall'] = dataset['Daily Total Rainfall'].astype('float')
    
    # finds the means of the temperature and rain
    mean_temperature=round(dataset['Daily Mean Temperature'].mean(),1)
    mean_rain=round(dataset['Daily Total Rainfall'].mean(),1)
    
    #print the sheet name and the mean of the temperature and rain
    print(sheet) 
    print("Mean daily temperature = "+str(mean_temperature)+"°C")
    print("Mean daily total rainfall = "+str(mean_rain)+"mm")
    #print a blank line
    print("\n")
   
    