import csv
path = "../input/train1/train.csv"
with open(path) as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    
    for row in readCSV:
        
        ad = row[0]
        yas = row[1].split('.')[0]  
        # yaş
        
        print("../input/trainresim/"+ad+" Yaşım "+(yas))        
        #resim yolu +ad şeklinde çağırabilirsin resimleri
        
