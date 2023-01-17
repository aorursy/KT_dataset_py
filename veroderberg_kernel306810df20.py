import pandas as pd

from matplotlib import pyplot as plt
file = "../input/Remboursements.xlsx"

sheets = [0,1,4,5]

data_dict = pd.read_excel(file, sheets)



for sheet in data_dict.keys():

    data_dict[sheet] = data_dict[sheet].dropna(axis=0, how="all").iloc[:, :8]

    

all_sheets = pd.concat([data_dict[i] for i in sheets], axis=0, ignore_index=True)



all_sheets.rename(columns={"Remb. Manuvie":"Manuvie", "Remb. CCQ":"CCQ", "Remb. SSQ":"SSQ"}, inplace=True)



start_date = "2016-01-01"

traitements = all_sheets[(all_sheets["Date"]>= start_date)].sort_values(by="Date")

traitements.replace("psy ", "psy", inplace=True)

traitements.replace("médicaments mars à mai", "médicaments", inplace=True)

traitements.reset_index(drop=True, inplace=True)



totaux = traitements.sum(axis = 0, skipna = True)

print("\nTotal cost of treatments and insurance benefits since " + start_date + ":")

print(totaux)

par_traitement_patient = traitements.groupby(["Traitement", "Patient"]).sum()

print(par_traitement_patient)



#par_patient_traitement = traitements.groupby(["Patient", "Traitement"]).sum()

#print(par_patient_traitement)
def make_pie(sizes, text, colors, labels, radius=1):

    col = [[i/255 for i in c] for c in colors]

    

    plt.axis('equal')

    width = 0.35

    kwargs = dict(colors=col, startangle=180)

    outside, _ = plt.pie(sizes, radius=radius, pctdistance=1-width/2,labels=labels,**kwargs)

    plt.setp( outside, width=width, edgecolor='white')



    kwargs = dict(size=20, fontweight='bold', va='center')

    plt.text(0, 0, text, ha='center', **kwargs)



colors_dict = {"masso": (115,0,153), 

               "médicaments": (230,230,0),

               "ostéo": (0, 0, 153), 

               "physio": (179,0,0), 

               "psy": (204,82,0)}



subcolors_dict = {('masso', 'Dad'): (153, 0, 204),

                  ('masso', 'Kid 2'): (236, 179, 255),

                  ('masso', 'Kid 1'): (217, 102, 255),

                  ('masso', 'Mom'): (198, 26, 255), 

                  ('médicaments', 'Dad') : (255, 255, 26),

                  ('médicaments', 'Kid 2'): (255, 255, 204),

                  ('médicaments', 'Kid 1'): (255, 255, 153),

                  ('médicaments', 'Mom') : (255, 255, 102), 

                  ('ostéo', 'Dad') : (0, 0, 204),

                  ('ostéo', 'Kid 2'): (179, 179, 255),

                  ('ostéo', 'Kid 1') : (102, 102, 255), 

                  ('ostéo', 'Mom') : (26, 26, 255),

                  ('physio', 'Dad'): (230,0,0),

                  ('physio', 'Kid 2'): (255, 204, 204),

                  ('physio', 'Kid 1'): (255, 128, 128),

                  ('physio', 'Mom'): (255, 51, 51), 

                  ('psy', 'Dad'): (255, 102, 0), 

                  ('psy', 'Kid 2'): (255, 209, 179),

                  ('psy', 'Kid 1'): (255,179,128), 

                  ('psy', 'Mom'): (255, 148, 77)}
years = [2016, 2017, 2018, 2019]

splitted_by_year = {}

traitement_patient_year = {}

par_traitement = {}

par_patient = {}

prix_par_traitement = {}

who_paid_what = {}

median_prices = {}



for year in years:

    start = str(year)+"-01-01"

    end = str(year)+"-12-31"

    splitted_by_year[year] = pd.DataFrame(traitements[(traitements["Date"]>= start) & (traitements["Date"] <= end)].sort_values(by="Date"))

    traitement_patient_year[year] = splitted_by_year[year].groupby(["Traitement", "Patient"]).sum()

#    traitement_patient_year[year].reset_index(inplace=True)

    par_traitement[year] = splitted_by_year[year].groupby(["Traitement"]).sum()

    par_patient[year] = splitted_by_year[year].groupby(["Patient"]).sum()

    prix_par_traitement[year] = splitted_by_year[year]["Prix"].round(2)

    median_prices[year] = splitted_by_year[year]["Prix"].median()

    print(year)

    print(traitement_patient_year[year])

    print("\nNet cost for " + str(year) + " is $" + str(splitted_by_year[year]["Coût net"].sum(axis=0, skipna=True)) + ".")

    

    who_paid_what[year] = traitement_patient_year[year].sum()[1:]

    plt.barh(who_paid_what[year].index, width=who_paid_what[year])

    plt.title("Who paid what?")

    plt.show()



# Exterior donut has a simple index.    

    make_pie(list(par_traitement[year]["Prix"]), "", [colors_dict.get(label) for label in list(par_traitement[year].index)], list(par_traitement[year].index), radius=1.2)

# Interior donut has a multi-index, but I only select the name of the patient.

    make_pie(list(traitement_patient_year[year]["Prix"]), "", [subcolors_dict.get(label) for label in list(traitement_patient_year[year].index)], [tup[1] for tup in list(traitement_patient_year[year].index)], radius=1)

    plt.title("Who received treatments from whom?")

    plt.show()

    

    plt.hist(prix_par_traitement[year], bins=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250], align="mid", rwidth=0.95)

    plt.title("Price of treatments")

    plt.xticks([25, 50, 75, 100, 125, 150, 175, 200, 225, 250])

    plt.show()

    

# Line plot for evolution of median price over years.

plt.plot(years, median_prices.values(), color='green', linestyle='dashed', marker='o',

     markerfacecolor='blue', markersize=12)

plt.xticks(years)

plt.title("Median price per year")

plt.show()
