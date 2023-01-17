import pandas as pd

import pandas_profiling as pp



from IPython.display import IFrame

%matplotlib inline
tgh= pd.read_csv("../input/Hospitals.csv")

tgh.head(1)
tgh.set_index(["Hospitals"], inplace = True)

tgh.index.rename("Districts", inplace = True)

tgh.head(1)
print(tgh.shape)

print(tgh.info())
tgh
tgh.describe(include = "all").T
corr =tgh.corr()

corr.style.background_gradient(cmap="coolwarm")
tgfru = tgh[tgh.columns[2:5]]

tgh.insert(5, column = "FRU", value = tgfru.sum(axis=1) )

tgh_new = tgh[["FRU", "Teaching Hospitals", "Doctors in all Hospitals", "Beds in all Hospitals"]]

corr2 =tgh_new.corr()

corr2.style.background_gradient(cmap="Spectral")
tgh_ayush = tgh[["Ayurveda Hospitals (incl. Dispensaries)", "Homeopathic Hospitals (incl. Dispensaries)", 

                 "Unani Hospitals (incl. Dispensaries)", "Naturopathy Hospitals (incl. Dispensaries)"]]

corr3 =tgh_ayush.corr()

corr3.style.background_gradient(cmap="PiYG")
tgh_hc = tgh[["Health Sub-Centres", "Primary Health Centres", "Community Health Centres"]]

corr4 =tgh_hc.corr()

corr4.style.background_gradient(cmap="PuOr")
# overriding correlations so as to avoid "rejected variables". 



tgh_profile = pp.ProfileReport(tgh, correlation_overrides=(['Health Sub-Centres', 'Primary Health Centres',

       'Community Health Centres', 'Area Hospitals', 'District Hospitals',

       'FRU', 'Teaching Hospitals', 'Ayurveda Hospitals (incl. Dispensaries)',

       'Homeopathic Hospitals (incl. Dispensaries)',

       'Unani Hospitals (incl. Dispensaries)',

       'Naturopathy Hospitals (incl. Dispensaries)',

       'Doctors in all Hospitals', 'Beds in all Hospitals']), check_correlation = True)



tgh_profile
no_area_hospitals = tgh["Area Hospitals"] == 0

print("Districts with no Area Hospitals")

tgh[no_area_hospitals]
no_CHC = tgh["Community Health Centres"] == 0

print("Districts with no Community Health Centres")

tgh[no_CHC]
no_FRU = tgh["FRU"] == 0

print("Districts with zero First Referral Units")

tgh[no_FRU]
naturo = tgh["Naturopathy Hospitals (incl. Dispensaries)"] > 0

print("Districts with Naturopathy Hospitals (incl. Dispensaries)")

tgh[naturo]
teaching = tgh["Teaching Hospitals"] > 0

print("Districts with Teaching Hospitals")

tgh[teaching]
IFrame ("https://public.tableau.com/views/GovernmentHospitalsinTelanganaIndia/FirstReferralUnits?:embed=yes&:display_count=yes&:showVizHome=no", width=700, height=500)
IFrame("https://public.tableau.com/views/GovernmentHospitalsinTelanganaIndia/NumberofDoctorsBeds?:embed=yes&:display_count=yes&:showVizHome=no", width=700, height=500)
IFrame("https://public.tableau.com/views/GovernmentHospitalsinTelanganaIndia/AyushHospitals?:embed=yes&:display_count=yes&:showVizHome=no", width=700, height=400)
IFrame("https://public.tableau.com/views/GovernmentHospitalsinTelanganaIndia/HealthCentres?:embed=yes&:display_count=yes&:showVizHome=no", width=500, height=500)