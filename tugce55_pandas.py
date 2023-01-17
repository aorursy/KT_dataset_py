import pandas as pd



print("                               KEŞİFLERE GÖRE DOĞU AKDENİZ'DE BULUNAN DOĞALGAZ REZERVİ")

print()

print()

print()

print("Amerikan Jeolojik Araştırma Merkezi (USGS) 2010 yılında yayınladığı rapora göre Levant Havzası denilen yerde 1,7 milyar varillik iki petrol rezervi olduğu tahmin ediliyor.")

print()

print("Bölgede büyük oranda deniz yatağında olan çıkarılabilir doğal gaz rezervinin 3,45 trilyon metreküp olduğu tahmin ediliyor.")
dictionary = {"KEŞİF ADI:":["Tamar","Leviathan","Zohr","Calypso","Glaucus-1"],

              "YIL":[2009,2010,2015,2018,2019],

              "KEŞİF YERİ:":["Münhasır Ekonomik Bölgesi (MEB)","Hayfa","Mısır","Calypso Sahası","ON Numaralı Blok"],

              "REZERV:":[280,622,849.000,226,227]}



dataFrame1= pd.DataFrame(dictionary)

print()

print("dataFrame1 :")

print(dataFrame1)

print(type(dataFrame1))
head = dataFrame1.head()

print()

print("Head :")

print(head)
tail = dataFrame1.tail()

print()

print("Tail :")

print(tail)
kolonlar = dataFrame1.columns

print()

print("Kolonlar :")

print(kolonlar)
Infomuz = dataFrame1.info()

print()

print("Info :")

print(Infomuz)
describe1 = dataFrame1.describe()

print()

print("Describe :")

print(describe1)

print()

print("Kaynak:")

print("https://tr.euronews.com/2019/05/05/dogu-akdeniz-ne-kadar-dogal-gaz-rezervi-var-en-buyuk-payi-hangi-ulkeler-alacak")

v_1=dataFrame1['KEŞİF ADI:']

print(v_1)
v_currenyList1 = ["TR","DEU","ENG","FR","USA"]

dataFrame1["DAHİL ÜLKELER"] = v_currenyList1



print(dataFrame1.head())
v_AllCapital = dataFrame1.loc[:,"REZERV:"]

print(v_AllCapital)

print()

print("Type of v_AllCapital is : " , type(v_AllCapital))
v_top3Currency = dataFrame1.loc[0:4,"REZERV:"]

print(v_top3Currency)
v_KESİF = dataFrame1.loc[:,["KEŞİF YERİ:","YIL:","KARŞI ÇIKAN ÜLKELER"]] 

print(v_KESİF)
v_Reverse1 = dataFrame1.loc[::-1,:]

print(v_Reverse1)
v_meanPop =dataFrame1["YIL:"].mean()

print(v_meanPop)
for a in dataFrame1["YIL:"]:

    print(a)
dataFrame1["POP LEVEL"] = ["Low" if v_meanPop > a else "HIGH" for a in dataFrame1["REZERV:"]]

print(dataFrame1)
print(dataFrame1.columns)

dataFrame1.columns = [a.lower() for a in dataFrame1.columns]



print(dataFrame1.columns)

dataFrame1.columns = [a.split()[0]+"_"+a.split()[1] if (len(a.split())>1) else a for a in dataFrame1.columns]

print(dataFrame1.columns)

dataFrame1["test1"] = [-1,-2,-3,-4,-5]

print(dataFrame1)
dataFrame1.drop(["test1"],axis=1,inplace = True) 

print(dataFrame1)
dataFrame1["test1"] = [a*2 for a in dataFrame1["rezerv:"]]

print(dataFrame1)
def f_multiply(v_yıl):

    return v_yıl*3



dataFrame1["test2"] = dataFrame1['rezerv:'].apply(f_multiply)

print(dataFrame1)