import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
veri=pd.read_excel("../input/ornekDF.xlsx")
veri#tarih=index olsun, productid ise colon indexi(dikkat 5 yok)
yeniDF=pd.DataFrame(0, index=veri.tarihSıra.unique(), columns=np.sort(veri.productid.unique()))
yeniDF
for i in veri.index:
    yeniDF[veri[veri.index==i].productid.values[0]][veri[veri.index==i].tarihSıra.values[0]]=veri[veri.index==i].soldquantity.values[0]
yeniDF
a=np.array(range(1,6+1))
a
b=np.repeat(a, 3+1)
b
c=np.array(range(0,3+1))
c
d=np.tile(c, 6)
d
bDF=pd.DataFrame(data=b,columns=["productid"])
bDF["tarihSıra"]=d
bDF["soldquantity"]=0
bDF
veri2 = veri.set_index(['productid',"tarihSıra"])
veri2
bDF2 = bDF.set_index(['productid',"tarihSıra"])
bDF2
spnDF=veri2+bDF2
spnDF
spnDF.reset_index()