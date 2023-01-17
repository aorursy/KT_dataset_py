import numpy as np
v_milliyazilimlar = np.array(['Kuzgun Mobil Canlı Yayın Yazılımı','Aselsan Entegre Komuta Kontrol Sistemi','GAG-Geniş Alan Gözetleme Sistemi'

                   ,'Aselsan AcroSAT Gemi Uydu Yazılımı', 'Kement Projesi','Adop Ateş DestekOtomasyon Sistemi',

                   'Tubitak Gerçek Zamanlı İletişim Sistemi'

                   ,'Denizgözü Martı Elektroptik Gözetleme Sistemi'])



print(v_milliyazilimlar)

print()

print('type:',type(v_milliyazilimlar))

# shape

v_shape = v_milliyazilimlar.shape

print("v_shape : " , v_shape , " and type is : " , type(v_shape))
# Reshape

v_array = v_milliyazilimlar.reshape(8,1)

print(v_array)

v_shape2 = v_milliyazilimlar.shape

print("v_shape2 : " , v_shape2 , " and type is : " , type(v_shape2))
# Dimension

v_dimen1 = v_milliyazilimlar.ndim

print("v_dimen1 : " , v_dimen1 , " type is : " , type(v_dimen1))
# Dtype.name

v_dtype1 = v_milliyazilimlar.dtype.name

print("v_dtype1 : " , v_dtype1 , " and type is : " , type(v_dtype1))
# Size

v_size1 = v_milliyazilimlar.size

print("v_size1 : " , v_size1 , " and type : " , type(v_size1))


v_array4 = np.array([[21,89,23,789],[8982,7201,7429,12],[89,12,45,23]])

print(v_array4)

print("---------------")

print("Shape is : " , v_array4.shape)


v_array5 = np.zeros((571,632))

print(v_array5)


v_array5[0,2] = 87

print(v_array5)
v_array6 = np.ones((0,1))

print(v_array6)
v_array7 = np.empty((12,34))

print(v_array7)
v_array8 = np.arange(2,68,7)

print(v_array8)

print(v_array8.shape)

v_array9 = np.linspace(3,9,56)

v_array10 = np.linspace(10,30,20)



print(v_array9)

print(v_array9.shape)

print("-----------------------")

print(v_array10)

print(v_array10.shape)
v_np1 = np.array([-67,34,9078])

v_np2 = np.array([0,-9087,4567])



print(v_np1 + v_np2)

print(v_np1 - v_np2)

print(v_np2 - v_np1)

print(v_np1 ** 2)
print(np.sin(v_np2))
v_np2_TF = v_np2 < 900

print(v_np2_TF)

print(v_np2_TF.dtype.name)

v_np1 = np.array([900876,4546,0,565])

v_np2 = np.array([-9075,356,1,0])

print(v_np1 * v_np2)

#Indexing and Slicing





print('ilk yazilim:', v_milliyazilimlar[0])

print('son yazilim:', v_milliyazilimlar[-1])

print(v_milliyazilimlar[2:-1])
v_milliyazilimlar_Rev = v_milliyazilimlar[1:-4]

print(v_milliyazilimlar_Rev)
v_2=np.array([[2,3,57],[2,3,89],['selam','ben','tugce'],[78,12,-2]])

print(v_2[2,2])
print(v_2[:,2])
print(v_2[2,:])
print(v_2[2,1:2])
v_3=v_milliyazilimlar.ravel()

print(v_3)

print()

print('and shape is:',v_3.shape)
v4=v_2.T

print(v4)
print(v_2)

print()

print(v_2.reshape(3,4))

print()

print(v_2)

print()

v_2.resize((2,6))

print(v_2)
dizi7=[[1,2],[3,4]]

dizi8=([[-1,-2],[-3,-4]])



dizi9=np.vstack((dizi7,dizi8))

dizi10=np.hstack((dizi7,dizi8))

dizi11=np.vstack((dizi8,dizi7))

dizi12=np.hstack((dizi8,dizi7))



print(dizi9)

print()

print(dizi10)

print(dizi11)

print(dizi12)
dizi34=np.array([4,5,6])

dizi35=dizi34.copy()

dizi36=dizi35.copy()

dizi36[1]=8

print(dizi34)

print(dizi35)

print(dizi36)
v_5 = np.array([[8,4,24],[90,1,4]])

v_5Transpose = v_5.T

print(v_np5)

print(v_np5.shape)

print()

print(v_5Transpose)

print(v_5Transpose.shape)
v_6 = v_5.dot(v_5Transpose)

print(v_6)
v_5Exp = np.exp(v_5)



print(v_5)

print(v_5Exp)
v_8 = np.random.random((3,2))

print(v_8)

print(v_8.shape)
v_8Sum = v_8.sum()

print("Sum of array : ", v_8Sum)  

print("Max of array : ", v_8.max()) 

print("Min of array : ", v_8.min()) 
print("Sum of Columns :")

print(v_8.sum(axis=0)) 

print()

print("Sum of Rows :")

print(v_8.sum(axis=1)) 
print(np.sqrt(v_8))

print()

print(np.square(v_8))
v_11 = np.array([10,20,30,40,50,60])

v_12=np.array([78,-223,12,534,657,97])



print(np.add(v_12,v_11))