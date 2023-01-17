# Optimizasyon_Minimum_Ödev

import numpy as np # numpy kütüphanesinin import edilmesi 

from scipy.optimize import minimize  # scipy kütüphanesinin import edilmesi 

fun = lambda x: 2*x[0]**2 - x[1]**2 + 6*x[1] # Amaç fonksiyonunun yazılması 

cons = ({'type': 'ineq', 'fun': lambda x:  x[0]**2 + x[1]**2 -16 })  # Kısıtın girilmesi 

bnds = ([-4.0,4.0],[-4.0,4.0]) #  x[0]  ile  x[1]   [-4,+4] arasında değer alabilir.

res = minimize(fun, (0, 0), method='SLSQP',bounds=bnds,constraints=cons)  # x[0]  için '0 ' x[1] için '0' başlangıç değeri atanır.

# Method olarak SLSQP (Sequential Least SQuares Programming ) yöntemi kullanılmıştır.

print(res)  # 2 iterasyonda optimum (min) değeri bulunmuştur.Amaç fonksiyon değeri -40 , 

# x[0] -> -2.98023224e-08 (yaklaşık 0 ) ,  x[1] -> -4.00000000e+00 (yaklaşık -4) değerini almıştır.

