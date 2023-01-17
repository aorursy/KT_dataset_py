# Maksimizasyon Problemi KKT

from scipy.optimize import minimize  # scipy kütüphanesinin import edilmesi 

sign = -1 # maximizasyon için kullanılacak olan sign değişkeninin oluşturulması

fun = lambda x: sign * (x[0]*x[1]) # sign fonksiyonunun amaç fonksiyonla çarpılması 

cons = ({'type': 'ineq', 'fun': lambda x:  sign * (x[0] + x[1]**2 -2) })  # sign fonksiyonunun kısıtla çarpılması 

res = minimize(fun, (1, 0), method='SLSQP',constraints=cons)  # x[0]  için '1 '  , x[1] için '0' başlangıç değeri atanır.

# Method olarak SLSQP (Sequential Least SQuares Programming ) yöntemi kullanılmıştır.

print(res)  #  6 iterasyonda optimum (max) değeri bulunmuştur. Amaç fonksiyon değeri +1.08866 , x[0] -> 1.3333 x[1] -> 0.8165

# KKT adımlarını izlediğimizde x[1]-Λ1+Λ2 = 0 , x[0]-2*x[1]*Λ1+Λ3 = 0 , Λ1*(2-x[0]-x[1]**2) = 0, Λ2 * x[0] = 0 ve Λ3 * x[1] = 0 eşitlikleri elde edilir.

# Olası durum a ) x[0]=x[1]=Λ1=Λ2=Λ2=0 ve Amaç fonksiyonu : 0 

# Olası durum b ) x[0]+x[1]**2 = 2 durumu baz alınarak eşitlikler düzenlenir.

# x = x[0] = 4/3 = 1.3333       ,   y = x[1] =  sqrt(2/3) = 0.8165      ,  Amaç Fonksiyonu : x * y = +1.08866

