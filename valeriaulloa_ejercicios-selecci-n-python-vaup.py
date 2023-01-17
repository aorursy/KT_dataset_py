edad=16
if edad>= 18:
    print("Eres mayor de edad")
#complementaria
else:
    print("Eres menor de edad")
minutos=120
horas=minutos/60
print(horas)
if minutos%60>0:
    horas=minutos//60+1
else:
    horas=minutos/60
pago=horas*15
print("Estuviste" ,minutos,"minutos,paga la cantidad de" ,pago, "pesos por",horas,"horas")
numero= 187
if numero%2==0:
    print("El numero" ,numero, "es par")
else:
    print("El numero" ,numero, "es impar")
docenas=10
obsequio=docenas-3
if docenas>3 :
    print("Se realizara un descuento del 15% de su compra y ademas se obsequiara" ,obsequio, "unidad(es) extra(s) por la compra de" ,docenas, "docenas")
else:
    print("Se realizara un desceunto del 10% de su compra")
numero=202
if numero > 99 and numero < 1000:
    print("el numero tiene tres cifras")
else:
    print("el numero no tiene tres cifras")

if numero//100==numero%100:
    print("y el numero es palindromo")
else:
    print("y el numero no es palindromo")