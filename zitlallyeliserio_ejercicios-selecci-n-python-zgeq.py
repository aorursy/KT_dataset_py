edad=23
if edad>=18:
    print("Eres mayor de edad")
else:
    print("eres menor de edad")
minutos= 121

print(horas)
if minutos%60>0 :
    horas=minutos//60+1
else:
    horas=minutos/60
pago=horas*15
print("estuviste", minutos,"paga la cantidad de",pago, "pesos por", horas,"horas")
numero= 187
if numero%2==0:
    print("el numero", numero,"es par")
else:
    print("el numero", numero,"es impar")

Docena=8
Costodocena=100
Mdecompra=Docena*Costodocena
if Docena<=3:
    Descuentoapli=.10*Mdecompra
    obsequios=0
else: 
    Descuentoapli=.15*Mdecompra
    obsequios=1*Docena
    
montototal=Mdecompra-Descuentoapli
print("Monto de la compra es", Mdecompra , "pesos")
print("Tiene un descuento de", Descuentoapli, "pesos")
print("El monto que se debera pagar es de", montototal, "pesos")
print("los numero de obsequios es de", obsequios)
Numero=475

n1=Numero//100
n2=(Numero % 100) // 10
n3=(Numero % 100) % 10

if (n1==n3):
    print("El numero es igual al reves")
else:
    print("El numero no es igual al reves")