edad=21

if edad>=18:

    print("Eres mayor de edad")

else:

    print("Eres menor de edad")

    
minutos=121

horas=minutos/60



if minutos%60>0:

    horas=minutos//60+1

else:

    horas=minutos/60

pago=horas*15

print("Estuviste",minutos, "minutos para la cantidad de",pago, "pesos por", horas, "horas")
numero=187



if numero%2==0:

    print("El numero",numero, "es par")

else:

    print("El numero",numero, "es impar")

#Variables

CantidadArtComp=12

Precio=50



#Determinación de Docenas Compradas

Docenas=CantidadArtComp//12



print("Usted compró",CantidadArtComp,"artículos, formando así",Docenas,"docenas, por lo que tenemos un:")



#Monto de Compra

MontoCompra=CantidadArtComp*Precio

print("Monto de Compra: $",MontoCompra)



#Descuento Aplicable

DescuentoCaso1=MontoCompra*0.15

DescuentoCaso2=MontoCompra*0.10

#Monto a Pagar

MontoPagarCaso1=MontoCompra-DescuentoCaso1

MontoPagarCaso2=MontoCompra-DescuentoCaso2



if Docenas>3:

    print("Monto del Descuento Aplicado: $",DescuentoCaso1)

    print("Monto a Pagar: $", MontoPagarCaso1)

else:

    print("Monto del Descuento Aplicado: $", DescuentoCaso2)

    print("Monto a Pagar: $", MontoPagarCaso2) 



#Unidades de Obsequio

if Docenas>4:

    print("Obsequio:",Docenas-3,"artículos")

else:

    print("Obsequio:",0,"artículos")



#Introducir un número de 3 cifras

Numero=565



digito1=int(Numero/100)

digito3=int(Numero%100%10)



if digito1==digito3:

    print("El número",Numero,"se lee igual de izquierda a derecha o de derecha a izquierda")

else:

    print("El número ingresado no se lee igual al revés")
