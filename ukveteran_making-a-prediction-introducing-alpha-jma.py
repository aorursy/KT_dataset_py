weight=0.5

input=2

goalpred = 0.8

alpha=0.1
for iteration in range(20):

    pred= input*weight

    error=(pred-goalpred)**2

    derivative=input*(pred-goalpred)

    weight=weight-(alpha*derivative)



print("Error:" +str(error) +"Prediction" +str(pred) )