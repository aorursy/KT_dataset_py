import numpy as np



bal = 1000

loss = (30 * (1+(5/100)) ** 12) * 1000



def borrowers(month):

    return 1000 * (1+(5/100)) ** month



annual_charges = (20 + 25 + 10) * borrowers(11)



def balance(month):

    return (bal * (1+(5/100)) ** month)



def revenue(n,N,i,principal,cust,interest):

    a = (principal * ((1 + (n * interest) / 100) ** N) * (1 + (i * interest)/100)) * cust

    return a



def calc(interest):

    gross_revenue = []

    cost_of_funds = []

    principal = []

    

    for month in range(12):

        

        revenue_1 = np.sum([balance(month)*0.37*borrowers(month),

                            revenue(1,1,0,balance(month),0.4*borrowers(month),interest),

                            revenue(2,1,0,balance(month),0.15*borrowers(month),interest),

                            revenue(3,1,0,balance(month),0.03*borrowers(month),interest),

                            revenue(3,1,1,balance(month),0.05*borrowers(month),interest)])

        

        cost = 0.065 * borrowers(month) * balance(month)

        p = borrowers(month) * balance(month)

        

        gross_revenue.append(revenue_1)

        cost_of_funds.append(cost)

        principal.append(p)

        

    return gross_revenue, cost_of_funds, principal





def profit(interest):

    net_revenue = annual_charges + np.sum(calc(interest)[0]) - (loss + np.sum(calc(interest)[1]))

    

    profit = net_revenue - np.sum(calc(interest)[2])

    profit_percentage = (profit / np.sum(calc(interest)[2])) * 100

    

    return profit, profit_percentage



profit2 = profit(15)[0] - borrowers (12)*20



print("Annual profit : Rs. "+str(profit(15)[0])+" which is "+str(profit(15)[1])+"%.")
bal = 2000



# Loss when loss rate is 3% of outstanding balance default



loss1 = (0.03 * np.sum(calc(15)[0])) * 2



print("Loss when loss rate is 3% of the outstanding balance default : Rs. "+str(loss1))



# Loss when loss rate is 3% of the borrowers default on entire balance.



loss2 = (30 * (1+(5/100)) ** 12) * 1000



print("Loss when loss rate is 3% of the borrowers default on entire balance : Rs. "+str(loss2))
# profit2 is calculated in answer 1

pp = (profit2/np.sum(calc(15)[2])) * 100



print ("Annual profit the affiliate group after customers aquisition : Rs. "+str(profit2)+" which is "+str(pp)+"%")
profit1 = profit(10)[1]

profit2 = profit(13)[1]



print("The financial institution can reduce the interest rate to range from 10-13% in order to convince the affiliate group not to want to buy the customers and the profi for the institution in this case would be in range "+str(profit1)+"-"+str(profit2)+"%") 
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



def calc1(p,interest):

    gross_revenue = []

    cost_of_funds = []

    principal = []

    

    for month in range(12):

        

        revenue_1 = np.sum([balance(month)*borrowers(month)*((100-p)/100),

                            revenue(3,1,1,balance(month),(p/100)*borrowers(month),interest)])

        

        cost = 0.065 * balance(month) * borrowers(month)

        p = borrowers(month) * balance(month)

        

        gross_revenue.append(revenue_1)

        cost_of_funds.append(cost)

        principal.append(p)

        

        return gross_revenue, cost_of_funds, principal

    

cust = [10,20,30,40,50]

profits = []



for i in range(5):

    

    net_revenue = annual_charges + np.sum(calc1(cust[i],15)[0]) - (loss + np.sum(calc1(cust[i],15)[1]))

    

    profit = net_revenue - np.sum(calc1(cust[i],15)[2])

    pp = (profit/np.sum(calc1(cust[i],15)[2])) * 100

        

    profits.append(pp)

    

df = pd.DataFrame({'percentage of total customers delaying payment':cust,

                   'Profit percentage': profits})



print(df)

sns.barplot(x='Profit percentage',y='percentage of total customers delaying payment',data=df)