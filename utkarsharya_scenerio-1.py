principal = 1000 * 1000 

annual_charges = (20 + 25 + 10) * 1000

loss = 30 * 1000

cost_of_funds = 0.065 * principal *12



def interest(n,i):

    a = ((n * 1000 * (15 * 12)/365) / 100) * i

    return a 



interest_37 = interest(52,400) 

interest_52 = interest(67,150) 

interest_75 = interest(92,50)



total_interest = 12 * (interest_37 + interest_52 + interest_75)



revenue = annual_charges + (principal * 12) + total_interest



net_revenue = revenue - cost_of_funds



profit = net_revenue - principal*12

profit_percentage = (profit/(principal*12)) * 100



print("Annual profit : Rs. "+str(profit)+" which is "+str(profit_percentage)+"%")

principal = 2000 



# If loss rate is 3% of the outstanding balance default.



def loss1(m,n):

    a = ((m * ((1 + 3/100) ** 12)) - principal) * n

    return a



def with_interest(i):

    gross = ((i * principal * (15 * 12)/365) / 100) + principal

    return gross



one = loss1(principal,370)

two = loss1(with_interest(52),400)

three = loss1(with_interest(67),150)

four = loss1(with_interest(92),80)



total = one+two+three+four



print("Loss when loss rate is 3% of the outstanding balance default : Rs. "+str(total))



# If loss rate is 3% of the borrower's default on entire balance.



def loss2():

    b = 0.03 * principal * 1000

    return b



print("Loss when loss rate is 3% of the borrowers default on etire balance : Rs. "+str(loss2()))
n_cust = 1000

profit2 = profit - (20 * n_cust)



pp = (profit2/(1000000*12)) * 100



print("Annual profit of the affiliate group after customers aquisition : Rs. "+str(profit2)+" which is "+str(pp)+"%")
profit_a = 0.5 * profit

profit_b = 0.7 * profit

principal = 1000 * 1000 *12



def pp(m):

    percentage = (m/principal) * 100

    return percentage



def interest(n):

    a = ((n+cost_of_funds+loss) - annual_charges) * (365/ (35450 * 120 * 12))

    return a



print("Interest can be reduced within range "+str(interest(profit_a))+" to "+str(interest(profit_b))+" % in order to convince the affiliate group not to want to buy the customers.The profit in this case would be in range "+str(pp(profit_a))+"% to "+str(pp(profit_b))+"%")
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



def pp(q):

    principal = 1000 * 1000 

    annual_charges = (20 + 25 + 10) * 1000

    loss = 30 * 1000

    cost_of_funds = 0.065 * principal *12

    

    def interest(n,i):

        a = ((n * 1000 * (15 * 12)/365) / 100) * i

        return a 

    

    interest_30 = interest(45,q)

    

    total_interest = 12 * interest_30

    

    revenue = annual_charges + (principal * 12) + total_interest

    net_revenue = revenue - cost_of_funds

    

    profit = net_revenue - (principal * 12)

    profit_percentage = (profit/(principal*12)) * 100

    return profit_percentage



one = [100,200,300,400,500]

two = [pp(100),pp(200),pp(300),pp(400),pp(500)]



df = pd.DataFrame({'no. of customers paying after 30 days of due date': one,

                   'Profit percentage':two})

df
sns.barplot(x='no. of customers paying after 30 days of due date',y='Profit percentage', data = df)
