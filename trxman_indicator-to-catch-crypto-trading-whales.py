from binance.client import Client
client = Client('API KEY','API SECRET')
import matplotlib.pyplot as plt
symbol = 'MTLBTC'
d = (client.get_order_book(symbol=symbol))  #get market depth
buys = []  #to store depth buy orders
sells = [] #to store depth sell orders

buy = (d['bids']) #extract bids orders 
for ordr in buy:
    volume = ordr[1]
    buys.append(float(volume))


for x in d['asks']: #extract asks orders
    v = x[1]
    sells.append(float(v))

print(sum(buys)) #print total bids
print(sum(sells)) # print total asks
print('buy - sell: ' + str(sum(buys) - sum(sells)))  #print difference between bids and asks

b_qty = [] #storing buy orders/quantity
trades = client.get_recent_trades(symbol='MTLBTC') #get recent trades of this pair 

print(trades)

for x in range(500):
    if trades[int(f"{x}")]['isBuyerMaker'] == False:  #here we get bids
        b_data = trades[int(f"{x}")]['qty'] #buy order qty
        b_qty.append(float(b_data))



s_qty = [] #storing sell orders/quantity
for i in range(500):
    if trades[int(f"{i}")]['isBuyerMaker'] == True: #here we get asks
        s_data = trades[int(f"{i}")]['qty'] #sell order qty 
        s_qty.append(float(s_data))
dd = [x for x in range(len(b_qty))] # creating list for b_qty x-axis
ddd = [x for x in range(len(s_qty))] # creating list for s_qty x-axis

plt.style.use('ggplot') # here i used ggplot style. You can choose what you like: print(plt.style.available)

plt.grid(True) # add a grid to the plot to make it more precise in analysis

plt.ylabel(('Token Order Qty'), fontsize=17) # volume of the trade in the analysed token, here MTL

plt.xlabel("x trade", fontsize=17) # index of the trade 

plt.title((f"{symbol} Recent Buys Orders Volumes") , fontsize=20) #plot title

plt.plot(ddd, [x*-1 for x in s_qty ], c='red', label='sell orders') # here i multiply each element in s_qty by -1 
                                                                    #to make a negative y value which will make the
                                                                    #visualization better 

plt.plot(dd, b_qty, c='blue', label='buy orders') #plotting buy orders
plt.legend() #adding a legend for the labels 
plt.show()
for i in b_qty:
    count = b_qty.count(i)
    print(count, i)