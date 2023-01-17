import numpy as np
from Stock import Stock
from predict import predictNextDay
from plotGraph import plotGraph


coname = input("회사명 : ")
coid = Stock.codeName(coname)
days = int(input("일 수 : "))
CO = Stock(coid)

prices, date = CO.returnPrices()
idx = CO.returnIdx()

prices=prices[-1148:,:]
dates = date[-1148:,0].tolist()

for i in range(days):
    predictedPrice = predictNextDay(prices, idx)[0,0]
    prices = np.concatenate((prices, np.array([predictedPrice,prices[-1,1]]).reshape(1,-1)), axis=0)
    dates.append("D+%s"%(i+1))
    
plotGraph(prices, dates)
