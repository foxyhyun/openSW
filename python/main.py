import numpy as np
from Stock import Stock
from predict import predictNextDay
from plotGraph import plotGraph


coid = input("회사명 : ")
days = int(input("일 수"))
CO = Stock(coid)

prices, date = CO.returnPrices()
idx = CO.returnIdx()

prices=prices[-1148:,:]
dates = date[-1148:,0].tolist()

for i in range(days):
    predictedPrice = predictNextDay(prices, idx)[0,0]
    prices = np.concatenate((prices, np.array([predictedPrice,prices[-1,1]]).reshape(1,-1)), axis=0)
    dates.append("%s일 후"%(i+1))
    
plotGraph(prices, dates)

