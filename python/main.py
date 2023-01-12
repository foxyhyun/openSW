import numpy as np
from Stock import Stock
from predict import *
import matplotlib.pyplot as plt

coid = input("회사명 : ")
days = int(input("일 수"))
CO = Stock(coid)

prices, date = CO.returnPrices()
idx = CO.returnIdx()

prediction = []

for i in range(5):
    predictedPrice = predictNextDay(prices, idx)[0,0]
    prices = np.concatenate((prices, np.array([predictedPrice,prices[-1,1]]).reshape(1,-1)), axis=0)
    prediction.append(predictedPrice)

print(prediction)
