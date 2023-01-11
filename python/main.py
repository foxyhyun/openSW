# import tensorflow as tf
import numpy as np
# from tensorflow.keras import models
from tensorflow.keras.models import load_model

# import matplotlib.pyplot as plt

from returnInputs import *
from Stock import Stock

dmodel = load_model('dailymodel.h5')
wmodel = load_model('weeklymodel.h5')
mmodel = load_model('monthlymodel.h5')

coid = input("회사명")

CO = Stock(coid)

a=CO.returnPrices()
data=CO.returnIdx()

pre_inputs = []

a=a[-1148:]
data=data[-1148:]
pre_inputs.append(returnInput(a, data))
pre_inputs = np.array(pre_inputs)
print(dmodel.predict(pre_inputs))