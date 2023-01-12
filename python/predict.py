from returnInputs import *
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model

dmodel = load_model('dailymodel.h5')

def predictNextDay(prices, idx):
    pre_inputs = []
    prices=prices[-1148:]
    idx=idx[-1148:]
    pre_inputs.append(returnInput(prices, idx))
    pre_inputs = np.array(pre_inputs)
    return dmodel.predict(pre_inputs)
