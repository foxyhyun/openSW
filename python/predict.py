from returnInputs import returnrInput
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model

dmodel = load_model('dmodel.h5')

def predictNextDay(prices, idx):
    pre_inputs = []
    prices=prices[-1127:]
    idx=idx[-1127:]
    pre_inputs.append(returnrInput(prices, idx))
    pre_inputs = np.array(pre_inputs)
    return dmodel.predict(pre_inputs)
