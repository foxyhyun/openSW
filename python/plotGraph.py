import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotGraph(prices, dates):
    data_df = pd.DataFrame(prices[:,0], dates)
    total_len = len(data_df.index[0:])
    plt.figure(figsize=(15,10))
    plt.plot(data_df.index[0:1127], data_df[0:1127], color='blue')
    plt.plot(data_df.index[1127:], data_df[1127:], color='red')
    plt.xticks(np.arange(0, total_len+1, 20), rotation=90)
    plt.show()
    