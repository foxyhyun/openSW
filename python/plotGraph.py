import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plotGraph(prices, dates):
    data_df = pd.DataFrame(prices[:,0], dates)
    total_len = len(data_df.index[1000:])
    plt.figure(figsize=(15,10))
    plt.plot(data_df.index[1000:1148], data_df[1000:1148], color='blue')
    plt.plot(data_df.index[1147:], data_df[1147:], color='red')
    plt.xticks(np.arange(0, total_len+1, 10), rotation=45)
    plt.show()
    