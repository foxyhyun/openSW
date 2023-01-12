import matplotlib.pyplot as plt
import pandas as pd

def plotGraph(prices, dates):
    data_df = pd.DataFrame(prices[:,0], dates)
    
    plt.figure(figsize=(15,10))
    plt.plot(data_df.index[1000:1148], data_df[1000:1148], color='blue')
    plt.plot(data_df.index[1147:], data_df[1147:], color='red')
    plt.show()