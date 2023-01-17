import numpy as np


from Stock import Stock
def returnInput(price, idx):
    price2018 = price[0:159]
    price2019 = price[159:405]
    price2020 = price[405:653]
    price2021 = price[653:901]
    price2022 = price[901:1147]
    price2023 = price[1147:]
    
    idxArray = np.concatenate((np.full((len(price2018),6), idx[:,0]),
                            np.full((len(price2019),6), idx[:,1]),
                            np.full((len(price2020),6), idx[:,2]),
                            np.full((len(price2021),6), idx[:,3]),
                            np.full((len(price2022),6), idx[:,4]),
                            np.full((len(price2023),6), idx[:,5])), axis=0)
    
    arr = np.concatenate((price, idxArray), axis=1)
    array = np.array(arr)
    
    er = Stock.exchangeRate()
    er = np.array(er).reshape(-1, 1)
    
    exchange = np.full((11,1), er[-57])
    exchange = np.concatenate((exchange, np.full((21,1), er[-56]),
                               np.full((22,1), er[-55]),
                               np.full((19,1), er[-54]),
                               np.full((20,1), er[-53]),
                               np.full((22,1), er[-52]),
                               np.full((21,1), er[-51]),
                               np.full((20,1), er[-50]),
                               np.full((21,1), er[-49]),
                               np.full((21,1), er[-48]),
                               np.full((21,1), er[-47]),
                               np.full((18,1), er[-46]),
                               np.full((20,1), er[-45]),
                               np.full((22,1), er[-44]),
                               np.full((22,1), er[-43]),
                               np.full((19,1), er[-42]),
                               np.full((19,1), er[-41]),
                               np.full((21,1), er[-40]),
                               np.full((22,1), er[-39]),
                               np.full((22,1), er[-38]),
                               np.full((19,1), er[-37]),
                               np.full((22,1), er[-36]),
                               np.full((22,1), er[-35]),
                               np.full((18,1), er[-34]),
                               np.full((20,1), er[-33]),
                               np.full((21,1), er[-32]),
                               np.full((21,1), er[-31]),
                               np.full((19,1), er[-30]),
                               np.full((21,1), er[-29]),
                               np.full((20,1), er[-28]),
                               np.full((23,1), er[-27]),
                               np.full((22,1), er[-26]),
                               np.full((19,1), er[-25]),
                               np.full((20,1), er[-24]),
                               np.full((22,1), er[-23]),
                               np.full((20,1), er[-22]),
                               np.full((20,1), er[-21]),
                               np.full((20,1), er[-20]),
                               np.full((21,1), er[-19]),
                               np.full((21,1), er[-18]),
                               np.full((19,1), er[-17]),
                               np.full((21,1), er[-16]),
                               np.full((23,1), er[-15]),
                               np.full((19,1), er[-14]),
                               np.full((21,1), er[-13]),
                               np.full((22,1), er[-12]),
                               np.full((20,1), er[-11]),
                               np.full((17,1), er[-10]),
                               np.full((22,1), er[-9]),
                               np.full((19,1), er[-8]),
                               np.full((22,1), er[-7]),
                               np.full((21,1), er[-6]),
                               np.full((17,1), er[-5]),
                               np.full((22,1), er[-4]),
                               np.full((22,1), er[-3]),
                               np.full((19,1), er[-2]),
                               np.full((7,1), er[-1])), axis=0)
    train_inputs = np.concatenate((array[-1148:], np.flipud(exchange.reshape(-1,1))), axis=1)
    return train_inputs[:-1,:].astype(float)
