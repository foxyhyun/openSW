import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs

import urllib
from urllib import parse


def normalize_scale(inp):# 환율데이터 표본적어서 *10
    result = []
    for i in inp:
        result.append(((i-np.mean(inp))/np.var(inp))*10)
    return result

def normalize(inp):# *10안함
    result = []
    for i in inp:
        result.append(((i-np.mean(inp))/np.var(inp)))
    return result

def normalizeAmount(array): #거래량 정규화 된 함수
    n=int(np.mean(array))
    count=10
    while(n != 0):
        n = int(n/10);
        if n == 0 : break
        count=count*10;
    return count

class Stock:
    
    Exchange='./Data/Exchange.csv' #객체가 공통으로 사용할 환율 데이터 (동적크롤링할줄몰라서 csv로 받아옴)
    
    def __init__(self,code):
        self.code=code
    @staticmethod  
    def codeName(name):
        csv = pd.read_csv("./Code.csv",encoding='cp949')
        csv = np.array(csv)
        data=csv[:][:]
        for i in range(len(data)):
            if(data[i][3] == name):
                return data[i][1];
    @staticmethod
    def exchangeRate():  ##1달간격의 환율의 종가데이터  1행배열 57요소( 18.05.01~  23.01.01) 배열로 출력 SIZE(1,57)
        csv = pd.read_csv(Stock.Exchange,encoding='utf8')
        #print(csv)  환율 전체데이터 확인하기            
        csv = np.array(csv)
        data=[]
        for i in range(len(csv)):   
            fnum=""
            for j in csv[i][1]:
                if(j == ','):
                    continue
                fnum+=j
            data.append(float(fnum))
        data = np.array(data)
        data = data.astype('float32')
        data = normalize_scale(data) 
        return data
    @staticmethod
    def consumIdx():
        csv = pd.read_csv("./totalIdx.csv",encoding='cp949')
        csv  =np.array(csv)
        csv = csv[0][1:]
        csv = csv.reshape(-1,1)
        return csv
    @staticmethod
    def kospiIdx():
        csv = pd.read_csv("./totalIdx.csv",encoding='cp949')
        csv  =np.array(csv)
        csv = csv[1][1:]
        csv = csv.reshape(-1,1)
        return csv

    def returnIdx(self):##ROA, ROE, EPS, BPS, DPS, PER, PBR의 각 항목당 (2018.12 ~ 2023.12)의 데이터 6개 갖음 SIZE(7,8)
        get_param = {
            'pGB':1,
            'gicode':'A%s'%(self.code),
            'cID':'',
            'MenuYn':'Y',
            'ReportGB':'',
            'NewMenuID':101,
            'stkGb':701,
        }
        get_param = parse.urlencode(get_param)
        url="http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?%s"%(get_param)
        tables = pd.read_html(url, header=0,encoding='utf-8')
        sit=np.array(tables[11]) 

        data=[]#정규화 시작
    
        for i in range(6):
            crw=[]
            if (i == 2 or i==3): continue
            for j in range(7):
                
                if(sit[18+i][j+1] != sit[18+i][j+1]):
                    idx =  (float(sit[18+i][j]) + float(sit[18+i][j+2]))/2.0
                    crw.append(idx)
                    continue
                idx = float(sit[18+i][j+1])
                crw.append(idx)
            data.append(normalize(crw))
        data=np.array(data)
        return data

    
    def returnPrices(self):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.272 Whale/2.9.118.16 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'}
        url = "https://finance.naver.com/item/sise_day.nhn?code=%s"%(self.code)
        df = pd.DataFrame()
        for i in range(1, 121):
            page_url = '{}&page={}'.format(url, i)
            req = urllib.request.Request(page_url, headers=headers)
            response = urllib.request.urlopen(req)
            html = bs(response.read(), 'lxml')    
            df = pd.concat([df, pd.read_html(str(html), header=0)[0]])
        df = df.dropna()
        arr = df.to_numpy()
        arr1 = np.flipud(arr[:,1].reshape(-1,1))/10000
        arr2 = np.flipud(arr[:,6].reshape(-1,1))
        arr2 /= normalizeAmount(arr2)
        date = np.flipud(arr[:,0].reshape(-1,1))
        return np.concatenate((arr1, arr2), axis=1), date
    

    def showChart(self): ##종목코드의 제무재표를 나타내는 함수
        get_param = {
            'pGB':1,
            'gicode':'A%s'%(self.code),
            'cID':'',
            'MenuYn':'Y',
            'ReportGB':'',
            'NewMenuID':101,
            'stkGb':701,
        }
        get_param = parse.urlencode(get_param)
        url="http://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?%s"%(get_param)
        tables = pd.read_html(url, header=0,encoding='utf-8')
        return tables[11]