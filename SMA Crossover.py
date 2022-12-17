#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import fxcmpy
import time
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# In[2]:


TOKEN = "c1d10848083badbc16664ddd3f69f2d9188acb9c"

con = fxcmpy.fxcmpy(access_token = TOKEN, log_level = 'error')

#pairs =['EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD']
pairs = ['EUR/USD','USD/JPY','AUD/USD','GBP/USD','USD/CAD','NZD/USD','AUD/CAD','AUD/JPY','CAD/JPY','CHF/JPY','EUR/AUD','EUR/CAD','EUR/CHF','EUR/GBP','EUR/JPY','GBP/CHF','GBP/JPY','USD/CHF','AUD/CHF','AUD/NZD','CAD/CHF','EUR/NZD','GBP/AUD','GBP/CAD','GBP/NZD','NZD/CAD','NZD/CHF','NZD/JPY']
pos_size = 10

upward_sma_dir = {}
dnward_sma_dir = {}
for i in pairs:
    upward_sma_dir[i] = False
    dnward_sma_dir[i] = False


# In[3]:


def get_currency_sentiment():
    url = "https://www.dailyfx.com/forex-rates"
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    currency_information = soup.find_all("div", {"class": "dfx-singleInstrument__nameAndSignal d-flex flex-column h-100 mr-1 justify-content-around align-self-start text-dark"})
    currency_sentiment = {}
    for info in currency_information:
        currency_sentiment[info.a.string] = info.find_all('span')[-1].string
    return currency_sentiment

def get_supports_and_resistances(currency_name):
    url = "https://www.dailyfx.com/" + currency_name
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    currency_support_resistances = soup.find_all("div", {"class": "dfx-supportResistanceBlock__values"})
    support_resistance={}
    previous_SP = ""
    lists = []
    for info in currency_support_resistances:
        count =0
        for item in info.find_all('span'):
            if item.string.startswith("S") or item.string.startswith("R"):
                previous_SP = item.string
            else:
                support_resistance[previous_SP] = [float(item.string), "", count]
                count +=1

    for index,item in enumerate(currency_support_resistances[0].find_all("div", {"class":strengths})):
        support_resistance[support_resistance_values[index]][1]= get_strength(str(item))
        
    return support_resistance

def get_daily_pivot_points(currency_name):
    url = "https://www.dailyfx.com/" + currency_name
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    pivots = soup.find_all("div", {"class": "d-flex px-0 py-1 col-4"})
    pivot_values={}
    for item in pivots:
        values =item.find_all("span")
        pivot_values[values[0].string.strip()]= float(values[1].string.strip())
    return pivot_values


# In[4]:


def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return (df["MACD"],df["Signal"])

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def renko_merge(DF):
    "function to merge renko  with original ohlc df"
    df = copy.deepcopy(DF)
    renko = renko_DF(df)
    #df['date']= df.index
    #df.set_index('date', inplace=True)
    merged_df= df.merge(renko.loc[:, ['date','bar_num']], how='outer', on='date')
    merged_df['bar_num'].fillna(method='ffill', inplace=True)
    merged_df['macd']=MACD(merged_df,12,26,9)[0]
    merged_df['macd_sig']=MACD(merged_df,12,26,9)[1]
    merged_df['macd_slope']=slope(merged_df['macd'],5)
    merged_df['macd_sig_slope']=slope(merged_df['macd_sig'],5)
    return merged_df

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5,round(ATR(DF,120)["ATR"][-1],0))
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df

def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)


# In[5]:


def trade_signal(df,curr):
    "function to generate signal"
    global upward_sma_dir, dnward_sma_dir
    signal = ""
    if df['sma_fast'][-1] > df['sma_slow'][-1] and df['sma_fast'][-2] < df['sma_slow'][-2]:
        upward_sma_dir[curr] = True
        dnward_sma_dir[curr] = False
    if df['sma_fast'][-1] < df['sma_slow'][-1] and df['sma_fast'][-2] > df['sma_slow'][-2]:
        upward_sma_dir[curr] = False
        dnward_sma_dir[curr] = True  
    if upward_sma_dir[curr] == True and min(df['K'][-1],df['D'][-1]) > 25 and max(df['K'][-2],df['D'][-2]) < 25:
        signal = "Buy"
    if dnward_sma_dir[curr] == True and min(df['K'][-1],df['D'][-1]) > 75 and max(df['K'][-2],df['D'][-2]) < 75:
        signal = "Sell"
    
    plt.subplot(211)
    plt.plot(df.loc[:,['Adj Close','sma_fast','sma_slow']])
    plt.title('SMA Crossover & Stochastic')
    plt.legend(('close','sma_fast','sma_slow'),loc='upper left')
    
    plt.subplot(212)
    plt.plot(df.loc[:,['K','D']])
    #plt.hlines(y=25,xmin=0,xmax=len(df),linestyles='dashed')
    #plt.hlines(y=75,xmin=0,xmax=len(df),linestyles='dashed')
    
    plt.show()
    
    return signal


# In[6]:


def get_data(currency_pair, per, candle_number):
    data = con.get_candles(currency, period=per, number=candle_number)
    ohlc = data.iloc[:,[0,1,2,3,8]]
    ohlc.columns = ["Open","Adj Close","High","Low","Volume"]
    ohlc['weekday'] = ohlc.index.dayofweek
    return ohlc


# In[7]:


def stochastic(df,a,b,c):
    "function to calculate stochastic"
    df['k']=((df['Adj Close'] - df['Low'].rolling(a).min())/(df['High'].rolling(a).max()-df['Low'].rolling(a).min()))*100
    df['K']=df['k'].rolling(b).mean() 
    df['D']=df['K'].rolling(c).mean()
    return df

def SMA(df,a,b):
    "function to calculate stochastic"
    df['sma_fast']=df['Adj Close'].rolling(a).mean() 
    df['sma_slow']=df['Adj Close'].rolling(b).mean() 
    return df


# In[45]:


def main():
    try:
        open_pos = con.get_open_positions()
        for currency in pairs:
            tradeId=""  
            ohlc = get_data(currency, 'm5', 250)
            #signal = trade_signal(ohlc, currency)
            signal = pivot_with_sentiment_signal(currency, ohlc)
            if(signal == "Buy"):
                if(len(open_pos[open_pos["currency"]==currency])==1):
                    index = open_pos[open_pos["currency"]==currency].index[0]
                    tradeId = open_pos[open_pos["currency"]==currency]['tradeId'][index]
                    if(open_pos[open_pos["currency"]==currency]['isBuy'][index]==False):
                        con.close_trade(tradeId, amount=pos_size)
                        con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')                        
                else:
                    con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                
                print("New long position initiated for ", currency)
                
            elif (signal =="Sell"):
                if(len(open_pos[open_pos["currency"]==currency])==1):
                    index = open_pos[open_pos["currency"]==currency].index[0]
                    tradeId = open_pos[open_pos["currency"]==currency]['tradeId'][index]
                    if(open_pos[open_pos["currency"]==currency]['isBuy'][index]==True):
                        con.close_trade(tradeId, amount=pos_size)
                        con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                else:                        
                    con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                print("New short position initiated for ", currency)
    except:
        print("error encountered....skipping this iteration")


# In[11]:


def get_currency_name_for_pivot(curr):
    currency_name = curr.replace("/", "-").lower()
    return currency_name


# In[32]:


def pivot_with_sentiment_signal(curr, df):
    sentiment = get_currency_sentiment()
    currency_sentiment = sentiment[curr]
    pivot_name = get_currency_name_for_pivot(curr)
    close = df.iloc[-1]['Adj Close']
    pivot_table = get_daily_pivot_points(pivot_name)
    pivot = pivot_table['P']
    s1 = pivot_table['S1']
    r1 = pivot_table['R1']
    
    if close>pivot and close<r1 and currency_sentiment=="Bullish":
        return "Buy"
    
    elif close<pivot and close>s1 and currency_sentiment=="Bearish":
        return "Sell"
    
    else:
        return ""
    


# In[48]:


get_daily_pivot_points("eur-usd")['S1']


# In[ ]:


starttime=time.time()
timeout = time.time() + 60*60*365  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep((60*5) - ((time.time() - starttime) % (60*5))) # 5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()

