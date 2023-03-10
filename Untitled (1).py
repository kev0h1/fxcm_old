#!/usr/bin/env python
# coding: utf-8

# In[51]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from stocktrends import Renko
import statsmodels.api as sm
import time
import numpy as np
import copy


# In[6]:


get_ipython().system('{sys.executable} -m pip install statsmodels')


# In[7]:


get_ipython().system('{sys.executable} -m pip install stocktrends')


# In[12]:


import fxcmpy
import time
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import copy


# In[15]:


TOKEN = "912ffdbeaec31419ef155a6cdc666d4dc28fb69c"


# In[17]:


con = fxcmpy.fxcmpy(access_token = TOKEN, log_level = 'error')


# In[10]:


pairs =['EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD']
pos_size = 10


# In[11]:


def main():
    try:
        open_pos = con.get_open_positions()
        for currency in pairs:
            data = con.get_candles(currency, period='m5', number=250)
            ohlc = data.iloc[:,[0,1,2,3,8]]
            ohlc.columns = ["Open","Adj Close","High","Low","Volume"]
            signal = np.random.randint(0,3)
            if(signal == 1):
                con.close_all_for_symbol(currency)
                con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                
                print("New long position initiated for ", currency)
            elif (signal ==2):
                con.close_all_for_symbol(currency)
                con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                print("New short position initiated for ", currency)
    except:
        print("error encountered....skipping this iteration")


# In[ ]:


starttime=time.time()
timeout = time.time() + 60*60*24  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep((60*5) - ((time.time() - starttime) % (60*5))) # 5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()


# In[47]:


def MACD(DF,a,b,c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return df


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

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    df2 = Renko(df)
    df2.brick_size = max(0.5,round(ATR(DF,120)["ATR"][-1],0))
    renko_df = df2.get_bricks()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df

def renko_merge(DF):
    "function to merge renko  with original ohlc df"
    df = copy.deepcopy(DF)
    renko = renko_DF(df)
    df['date']= df.index
    merged_df= df.merge(renko.loc[:, ['date','bar_num']], how='outer', on='date')
    merged_df['bar_num'].fillna(method='ffill', inplace=True)
    merged_df['macd']=MACD(merged_df,12,26,9)[0]
    merged_df['macd_sig']=MACD(merged_df,12,26,9)[1]
    merged_df['macd_slope']=slope(merged_df['macd'],5)
    merged_df['macd_sig_slope']=slope(merged_df['macd_sig'],5)
    return merged_df


# In[21]:


currency = 'EUR/USD'
data = con.get_candles(currency, period='m5', number=250)
ohlc = data.iloc[:,[0,1,2,3,8]]
ohlc.columns = ["Open","Adj Close","High","Low","Volume"]


# In[40]:


macd = MACD(ohlc, 12, 24, 9)
macd.head()


# In[41]:


macd[['Signal', 'MACD']].plot()


# In[42]:


def trade_signal(MERGED_DF,l_s):
    "function to generate signal"
    signal = ""
    df = copy.deepcopy(MERGED_DF)
    if l_s == "":
        if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Buy"
        elif df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Sell"
            
    elif l_s == "long":
        if df["bar_num"].tolist()[-1]<=-2 and df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Close_Sell"
        elif df["macd"].tolist()[-1]<df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]<df["macd_sig_slope"].tolist()[-1]:
            signal = "Close"
            
    elif l_s == "short":
        if df["bar_num"].tolist()[-1]>=2 and df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Close_Buy"
        elif df["macd"].tolist()[-1]>df["macd_sig"].tolist()[-1] and df["macd_slope"].tolist()[-1]>df["macd_sig_slope"].tolist()[-1]:
            signal = "Close"
    return signal


# In[55]:


renk = renko_DF(ohlc)


# In[56]:


df = Renko(ohlc)


# In[57]:


df = ohlc.copy()


# In[59]:


df.reset_index(inplace=True)
df = df.iloc[:,[0,1,2,3,4,5]]
df.columns = ["date","open","high","low","close","volume"]

df2 = Renko(df)


# In[60]:


df2


# In[63]:


df2.brick_size


# In[ ]:




