#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U fxcmpy-1.2.6.tar.gz')


# In[3]:


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


# In[ ]:


import json
import requests
import socketio


# In[ ]:


socketIO = socketio.Client()
TRADING_API_URL = 'https://api-demo.fxcm.com'
WEBSOCKET_PORT = 443
ACCESS_TOKEN = 'f2eb943fd86c784d55dcd79906d7e165d2f7fdb3'
def on_connect():
    print ('Websocket connected: ' + socketIO.eio.sid)

def on_close():

    print ('Websocket closed.')

def on_error():
    print ('Websocket error.')


COLLECTIONS = {}
socketIO.connect(TRADING_API_URL +":" +str(WEBSOCKET_PORT) +"/?access_token=" +ACCESS_TOKEN)
print (socketIO.eio.sid)
bearer_access_token="Bearer "+ socketIO.eio.sid + ACCESS_TOKEN
print (bearer_access_token)
socketIO.on('disconnect', on_close)
socketIO.on('connect', on_connect)
socketIO.on('connect_error', on_error)


# In[ ]:


socketIO


# In[ ]:


import smtplib
PASSWORD = "Roronoa_12"
SUBJECT = "FXCMPY Logs"
TO = "kevin.maingi12@gmail.com"
FROM = "kmaingi12@hotmail.com"
server = smtplib.SMTP(host='smtp-mail.outlook.com', port=587)
server.starttls()
server.login(FROM, PASSWORD)


# In[ ]:


def send_email(message):
    BODY = "\r\n".join((
            "From: %s" % FROM,
            "To: %s" % TO,
            "Subject: %s" % SUBJECT ,
            "",
            message
            ))
    server.sendmail(FROM, [TO], BODY)
    server.quit()


# In[ ]:


TRADING_API_URL = 'https://api-demo.fxcm.com:443'
WEBSOCKET_PORT = 443
TOKEN = "f2eb943fd86c784d55dcd79906d7e165d2f7fdb3"
socketIO =  SocketIO(TRADING_API_URL, WEBSOCKET_PORT, params={'access_token' : TOKEN})


# In[4]:


TOKEN = "f2eb943fd86c784d55dcd79906d7e165d2f7fdb3"

con = fxcmpy.fxcmpy(access_token = TOKEN, log_level = 'error')


# In[ ]:


#pairs =['EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD']
#pairs = ['EUR/USD','USD/JPY','AUD/USD','GBP/USD','USD/CAD','NZD/USD','AUD/CAD','AUD/JPY','CAD/JPY','CHF/JPY','EUR/AUD','EUR/CAD','EUR/CHF','EUR/GBP','EUR/JPY','GBP/CHF','GBP/JPY','USD/CHF','AUD/CHF','AUD/NZD','CAD/CHF','EUR/NZD','GBP/AUD','GBP/CAD','GBP/NZD','NZD/CAD','NZD/CHF','NZD/JPY']
pairs = ['EUR/USD','USD/JPY','AUD/USD','GBP/USD','USD/CAD','NZD/USD','AUD/JPY','EUR/CHF','EUR/GBP','EUR/JPY','GBP/JPY','USD/CHF', 'XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40']
pos_size = 5
stop = 20
upward_sma_dir = {}
dnward_sma_dir = {}
currency_s_r = {}
for i in pairs:
    upward_sma_dir[i] = False
    dnward_sma_dir[i] = False
    currency_s_r[i] = {}


# In[ ]:


alternative_dictionary = {'XAU/USD':'gold-price', 'XAG/USD':'silver-prices', 'BTC/USD': 'bitcoin', 'GER30':'dax-30', 'FRA40':'cac-40'}
alternative_sentiment_dictionary = {'XAU/USD':'Gold', 'XAG/USD':'Silver', 'BTC/USD': 'Bitcoin', 'GER30':'Germany 30', 'FRA40':'France 40'}


# In[ ]:


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


# In[ ]:


net_buy ={}
net_sell={}
def get_net_buy_sell():
    url = "https://www.dailyfx.com/sentiment"
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    currency_information = soup.find_all("div", {"class": "dfx-technicalSentimentCard__netLongContainer"})
    
    for i in range(0,len(currency_information)):
        instrument = currency_information[i].find_all("span")[1].attrs.get("data-market-id", None)
        value = currency_information[i].find_all("span")[1].attrs.get("data-value", None)
        net_buy[instrument]=value


    currency_information = soup.find_all("div", {"class": "dfx-technicalSentimentCard__netShortContainer"})
    for i in range(0,len(currency_information)):
        instrument = currency_information[i].find_all("span")[1].attrs.get("data-market-id", None)
        value = currency_information[i].find_all("span")[1].attrs.get("data-value", None)
        net_sell[instrument]=value


# In[ ]:


strengths = ["dfx-supportResistanceLevelStrength dfx-supportResistanceLevelStrength--strong dfx-supportResistanceBlock__valueLevelStrength mx-1 mx-md-0",
            "dfx-supportResistanceLevelStrength dfx-supportResistanceLevelStrength--moderate dfx-supportResistanceBlock__valueLevelStrength mx-1 mx-md-0",
            "dfx-supportResistanceLevelStrength dfx-supportResistanceLevelStrength--weak dfx-supportResistanceBlock__valueLevelStrength mx-1 mx-md-0"]

support_resistance_values = {0:'S1', 1:'S2', 2:'S3', 3:'R1', 4:'R2', 5:'R3'}

def get_currency_sentiment():
    url = "https://www.dailyfx.com/forex-rates"
    try:
        page = requests.get(url)
        page_content = page.content
        soup = BeautifulSoup(page_content, 'html.parser')
        currency_information = soup.find_all("div", {"class": "dfx-singleInstrument__nameAndSignal d-flex flex-column h-100 mr-1 justify-content-around align-self-start text-dark"})
        currency_sentiment = {}
        for info in currency_information:
            currency_sentiment[info.a.string] = info.find_all('span')[-1].string
        return currency_sentiment
    except SSLError:
        return {}


def get_supports_and_resistances(currency_name):
    url = "https://www.dailyfx.com/" + currency_name
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    currency_support_resistances = soup.find_all("div", {"class": "dfx-supportResistanceBlock__values"})
    support_resistance={}
    previous_SP = ""
    lists = []
    if(len(currency_support_resistances)>0):
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

def get_strength(soup_string):
    if "strong" in soup_string:
        return "strong"
    elif "moderate" in soup_string:
        return "moderate"
    elif "weak" in soup_string:
        return "weak"
    else:
        return ""

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


# In[ ]:


#my_list = supports_less_than_close(s_r, close)
#my_list2 = resistances_greater_than_close(s_r, close)
#get_highest_support(s_r, my_list)
#get_lowest_resistance(s_r, my_list2)

def supports_less_than_close(support_resistances, close):
    supports_to_consider = []
    if(support_resistances['S1'][0]<close):
        supports_to_consider.append('S1')
    if(support_resistances['S2'][0]<close):
        supports_to_consider.append('S2')
    if(support_resistances['S3'][0]<close):
        supports_to_consider.append('S3')
    return supports_to_consider

def resistances_greater_than_close(support_resistances, close):
    resistances_to_consider = []
    if(support_resistances['R1'][0]>close):
        resistances_to_consider.append('R1')
    if(support_resistances['R2'][0]>close):
        resistances_to_consider.append('R2')
    if(support_resistances['R3'][0]>close):
        resistances_to_consider.append('R3')
    return resistances_to_consider

def get_highest_support(support_resistances, supports_to_consider):
    supports = []
    for item in supports_to_consider:
        supports.append(support_resistances[item][0])
    
    return max(supports)

def get_lowest_resistance(support_resistances, resistances_to_consider):
    resistances = []
    for item in resistances_to_consider:
        resistances.append(support_resistances[item][0])
    
    return min(resistances)


# In[ ]:


def largest(num1, num2, num3):
    if (num1 > num2) and (num1 > num3):
        largest_num = num1
    elif (num2 > num1) and (num2 > num3):
        largest_num = num2
    else:
        largest_num = num3
    print("The largest of the 3 numbers is : ", largest_num)
def smallest(num1, num2, num3):
    if (num1 < num2) and (num1 < num3):
        smallest_num = num1
    elif (num2 < num1) and (num2 < num3):
        smallest_num = num2
    else:
        smallest_num = num3
    print("The smallest of the 3 numbers is : ", smallest_num)


# In[ ]:


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

def OBV(DF):
    """function to calculate On Balance Volume"""
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df


# In[ ]:


def trade_signal(df,curr, currency_sentiment):
    "function to generate signal"
    global upward_sma_dir, dnward_sma_dir
    signal = ""
    if df['sma_fast'][-1] > df['sma_slow'][-1] and df['sma_fast'][-2] < df['sma_slow'][-2]:
        upward_sma_dir[curr] = True
        dnward_sma_dir[curr] = False
    if df['sma_fast'][-1] < df['sma_slow'][-1] and df['sma_fast'][-2] > df['sma_slow'][-2]:
        upward_sma_dir[curr] = False
        dnward_sma_dir[curr] = True  
    if upward_sma_dir[curr] == True and min(df['K'][-1],df['D'][-1]) > 25 and max(df['K'][-2],df['D'][-2]) < 25 and currency_sentiment=="Bullish":
        signal = "Buy"
    if dnward_sma_dir[curr] == True and min(df['K'][-1],df['D'][-1]) > 75 and max(df['K'][-2],df['D'][-2]) < 75 and currency_sentiment=="Bearish":
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


# In[ ]:


def get_data(currency_pair, per, candle_number):
    data = con.get_candles(currency_pair, period=per, number=candle_number)
    ohlc = data.iloc[:,[0,1,2,3,8]]
    ohlc.columns = ["Open","Adj Close","High","Low","Volume"]
    ohlc['weekday'] = ohlc.index.dayofweek
    return ohlc


# In[ ]:


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


# In[ ]:


market_positions = ""
def main():
    #try:
    open_position = con.get_open_positions()
    
    for currency_pair in pairs:
        #try:
        alternative_list =['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40']
        sentiment = ""
        pivot_name = ""
        sentiment_dictionary = get_currency_sentiment()
        if sentiment_dictionary == {}:
            return
        if currency_pair not in alternative_list:
            sentiment = sentiment_dictionary[currency_pair]
            pivot_name = get_currency_name_for_pivot(currency_pair)
        else:
            sentiment = sentiment_dictionary[alternative_sentiment_dictionary[currency_pair]]
            pivot_name = alternative_dictionary[currency_pair]
            
        #print(sentiment)
        #print(pivot_name)
            
        tradeId=""  
        ohlc = get_data(currency_pair, 'm5', 250)
        #close = ohlc.iloc[-1]['Adj Close']
        ohlc_df = stochastic(data,14,3,3)
        ohlc_df = SMA(ohlc_df,100,200)
        signal = trade_signal(ohlc_df, currency, sentiment)
        #signal = pivot_with_sentiment_signal(currency_pair, ohlc, sentiment, pivot_name, close)
        #print("signal " + signal)
        s_r = get_supports_and_resistances(pivot_name)
        support = -1
        resistance = -1
        
        if len(s_r)>0:
            my_list = supports_less_than_close(s_r, close)
            my_list2 = resistances_greater_than_close(s_r, close)
            if len(my_list)>0:
                support = get_highest_support(s_r, my_list)
            if len(my_list2)>0:
                resistance = get_lowest_resistance(s_r, my_list2)
        
        close_if_support_resistances_exceeded(open_position, close, currency_pair)
        #print("support "+ str(support))
        #print("resistance " +str(resistance))
        #print("==============")
        #print("")

        if(sentiment=="Bullish"):
            if(len(open_position)>0):
                close_sell(open_position,"", currency_pair)
                currency_s_r[currency_pair]={}

        elif(sentiment =="Bearish"):
            if(len(open_position)>0):
                close_buy(open_position,"", currency_pair)
                currency_s_r[currency_pair]={}

        if(signal == "Buy"):
            if(len(open_position)>0):
                close_sell(open_position, signal, currency_pair) 
            else:
                con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=True, amount=pos_size, 
                           time_in_force='GTC', stop=-stop, limit=stop, trailing_step =True, order_type='AtMarket')
                
                print("New long position initiated for ", currency_pair)
                message = "New long position initiated for " + currency_pair
                market_positions += message + "\r"
                
            currency_s_r[currency_pair]={"support":support, "resistance":resistance}

        elif (signal =="Sell"):
            if(len(open_position)>0):
                close_buy(open_position, signal, currency_pair)
            else:                        
                con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=True, amount=pos_size, 
                           time_in_force='GTC', stop=-stop, limit=stop, trailing_step =True, order_type='AtMarket')
                print("New short position initiated for ", currency_pair)
                message = "New short position initiated for ", currency_pair
                market_positions += message + "\r"
                
            currency_s_r[currency_pair]={"support":support, "resistance":resistance}
            
        #except KeyError:
                #print("Error with currency pair " + currency_pair)
    #except:
        #print("error encountered....skipping this iteration")


# In[ ]:


def close_if_support_resistances_exceeded(open_pos, _close, currency):
    if(len(open_pos)>0):
        if(len(open_pos[open_pos["currency"]==currency])==1):
            index = open_pos[open_pos["currency"]==currency].index[0]
            tradeId = open_pos[open_pos["currency"]==currency]['tradeId'][index]
            if currency_s_r[currency] != {}:
                if _close<= currency_s_r[currency]['support'] or _close>= currency_s_r[currency]['resistance']:
                    con.close_trade(tradeId, amount=pos_size)
                currency_s_r[currency] = {}

def close_buy(open_pos, sig, currency):
    if(len(open_pos[open_pos["currency"]==currency])==1):
        index = open_pos[open_pos["currency"]==currency].index[0]
        tradeId = open_pos[open_pos["currency"]==currency]['tradeId'][index]
        if(open_pos[open_pos["currency"]==currency]['isBuy'][index]==True):
            con.close_trade(tradeId, amount=pos_size)
            if(sig =="Sell"):
                con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, 
                                   time_in_force='GTC', stop=-stop, limit=stop,trailing_step =True, order_type='AtMarket')
                print("New short position initiated for ", currency)
                message = "New short position initiated for ", currency_pair
                market_positions += message + "\r"
    
def close_sell(open_pos, sig, currency):
    if(len(open_pos[open_pos["currency"]==currency])==1):
        index = open_pos[open_pos["currency"]==currency].index[0]
        tradeId = open_pos[open_pos["currency"]==currency]['tradeId'][index]
        if(open_pos[open_pos["currency"]==currency]['isBuy'][index]==False):
            con.close_trade(tradeId, amount=pos_size)
            if(sig=="Buy"):
                con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, 
                                   time_in_force='GTC', stop=-stop, limit=stop,trailing_step =True, order_type='AtMarket')                        
                print("New long position initiated for ", currency)
                message = "New long position initiated for " + currency_pair
                market_positions += message + "\r"


# In[ ]:


def get_currency_name_for_pivot(curr):
    currency_name = curr.replace("/", "-").lower()
    return currency_name


# In[ ]:


def pivot_with_sentiment_signal(curr, df, currency_sentiment, pivot_nm, _close):  
    pivot_table = get_daily_pivot_points(pivot_nm)
    #print(pivot_table)
    if len(pivot_table)== 0:
        return ""
    
    pivot = pivot_table['P']
    s1 = pivot_table['S1']
    r1 = pivot_table['R1']
    
    obv = OBV(df)
    obv['slope'] = slope(obv["obv"],5)
    angle = obv['slope'].iloc[-1]
    
    if _close<r1 and currency_sentiment=="Bullish" and angle>10:
        return "Buy"
    
    elif _close>s1 and currency_sentiment=="Bearish" and angle <-10:
        return "Sell"
    
    return ""
    


# In[ ]:


starttime=time.time()
timeout = time.time() + 60*60*365  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        robot_message = "passthrough at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\r"
        print(robot_message)
        #con = fxcmpy.fxcmpy(access_token = TOKEN, log_level = 'error')
        try:
            main()
            robot_message += message
            message = ""
        except OSError:
            send_email("fxcmpy error! ==> Connection aborted, failed to reconnect")        
        send_email(str(robot_message))
        time.sleep((60*5) - ((time.time() - starttime) % (60*5))) # 5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()


# In[ ]:


url = "https://www.dailyfx.com/sentiment-report"
page = requests.get(url)
page_content = page.content
soup = BeautifulSoup(page_content, 'html.parser')
currency_support_resistances = soup.find_all("td")


# In[ ]:


currency_support_resistances[11]


# In[ ]:




