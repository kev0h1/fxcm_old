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


import smtplib
import re
import sys


# In[3]:


def send_email(message, subject):
    PASSWORD = "Roronoa_12"
    SUBJECT = subject
    TO = "kevin.maingi12@gmail.com"
    FROM = "kmaingi12@hotmail.com"
    try:
        server = smtplib.SMTP(host='smtp-mail.outlook.com', port=587)
        server.starttls()
        server.login(FROM, PASSWORD)
        BODY = "\r\n".join((
            "From: %s" % FROM,
            "To: %s" % TO,
            "Subject: %s" % SUBJECT,
            "",
            message + "\r :)"
        ))
        server.sendmail(FROM, [TO], BODY)
        server.quit()
    except:
        print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
        send_email("Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(sys.exc_info()[2]),
                   "Exceptional Error")


# In[4]:


TOKEN = "f2eb943fd86c784d55dcd79906d7e165d2f7fdb3"

con = fxcmpy.fxcmpy(access_token = TOKEN, log_level = 'error')


# In[5]:


# pairs = ['EUR/USD','USD/JPY','AUD/USD','GBP/USD','USD/CAD','NZD/USD','AUD/JPY','EUR/CHF','EUR/GBP','EUR/JPY','GBP/JPY','USD/CHF', 'XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD"]
pairs = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'NZD/USD', 'AUD/JPY', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY',
         'GBP/JPY', 'USD/CHF', 'BTC/USD', "ETH/USD", "LTC/USD"]
pos_size = 5
stop = 20
upward_sma_dir = {}
dnward_sma_dir = {}
currency_s_r = {}
for i in pairs:
    upward_sma_dir[i] = False
    dnward_sma_dir[i] = False
    currency_s_r[i] = {}


# In[6]:


risk_percentage = 1
risk = risk_percentage*0.01


# In[7]:


alternative_dictionary = {'XAU/USD':'gold-price', 'XAG/USD':'silver-prices', 'BTC/USD': 'bitcoin', 'GER30':'dax-30', 'FRA40':'cac-40', 'ETH/USD':'ether-eth', 'LTC/USD':'litecoin-ltc'}
alternative_sentiment_dictionary = {'XAU/USD':'Gold', 'XAG/USD':'Silver', 'BTC/USD': 'Bitcoin', 'GER30':'Germany 30', 'FRA40':'France 40'}
full_sentiment_dictionary={"BTC/USD":"Bitcoin", "GER30":"Germany 30"
                          , "ETH/USD":"Ethereum", "FRA40":"France 40", "XAU/USD":"Gold"
                          , "LTC/USD":"Litecoin", "XAG/USD":"Silver"
                          }
full_pivot_table_dictionary={"BTC/USD":"Bitcoin","ETH/USD":"Ethereum",
                            "LTC/USD":"Litecoin","XAU/USD":"Gold","XAG/USD":"Silver",
                            "GER30":"Germany30", "FRA40":"France40"}


# In[8]:


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


# In[9]:


def get_net_buy_sell():
    currency_dictionary_full={}
    url = "https://www.dailyfx.com/sentiment-report"
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    currency_information = soup.find_all("tr")
    for i in range(1,len(currency_information)):
        currency_dictionary_full[currency_information[i].find_all("span")[0].contents[0]]={"Sentiment": currency_information[i].find_all("span")[1].contents[0],
                                 "Net Long": currency_information[i].find_all("span")[2].contents[0],
                                  "Net Short": currency_information[i].find_all("span")[3].contents[0]
                                 }
    return currency_dictionary_full

def p2f(x):
    return float(x.strip('%'))
                             


# In[10]:


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


# In[11]:


def full_pivot_table():
    url = "https://www.dailyfx.com/pivot-points"
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    pivots = soup.find_all("div", {"class": "dfx-pivotPointsTable dfx-pivotPointsWidget__table container pt-2 pt-md-0 dfx-font-size-3"})
    pivot_dictionary={}
    for i in range(0, len(pivots)):
        try:
            spans = pivots[i].find_all("span")
            pivot_dictionary[re.sub('\s+','',pivots[i].a.contents[0])]={re.sub('\s+','',spans[0].contents[0]): float(re.sub('\s+','',spans[1].contents[0])),
                                                                   re.sub('\s+','',spans[2].contents[0]):float(re.sub('\s+','',spans[3].contents[0])),
                                                                    re.sub('\s+','',spans[4].contents[0]):float(re.sub('\s+','',spans[5].contents[0])),
                                                                    re.sub('\s+','',spans[6].contents[0]):float(re.sub('\s+','',spans[7].contents[0])),
                                                                    re.sub('\s+','',spans[8].contents[0]):float(re.sub('\s+','',spans[9].contents[0])),
                                                                    re.sub('\s+','',spans[10].contents[0]):float(re.sub('\s+','',spans[11].contents[0])),
                                                                    re.sub('\s+','',spans[12].contents[0]):float(re.sub('\s+','',spans[13].contents[0]))
                                                                       
                                                                   }
        except IndexError:
            value=""
    return pivot_dictionary


# In[12]:


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


# In[13]:


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


# In[14]:


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


# In[15]:


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


# In[16]:


def get_data(currency_pair, per, candle_number):
    data = con.get_candles(currency_pair, period=per, number=candle_number)
    ohlc = data.iloc[:,[0,1,2,3,8]]
    ohlc.columns = ["Open","Adj Close","High","Low","Volume"]
    ohlc['weekday'] = ohlc.index.dayofweek
    return ohlc


# In[17]:


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


# In[18]:


day_of_the_week = -1
data_dictionary = {}


def main():
    open_position = con.get_open_positions()
    for currency_pair in pairs:
        alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD"]
        check_closed_positions(currency_pair)
        tradeId = ""
        ohlc = get_data(currency_pair, 'm5', 500)
        equity = con.get_accounts_summary()['equity'][0]
        pos_size = 200
        if currency_pair not in alternative_list:
            pos_size = (risk * equity) / 2

        day_of_the_week = int(ohlc.iloc[-1].weekday)
        close = ohlc.iloc[-1]['Adj Close']
        _open = ohlc.iloc[-1]['Open']
        high = ohlc.iloc[-1]['High']
        low = ohlc.iloc[-1]['Low']
        vol = ohlc.iloc[-1]['Volume']
        day = ohlc.iloc[-1].name.to_pydatetime().strftime("%A")
        date = ohlc.iloc[-1].name.to_pydatetime().strftime("%d %b %Y")
        time = ohlc.iloc[-1].name.to_pydatetime().strftime("%I:%M%p")
        hour = ohlc.iloc[-1].name.hour
        minute = ohlc.iloc[-1].name.minute
        ohlc_df = stochastic(ohlc, 14, 3, 3)
        ohlc_df = SMA(ohlc_df, 100, 200)
        pivot_name = ""
        if currency_pair not in alternative_list:
            pivot_name = get_currency_name_for_pivot(currency_pair)
        else:
            pivot_name = alternative_dictionary[currency_pair]

        signal = pivot_with_sentiment_signal(currency_pair, ohlc_df, close, pivot_name)

        if datetime.now().hour > 20:
            close_all(open_position, currency_pair)

        try:
            if signal != {}:
                if (signal["Signal"] == "Buy"):
                    if (len(open_position) > 0):
                        close_sell(open_position, signal, currency_pair, close, _open, high, low, vol, date, time, day, hour, minute)
                    else:
                        if datetime.now().hour < 17:
                            con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=False, amount=pos_size,
                                           time_in_force='GTC', stop=signal['Support'], limit=signal['Resistance'],
                                           trailing_step=False, order_type='AtMarket')

                            print("New long position initiated for ", currency_pair)

                            message = "New long position initiated for " + currency_pair + "\rEntry: " + str(
                                close) + " \rStop Loss: " + str(signal['Support']) + " \rTake Profit: " + str(
                                signal['Resistance'])
                            send_email(message, "FXCMPY Trade Buy")
                            add_parameters_to_data_dictionary_after_signal(currency_pair, signal["Signal"],
                                                                           signal['Support'], signal['Resistance'],
                                                                           open_position, _open, high, low, vol, date, time, day, hour, minute)
                    # currency_s_r[currency_pair]={"support":support, "resistance":resistance}

                elif (signal["Signal"] == "Sell"):
                    if (len(open_position) > 0):
                        close_buy(open_position, signal, currency_pair, close, _open, high, low, vol, date, time, day, hour, minute)
                    else:
                        if datetime.now().hour < 17:
                            con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=False, amount=pos_size,
                                           time_in_force='GTC', stop=signal['Resistance'], limit=signal['Support'],
                                           trailing_step=False, order_type='AtMarket')
                            print("New short position initiated for ", currency_pair)
                            message = "New short position initiated for " + currency_pair + "\rEntry: " + str(
                                close) + " \rStop Loss: " + str(signal['Resistance']) + " \rTake Profit: " + str(
                                signal['Support'])
                            send_email(message, "FXCMPY Trade Sell")
                            add_parameters_to_data_dictionary_after_signal(currency_pair, signal["Signal"],
                                                                           signal['Resistance'], signal['Support'],
                                                                           open_position, _open, high, low, vol, date, time, day, hour, minute)

                    # currency_s_r[currency_pair]={"support":support, "resistance":resistance}
        except TypeError:
            print(currency_pair)


# In[19]:


def add_parameters_to_data_dictionary_after_signal(currency_pair, signal, stop, limit, open_pos, _open, high, low, vol,
                                                   date, time, day, hour, minute):
    data_dictionary[currency_pair]["signal"] = signal
    data_dictionary[currency_pair]["stop"] = stop
    data_dictionary[currency_pair]["limit"] = limit
    data_dictionary[currency_pair]["open"] = _open
    data_dictionary[currency_pair]["high"] = high
    data_dictionary[currency_pair]["low"] = low
    data_dictionary[currency_pair]["volume"] = vol
    data_dictionary[currency_pair]["date"] = date
    data_dictionary[currency_pair]["time"] = time
    data_dictionary[currency_pair]["day"] = day
    data_dictionary[currency_pair]["hour"] = hour
    data_dictionary[currency_pair]["minute"] = minute
    if (len(open_pos[open_pos["currency"] == currency_pair]) == 1):
        index = open_pos[open_pos["currency"] == currency_pair].index[0]
        tradeId = open_pos[open_pos["currency"] == currency_pair]['tradeId'][index]
        data_dictionary[currency_pair]["tradeId"] = tradeId


# In[20]:


def close_all(open_pos, currency):
    if (len(open_pos) > 0):
        if (len(open_pos[open_pos["currency"] == currency]) == 1):
            con.close_all_for_symbol(currency)


def close_if_support_resistances_exceeded(open_pos, _close, currency):
    if (len(open_pos) > 0):
        if (len(open_pos[open_pos["currency"] == currency]) == 1):
            if currency_s_r[currency] != {}:
                if _close <= currency_s_r[currency]['support'] or _close >= currency_s_r[currency]['resistance']:
                    con.close_all_for_symbol(currency)
                currency_s_r[currency] = {}


def close_buy(open_pos, sig, currency, _close, _open, high, low, vol, date, time, day, hour, minute):
    if (len(open_pos) > 0):
        if (len(open_pos[open_pos["currency"] == currency]) == 1):
            index = open_pos[open_pos["currency"] == currency].index[0]
            if (open_pos[open_pos["currency"] == currency]['isBuy'][index] == True):
                con.close_all_for_symbol(currency)
                if (sig['Signal'] == "Sell"):
                    if datetime.now().hour < 17:
                        con.open_trade(symbol=currency, is_buy=False, is_in_pips=False, amount=pos_size,
                                       time_in_force='GTC', stop=sig['Resistance'], limit=sig['Support'],
                                       trailing_step=False, order_type='AtMarket')
                        print("New short position initiated for ", currency)
                        message = "New short position initiated for " + currency + "\rEntry: " + str(
                            _close) + " \rStop Loss: " + str(sig['Resistance']) + " \rTake Profit: " + str(
                            sig['Support'])
                        send_email(message, "FXCMPY Trade Close Buy and Sell")
                        add_parameters_to_data_dictionary_after_signal(currency, sig["Signal"],
                                                                       sig['Support'], sig['Resistance'],
                                                                       open_pos, _open, high, low, vol, date, time, day, hour, minute)


def close_sell(open_pos, sig, currency, _close, _open, high, low, vol, date, time, day, hour, minute):
    if (len(open_pos) > 0):
        if (len(open_pos[open_pos["currency"] == currency]) == 1):
            index = open_pos[open_pos["currency"] == currency].index[0]
            if (open_pos[open_pos["currency"] == currency]['isBuy'][index] == False):
                con.close_all_for_symbol(currency)
                if (sig['Signal'] == "Buy"):
                    if datetime.now().hour < 17:
                        con.open_trade(symbol=currency, is_buy=True, is_in_pips=False, amount=pos_size,
                                       time_in_force='GTC', stop=sig['Support'], limit=sig['Resistance'],
                                       trailing_step=False, order_type='AtMarket')
                        print("New long position initiated for ", currency)
                        message = "New long position initiated for " + currency + "\rEntry: " + str(
                            _close) + " \rStop Loss: " + str(sig['Support']) + " \rTake Profit: " + str(
                            sig['Resistance'])
                        send_email(message, "FXCMPY Trade Close Sell and Buy")
                        add_parameters_to_data_dictionary_after_signal(currency, sig["Signal"],
                                                                       sig['Resistance'], sig['Support'],
                                                                       open_pos, _open, high, low, vol, date, time, day, hour, minute)


# In[58]:


def check_closed_positions(currency):
    closed_positions = con.get_closed_positions()
    if(len(closed_positions)>0):
        if (len(closed_positions[closed_positions["currency"] == currency]) == 1):
            if currency in data_dictionary:
                index = closed_positions[closed_positions["currency"] == currency].index[0]
                tradeId = closed_positions[closed_positions["currency"] == currency]['tradeId'][index]
                if 'tradeId' in data_dictionary[currency]:
                    if tradeId == data_dictionary[currency]['tradeId']:
                        if con.get_closed_positions()["grossPL"][0] > 0:
                            data_dictionary[currency]['win_loss'] = 1
                        else:
                            data_dictionary[currency]['win_loss'] = 0
                            
                        write_to_csv(
                            currency, data_dictionary[currency]['time'], data_dictionary[currency]['open'], data_dictionary[currency]['high'],
                            data_dictionary[currency]['low'], data_dictionary[currency]['close'],
                            data_dictionary[currency]['sentiment'],
                            data_dictionary[currency]['net_long'], data_dictionary[currency]['net_short'],
                            data_dictionary[currency]['s3'], data_dictionary[currency]['s2'], data_dictionary[currency]['s1'],
                            data_dictionary[currency]['pivot'], data_dictionary[currency]['r1'],
                            data_dictionary[currency]['r2'],
                            data_dictionary[currency]['r3'], data_dictionary[currency]['stop'],
                            data_dictionary[currency]['limit'],
                            data_dictionary[currency]['signal'], data_dictionary[currency]['win_loss'],
                            data_dictionary[currency]['volume'], data_dictionary[currency]['tradeId'],
                            data_dictionary[currency]['date'], data_dictionary[currency]['day'],
                            data_dictionary[currency]['hour'], data_dictionary[currency]['minute']
                        )
                        del data_dictionary[currency]


# In[22]:


def get_currency_name_for_pivot(curr):
    currency_name = curr.replace("/", "-").lower()
    return currency_name


# In[23]:


def pivot_with_sentiment_signal(curr, df, _close, pivot_nm):
    alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD"]
    pivot_table = get_daily_pivot_points(pivot_nm)
    # print(pivot_table)
    if len(pivot_table) == 0:
        return ""
    pivot = pivot_table['P']
    s1 = pivot_table['S1']
    r1 = pivot_table['R1']
    s2 = pivot_table['S2']
    r2 = pivot_table['R2']
    s3 = pivot_table['S3']
    r3 = pivot_table['R3']

    sentiment_table = get_net_buy_sell()
    sentiment_table_name = curr
    if curr in alternative_list:
        sentiment_table_name = full_sentiment_dictionary[curr]

    # obv = OBV(df)
    # obv['slope'] = slope(obv["obv"],5)
    # angle = obv['slope'].iloc[-1]

    low = df['Low'].iloc[-2]
    high = df['High'].iloc[-2]

#     if (p2f(sentiment_table[sentiment_table_name]['Net Long']) > 60 and sentiment_table[sentiment_table_name][
#         'Sentiment'] == "MIXED") or sentiment_table[sentiment_table_name]['Sentiment'] == "BULLISH":
    if (low < pivot and _close > pivot):
        add_signal_params(sentiment_table, sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3, curr, _close)
        return {"Signal": "Buy", "Support": (s1 + pivot) / 2, "Resistance": (r1 + pivot) / 2}
    if (low < s1 and _close > s1):
        add_signal_params(sentiment_table, sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3, curr, _close)
        return {"Signal": "Buy", "Support": (s1 + s2) / 2, "Resistance": (pivot + s1) / 2}
    if (low < s2 and _close > s2):
        add_signal_params(sentiment_table, sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3, curr, _close)
        return {"Signal": "Buy", "Support": (s2 + s3) / 2, "Resistance": (s2 + s1) / 2}

#     elif (p2f(sentiment_table[sentiment_table_name]['Net Long']) < 60 and sentiment_table[sentiment_table_name][
#         'Sentiment'] == "MIXED") or sentiment_table[sentiment_table_name]['Sentiment'] == "BEARISH":
    if (high > pivot and _close < pivot):
        add_signal_params(sentiment_table, sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3, curr, _close)
        return {"Signal": "Sell", "Support": (s1 + pivot) / 2, "Resistance": (pivot + r1) / 2}
    if (high > r1 and _close < r1):
        add_signal_params(sentiment_table, sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3, curr, _close)
        return {"Signal": "Sell", "Support": (pivot + r1) / 2, "Resistance": (r1 + r2) / 2}
    if (high > r2 and _close < r2):
        add_signal_params(sentiment_table, sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3, curr, _close)
        return {"Signal": "Sell", "Support": (r1 + r2) / 2, "Resistance": (r2 + r3) / 2}

    return {}


# In[24]:


def add_signal_params(sentiment_table, sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3, curr, _close):
    data_dictionary[curr] = {"s1": s1, "s2": s2, "s3": s3, "pivot": pivot, "r1": r1,
                             "r2": r2, "r3": r3, "sentiment": sentiment_table[sentiment_table_name]['Sentiment'],
                             "net_long": p2f(sentiment_table[sentiment_table_name]['Net Long']),
                             "net_short": p2f(sentiment_table[sentiment_table_name]['Net Short']), "close": _close
                             }


# In[69]:


import csv

field_names = ['Currency', 'Day','Hour', 'Minute', 'Date', 'Time', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment',
               'Net Long', 'Net Short', 'S3', 'S2',
               'S1', 'P', 'R1', 'R2', 'R3', 'Stop_loss', 'Take_Profit', 'Buy/Sell', 'Win/Loss']

# with open('output.csv', 'w', newline='') as f:
#     thewriter = csv.DictWriter(f, fieldnames=field_names)
#     thewriter.writeheader()


def write_to_csv(currency, time, _open, high, low, close, sentiment, net_long,
                 net_short, s3, s2, s1, p, r1, r2, r3, stop, limit, order_type,
                 win_loss, volume, tradeid, date, day, hour, minute):
    with open('output.csv', 'a', newline='') as f:
       

        thewriter = csv.DictWriter(f, fieldnames=field_names)
        thewriter.writerow({"Currency": currency,
                            "Day" : day,
                            "Hour" : hour, 
                            "Minute": minute,
                            "Date": date,
                            "Time": time,
                            "Open": _open,
                            "High": high,
                            "Low": low,
                            "Close": close,
                            "Volume": volume,
                            "Sentiment": sentiment,
                            "Net Long": net_long,
                            "Net Short": net_short,
                            "S3": s3,
                            "S2": s2,
                            "S1": s1,
                            "P": p,
                            "R3": r3,
                            "R2": r2,
                            "R1": r1,
                            "Stop_loss": stop,
                            "Take_Profit": limit,
                            "Buy/Sell": order_type,
                            "Win/Loss": win_loss,
                            "TradeId": tradeid

                            })


# In[63]:


count = 0
count2 = 0
starttime = time.time()
timeout = time.time() + 60 * 60 * 100000  # 60 seconds times 60 meaning the script will run for 1 hr
# while time.time() <= timeout:
while True:
    robot_message = "passthrough at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\r"
    print(robot_message)
    main()
#     try:
#         if datetime.today().weekday() != 5 and datetime.today().weekday() != 6 and datetime.now().hour < 22:
            
#     except:
#         count2 += 1
#         if count2 == 6:
#             TOKEN = "f2eb943fd86c784d55dcd79906d7e165d2f7fdb3"
#             con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
#             send_email("Error connecting to FXCMPY", "Error Logs")
#             count2 = 0
#         print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
#         send_email("Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(sys.exc_info()[2]),
#                    "Exceptional Error")

    count += 1
    if count == 6:
        if day_of_the_week != 5 and day_of_the_week != 6:
            send_email(str(robot_message), "FXCMPY Logs")
        count = 0
    time.sleep((60 * 5) - ((time.time() - starttime) % (60 * 5)))  # 5 minute interval between each new execution


# In[64]:


ohlc = get_data("EUR/USD", "m5", 250)


# In[68]:


ohlc.iloc[-1].name.minute


# In[ ]:



