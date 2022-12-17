
# coding: utf-8

# In[1]:

import requests
from bs4 import BeautifulSoup
import pandas as pd
import fxcmpy
import time
import numpy as np
from stocktrends import Renko
# import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# In[2]:

import smtplib
import re
import sys


# In[3]:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree


# In[4]:

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


# In[5]:

TOKEN = "54283cd549ee881467a84ecaf5282bd4a6360ef8"

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='real')

output_file_name = 'output3.csv'

log_file = 'real_log.txt'


# In[6]:

# pairs = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'NZD/USD', 'AUD/JPY', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY',
#          'GBP/JPY', 'USD/CHF', 'BTC/USD', "ETH/USD", "LTC/USD",'GER30', 'FRA40' ]

pairs = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'NZD/USD', 'AUD/JPY', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY',
         'GBP/JPY', 'USD/CHF', 'BTC/USD','GER30', 'FRA40' ]


# In[7]:

alternative_dictionary = {'XAU/USD': 'gold-price', 'XAG/USD': 'silver-prices', 'BTC/USD': 'bitcoin', 'GER30': 'dax-30',
                          'FRA40': 'cac-40', 'ETH/USD': 'ether-eth', 'LTC/USD': 'litecoin-ltc'}
alternative_sentiment_dictionary = {'XAU/USD': 'Gold', 'XAG/USD': 'Silver', 'BTC/USD': 'Bitcoin', 'GER30': 'Germany 30',
                                    'FRA40': 'France 40'}
full_sentiment_dictionary = {"BTC/USD": "Bitcoin", "GER30": "Germany 30"
    , "ETH/USD": "Ethereum", "FRA40": "France 40", "XAU/USD": "Gold"
    , "LTC/USD": "Litecoin", "XAG/USD": "Silver"
                             }
full_pivot_table_dictionary = {"BTC/USD": "Bitcoin", "ETH/USD": "Ethereum",
                               "LTC/USD": "Litecoin", "XAU/USD": "Gold", "XAG/USD": "Silver",
                               "GER30": "Germany30", "FRA40": "France40"}

k_scaler = 1000

leverage = {'EUR/USD':30, 'USD/JPY':30, 'AUD/USD':30, 'GBP/USD':30, 'USD/CAD':30, 
                   'NZD/USD':30, 'AUD/JPY':20, 'EUR/CHF':20, 'EUR/GBP':20, 'EUR/JPY':20,
         'GBP/JPY':20, 'USD/CHF':30, 'BTC/USD':2, "ETH/USD":2, "LTC/USD":2, 'GER30':20, 'FRA40':20}

lot_size_scaler = {'EUR/USD':10000, 'USD/JPY':100, 'AUD/USD':10000, 'GBP/USD':10000, 'USD/CAD':10000, 
                   'NZD/USD':10000, 'AUD/JPY':100, 'EUR/CHF':10000, 'EUR/GBP':10000, 'EUR/JPY':100,
         'GBP/JPY':100, 'USD/CHF':10000, 'BTC/USD':1, "ETH/USD":1, "LTC/USD":100, 'GER30':1, 'FRA40':1}


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
    currency_dictionary_full = {}
    url = "https://www.dailyfx.com/sentiment-report"
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    currency_information = soup.find_all("tr")
    for i in range(1, len(currency_information)):
        currency_dictionary_full[currency_information[i].find_all("span")[0].contents[0]] = {
            "Sentiment": currency_information[i].find_all("span")[1].contents[0],
            "Net Long": currency_information[i].find_all("span")[2].contents[0],
            "Net Short": currency_information[i].find_all("span")[3].contents[0]
        }
    return currency_dictionary_full


def p2f(x):
    return float(x.strip('%'))


# In[10]:

strengths = [
    "dfx-supportResistanceLevelStrength dfx-supportResistanceLevelStrength--strong dfx-supportResistanceBlock__valueLevelStrength mx-1 mx-md-0",
    "dfx-supportResistanceLevelStrength dfx-supportResistanceLevelStrength--moderate dfx-supportResistanceBlock__valueLevelStrength mx-1 mx-md-0",
    "dfx-supportResistanceLevelStrength dfx-supportResistanceLevelStrength--weak dfx-supportResistanceBlock__valueLevelStrength mx-1 mx-md-0"]

support_resistance_values = {0: 'S1', 1: 'S2', 2: 'S3', 3: 'R1', 4: 'R2', 5: 'R3'}


def get_currency_sentiment():
    url = "https://www.dailyfx.com/forex-rates"
    try:
        page = requests.get(url)
        page_content = page.content
        soup = BeautifulSoup(page_content, 'html.parser')
        currency_information = soup.find_all("div", {
            "class": "dfx-singleInstrument__nameAndSignal d-flex flex-column h-100 mr-1 justify-content-around align-self-start text-dark"})
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
    support_resistance = {}
    previous_SP = ""
    lists = []
    if (len(currency_support_resistances) > 0):
        for info in currency_support_resistances:
            count = 0
            for item in info.find_all('span'):
                if item.string.startswith("S") or item.string.startswith("R"):
                    previous_SP = item.string
                else:
                    support_resistance[previous_SP] = [float(item.string), "", count]
                    count += 1

        for index, item in enumerate(currency_support_resistances[0].find_all("div", {"class": strengths})):
            support_resistance[support_resistance_values[index]][1] = get_strength(str(item))

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
    pivot_values = {}
    for item in pivots:
        values = item.find_all("span")
        pivot_values[values[0].string.strip()] = float(values[1].string.strip())
    return pivot_values


# In[11]:

def full_pivot_table():
    url = "https://www.dailyfx.com/pivot-points"
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    pivots = soup.find_all("div", {
        "class": "dfx-pivotPointsTable dfx-pivotPointsWidget__table container pt-2 pt-md-0 dfx-font-size-3"})
    pivot_dictionary = {}
    for i in range(0, len(pivots)):
        try:
            spans = pivots[i].find_all("span")
            pivot_dictionary[re.sub('\s+', '', pivots[i].a.contents[0])] = {
                re.sub('\s+', '', spans[0].contents[0]): float(re.sub('\s+', '', spans[1].contents[0])),
                re.sub('\s+', '', spans[2].contents[0]): float(re.sub('\s+', '', spans[3].contents[0])),
                re.sub('\s+', '', spans[4].contents[0]): float(re.sub('\s+', '', spans[5].contents[0])),
                re.sub('\s+', '', spans[6].contents[0]): float(re.sub('\s+', '', spans[7].contents[0])),
                re.sub('\s+', '', spans[8].contents[0]): float(re.sub('\s+', '', spans[9].contents[0])),
                re.sub('\s+', '', spans[10].contents[0]): float(re.sub('\s+', '', spans[11].contents[0])),
                re.sub('\s+', '', spans[12].contents[0]): float(re.sub('\s+', '', spans[13].contents[0]))

            }
        except IndexError:
            value = ""
    return pivot_dictionary


# In[12]:

def supports_less_than_close(support_resistances, close):
    supports_to_consider = []
    if (support_resistances['S1'][0] < close):
        supports_to_consider.append('S1')
    if (support_resistances['S2'][0] < close):
        supports_to_consider.append('S2')
    if (support_resistances['S3'][0] < close):
        supports_to_consider.append('S3')
    return supports_to_consider


def resistances_greater_than_close(support_resistances, close):
    resistances_to_consider = []
    if (support_resistances['R1'][0] > close):
        resistances_to_consider.append('R1')
    if (support_resistances['R2'][0] > close):
        resistances_to_consider.append('R2')
    if (support_resistances['R3'][0] > close):
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


def get_data(currency_pair, per, candle_number):
    data = con.get_candles(currency_pair, period=per, number=candle_number)
    ohlc = data.iloc[:, [0, 1, 2, 3, 8]]
    ohlc.columns = ["Open", "Adj Close", "High", "Low", "Volume"]
    ohlc['weekday'] = ohlc.index.dayofweek
    return ohlc


# In[14]:

def stochastic(df, a, b, c):
    "function to calculate stochastic"
    df['k'] = ((df['Adj Close'] - df['Low'].rolling(a).min()) / (
            df['High'].rolling(a).max() - df['Low'].rolling(a).min())) * 100
    df['K'] = df['k'].rolling(b).mean()
    df['D'] = df['K'].rolling(c).mean()
    return df


def SMA(df, a, b):
    "function to calculate stochastic"
    df['sma_fast'] = df['Adj Close'].rolling(a).mean()
    df['sma_slow'] = df['Adj Close'].rolling(b).mean()
    return df


def OBV(DF):
    """function to calculate On Balance Volume"""
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']

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

def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.api.add_constant(x_scaled)
        model = sm.api.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","close","high","low","volume"]
    df2 = Renko(df)
    df2.brick_size = round(ATR(DF,120)["ATR"][-1],4)
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df["bar_num"])):
        if renko_df["bar_num"][i]>0 and renko_df["bar_num"][i-1]>0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
        elif renko_df["bar_num"][i]<0 and renko_df["bar_num"][i-1]<0:
            renko_df["bar_num"][i]+=renko_df["bar_num"][i-1]
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    return renko_df

def renko_merge(DF):
    "function to merging renko df with original ohlc df"
    df = copy.deepcopy(DF)
    df = DF.copy()
    df["Date"] = df.index
    renko = renko_DF(df)
    renko.columns = ["Date","open","high","low","close","uptrend","bar_num"]
    merged_df = df.merge(renko.loc[:,["Date","bar_num"]],how="outer",on="Date")
    merged_df["bar_num"].fillna(method='ffill',inplace=True)
    merged_df["macd"]= MACD(merged_df,12,26,9)[0]
    merged_df["macd_sig"]= MACD(merged_df,12,26,9)[1]
    merged_df["macd_slope"] = slope(merged_df["macd"],5)
    merged_df["macd_sig_slope"] = slope(merged_df["macd_sig"],5)
    return merged_df

def ADX(DF,n):
    "function to calculate ADX"
    df2 = DF.copy()
    df2['TR'] = ATR(df2,n)['TR'] #the period parameter of ATR function does not matter because period does not influence TR calculation
    df2['DMplus']=np.where((df2['High']-df2['High'].shift(1))>(df2['Low'].shift(1)-df2['Low']),df2['High']-df2['High'].shift(1),0)
    df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
    df2['DMminus']=np.where((df2['Low'].shift(1)-df2['Low'])>(df2['High']-df2['High'].shift(1)),df2['Low'].shift(1)-df2['Low'],0)
    df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for i in range(len(df2)):
        if i < n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == n:
            TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
    df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
    df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
    df2['DIsum']=df2['DIplusN']+df2['DIminusN']
    df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2*n-1:
            ADX.append(np.NaN)
        elif j == 2*n-1:
            ADX.append(df2['DX'][j-n+1:j+1].mean())
        elif j > 2*n-1:
            ADX.append(((n-1)*ADX[j-1] + DX[j])/n)
    df2['ADX']=np.array(ADX)
    return df2['ADX']


# In[15]:

day_of_the_week = -1
data_dictionary = {}

risk = 5
risk_percentage = risk/100

def main():
    sentiment_table = get_net_buy_sell()
    for currency_pair in pairs:
        try:
            if con.is_connected() == False:
                return
            alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD"]

            ohlc = get_data(currency_pair, 'm15', 250)
            equity = con.get_accounts_summary()['equity'][0]
            pos_size = 1

            close = ohlc.iloc[-1]['Adj Close']
            _open = ohlc.iloc[-1]['Open']
            vol = ohlc.iloc[-1]['Volume']
            day = ohlc.iloc[-1].name.to_pydatetime().strftime("%A")
            date = ohlc.iloc[-1].name.to_pydatetime().strftime("%d %b %Y")
            _time = ohlc.iloc[-1].name.to_pydatetime().strftime("%I:%M%p")
            hour = ohlc.iloc[-1].name.hour
            minute = ohlc.iloc[-1].name.minute
            stoch = stochastic(ohlc, 14, 3, 3).dropna()
            stoch_k = stoch.iloc[-1]['K']
            stoch_d = stoch.iloc[-1]['D']
            ohlc_df = SMA(ohlc, 5, 13).dropna()
            sma_slow = ohlc_df.iloc[-1]['sma_slow']
            sma_fast = ohlc_df.iloc[-1]['sma_fast']
            mac = MACD(ohlc, 12, 26, 9)[-1][-1]
            obv = OBV(ohlc).iloc[-1]
            atr = ATR(ohlc, 10).dropna().iloc[-1]['ATR']
            adx = ADX(ohlc, 14).dropna().iloc[-1]
            renko = renko_DF(ohlc)
            uptrend = renko.iloc[-1]['uptrend']
            renko_bar_num = renko.iloc[-1]['bar_num']
            if currency_pair not in alternative_list:
                pivot_name = get_currency_name_for_pivot(currency_pair)
            else:
                pivot_name = alternative_dictionary[currency_pair]

            pivot_table = get_daily_pivot_points(pivot_name)

            if len(pivot_table) == 0:
                return ""
            pivot = pivot_table['P']
            s1 = pivot_table['S1']
            r1 = pivot_table['R1']
            s2 = pivot_table['S2']
            r2 = pivot_table['R2']
            s3 = pivot_table['S3']
            r3 = pivot_table['R3']

            sentiment_table_name = currency_pair
            if currency_pair in alternative_list:
                sentiment_table_name = full_sentiment_dictionary[currency_pair]

            low = ohlc_df['Low'].iloc[-2]
            high = ohlc_df['High'].iloc[-2]

            signal = pivot_with_sentiment_signal(high, low, pivot, close, s1, s2, s3, r1, r2, r3)

            if signal != {}:
                if signal["Signal"] == "Buy":
                    if datetime.now().hour < 21:
                        test_row_df = create_prediction_param_data_frame(currency=currency_pair, _open=_open,
                                                                         high=high, low=low, close=close, sentiment=
                                                                         sentiment_table[sentiment_table_name][
                                                                             'Sentiment'],
                                                                         net_long=p2f(
                                                                             sentiment_table[sentiment_table_name][
                                                                                 'Net Long']),
                                                                         net_short=p2f(
                                                                             sentiment_table[sentiment_table_name][
                                                                                 'Net Short']),
                                                                         s3=s3, s2=s2, s1=s1, p=pivot, r1=r1, r2=r2,
                                                                         r3=r3, stop=signal['Support'],
                                                                         limit=signal['Resistance'],
                                                                         order_type=signal["Signal"], win_loss=0,
                                                                         volume=vol, date=date, day=day, hour=hour,
                                                                         minute=minute, stoch_k=stoch_k, stoch_d=stoch_d,
                                                                         sma_slow=sma_slow, sma_fast=sma_fast, renko_uptrend=uptrend, renko_bar_num=renko_bar_num,
                                                                         macd=mac, obv=obv, atr=atr, adx=adx)

                        combined_data_frame = add_to_imported_data_frame(test_row_df)

                        win_loss_value = regression_model.predict(combined_data_frame)[-1]
                        
                        pos_size = ((risk_percentage*equity)/leverage[currency_pair])/((close-signal['Support'])*lot_size_scaler[currency_pair]*10)
                        
                        pos_size = round(pos_size * k_scaler)
                        
                        if pos_size <1:
                            pos_size =1
                        
                        if win_loss_value == 1:
                            con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=False, amount=pos_size,
                                           time_in_force='GTC', stop=signal['Support'], limit=signal['Resistance'],
                                           trailing_step=False, order_type='AtMarket')

#                             print("New long position initiated for ", currency_pair)
                            
                            message = "Trade Buy - New long position initiated for " + currency_pair + "\rEntry: " + str(
                                close) + " \rStop Loss: " + str(signal['Support']) + " \rTake Profit: " + str(
                                signal['Resistance'])
                            logger(message)


                elif signal["Signal"] == "Sell":
                    if datetime.now().hour < 21:
                        test_row_df = create_prediction_param_data_frame(currency=currency_pair, _open=_open,
                                                                         high=high, low=low, close=close, sentiment=
                                                                         sentiment_table[sentiment_table_name][
                                                                             'Sentiment'],
                                                                         net_long=p2f(
                                                                             sentiment_table[sentiment_table_name][
                                                                                 'Net Long']),
                                                                         net_short=p2f(
                                                                             sentiment_table[sentiment_table_name][
                                                                                 'Net Short']),
                                                                         s3=s3, s2=s2, s1=s1, p=pivot, r1=r1, r2=r2,
                                                                         r3=r3, stop=signal['Resistance'],
                                                                         limit=signal['Support'],
                                                                         order_type=signal["Signal"], win_loss=0,
                                                                         volume=vol, date=date, day=day, hour=hour,
                                                                         minute=minute, stoch_k=stoch_k, stoch_d=stoch_d,
                                                                         sma_slow=sma_slow, sma_fast=sma_fast, renko_uptrend=uptrend, renko_bar_num=renko_bar_num,
                                                                         macd=mac, obv=obv, atr=atr, adx=adx)


                        combined_data_frame = add_to_imported_data_frame(test_row_df)

                        win_loss_value = regression_model.predict(combined_data_frame)[-1]
                        
                        pos_size = ((risk_percentage*equity)/leverage[currency_pair])/((signal['Resistance']-close)*lot_size_scaler[currency_pair]*10)
                        
                        pos_size = round(pos_size * k_scaler)
                        
                        if pos_size <1:
                            pos_size =1
                            
                        if win_loss_value == 1:
                            con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=False, amount=pos_size,
                                           time_in_force='GTC', stop=signal['Resistance'], limit=signal['Support'],
                                           trailing_step=False, order_type='AtMarket')
                            
                            message = "Trade Sell - New short position initiated for " + currency_pair + "\rEntry: " + str(
                                close) + " \rStop Loss: " +  str(signal['Resistance']) + " \rTake Profit: " + str(signal['Support'])
                            logger(message)


        except:
            message = "Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(sys.exc_info()[2])
            print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
            logger(message)
            if con.is_connected() == True:
                con.close()
            
            


# In[16]:

currency_s_r = {}
for i in pairs:
    currency_s_r[i] = {}


# In[17]:

def close_all(open_pos, currency):
    if len(open_pos) > 0:
        if len(open_pos[open_pos["currency"] == currency]) == 1:
            con.close_all_for_symbol(currency)


# In[18]:

def close_trade(open_pos, sig, currency, _close, _open, high, low, vol, date, _time, day, hour, minute, sentiment_table,
                sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3,
                currency_pair, close, pos_size):
    if len(open_pos) > 0 and len(open_pos[open_pos["currency"] == currency]) == 1:
        index = open_pos[open_pos["currency"] == currency].index[0]
        if sig['Signal'] == "Sell" and open_pos[open_pos["currency"] == currency]['isBuy'][index] == True:
            con.close_all_for_symbol(currency)

            if datetime.now().hour < 21:
                con.open_trade(symbol=currency, is_buy=False, is_in_pips=False, amount=pos_size,
                               time_in_force='GTC', stop=sig['Resistance'], limit=sig['Support'],
                               trailing_step=False, order_type='AtMarket')
                print("New short position initiated for ", currency)
                message = "New short position initiated for " + currency + "\rEntry: " + str(
                    _close) + " \rStop Loss: " + str(sig['Resistance']) + " \rTake Profit: " + str(
                    sig['Support'])
                send_email(message, "FXCMPY Trade Close Buy and Sell")

        if sig['Signal'] == "Buy" and open_pos[open_pos["currency"] == currency]['isBuy'][index] == False:

            con.close_all_for_symbol(currency)
            if datetime.now().hour < 21:
                con.open_trade(symbol=currency, is_buy=True, is_in_pips=False, amount=pos_size,
                               time_in_force='GTC', stop=sig['Support'], limit=sig['Resistance'],
                               trailing_step=False, order_type='AtMarket')
                print("New long position initiated for ", currency)
                message = "New long position initiated for " + currency + "\rEntry: " + str(
                    _close) + " \rStop Loss: " + str(sig['Support']) + " \rTake Profit: " + str(
                    sig['Resistance'])
                send_email(message, "FXCMPY Trade Close Sell and Buy")


# In[19]:

def get_currency_name_for_pivot(curr):
    currency_name = curr.replace("/", "-").lower()
    return currency_name


# In[20]:

def pivot_with_sentiment_signal(high, low, pivot, _close, s1, s2, s3, r1, r2, r3):
    s1_pivot = (s1 + pivot) / 2
    s1_s2 = (s1 + s2) / 2
    s2_s3 = (s2 + s3) / 2

    r1_pivot = (r1 + pivot) / 2
    r1_r2 = (r1 + r2) / 2
    r2_r3 = (r2 + r3) / 2

    if low < pivot < _close and s1_pivot < _close < r1:
        return {"Signal": "Buy", "Support": s1_pivot, "Resistance": r1}
    if low < s1 < _close and s1_s2 < _close < pivot:
        return {"Signal": "Buy", "Support": s1_s2, "Resistance": pivot}
    if low < s2 < _close and s2_s3 < _close < s1:
        return {"Signal": "Buy", "Support": s2_s3, "Resistance": s1}
    if high > pivot > _close and s1 < _close < r1_pivot:
        return {"Signal": "Sell", "Support": s1, "Resistance": r1_pivot}
    if high > r1 > _close and pivot < _close < r1_r2:
        return {"Signal": "Sell", "Support": pivot, "Resistance": r1_r2}
    if high > r2 > _close and r1 < _close < r2_r3:
        return {"Signal": "Sell", "Support": r1, "Resistance": r2_r3}

    return {}


# In[21]:

trained = False

def machine_learning():
    imported_dataframe = pd.read_csv(output_file_name, index_col='Date')
    imported_dataframe.drop('TradeId', axis=1, inplace=True)
    buy_sell_dummies = get_dummies(imported_dataframe)
    X = buy_sell_dummies.drop('Win/Loss', axis=1)
    Y = buy_sell_dummies['Win/Loss']
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.99, random_state=0)
    regression_model = tree.DecisionTreeClassifier(max_depth=6)
    regression_model.fit(X_train, Y_train)
    return regression_model


# In[22]:

regression_model = tree.DecisionTreeClassifier()


# In[23]:

def create_prediction_param_data_frame(currency, _open, high, low, close, sentiment, net_long,
                                       net_short, s3, s2, s1, p, r1, r2, r3, stop, limit, order_type,
                                       win_loss, volume, date, day, hour, minute, stoch_k, stoch_d, sma_slow, sma_fast,
                                       renko_uptrend,
                                       renko_bar_num, macd, obv, atr, adx):
    data = {0: [date, currency, day, hour, minute, _open, high, low, close, volume,
                sentiment, net_long, net_short, s3, s2, s1, p, r3, r2, r1, stop, limit,
                stoch_k, stoch_d, sma_slow, sma_fast, renko_uptrend,
                renko_bar_num, macd, obv, atr, adx, order_type, win_loss]}

    df = pd.DataFrame.from_dict(data, orient='index', columns=['Date', "Currency",
                                                               "Day", "Hour", "Minute", "Open",
                                                               "High", "Low", "Close", "Volume",
                                                               "Sentiment", "Net Long", "Net Short",
                                                               "S3", "S2", "S1", "P", "R3", "R2", "R1",
                                                               "Stop_loss", "Take_Profit",
                                                               "Stochastic_K", "Stochastic_D", "SMA_slow", "SMA_fast",
                                                               "Renko_Uptrend",
                                                               "Renko_bar_num", "MACD", "OBV", "ATR", "ADX", "Buy/Sell",
                                                               "Win/Loss"])
    df = df.set_index('Date')
    return df


# In[24]:

def add_to_imported_data_frame(predicting_param_data_frame):
    read_df = pd.read_csv(output_file_name, index_col='Date')
    read_df.drop('TradeId', axis=1, inplace=True)
    joined_df = pd.concat([read_df, predicting_param_data_frame])
    joined_df = get_dummies(joined_df)
    X = joined_df.drop('Win/Loss', axis=1)
    return X


# In[25]:

def get_dummies(df):
    return pd.get_dummies(df, columns=['Currency', 'Day', 'Sentiment', 'Buy/Sell',
                                       'Renko_Uptrend'])


# In[28]:

def write_logger_header():
    with open(log_file, 'w', newline='') as f:
        f.write("Real Logs")
        
# write_logger_header()

def logger(message):
    with open(log_file, 'a', newline='') as f:
        f.write("\n"+message)


# In[ ]:

count = 0
count2 = 0
starttime = time.time()
first_iteration = True
while True:
    try:
        message = "pass through at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        logger(message)
        print(message)
        if datetime.today().weekday() != 5 and datetime.today().weekday() != 6 and datetime.now().hour < 22:
            if not first_iteration:
                    first_iteration = True
            if datetime.now().hour > 7: 
                if trained == False:
                    regression_model = machine_learning()
                    trained == True
                if datetime.now().hour > 18 and datetime.now().minute >= 45:
                    if len(con.get_open_positions()) > 0:
                        if con.is_connected() == True:
                            con.close_all()

                if datetime.now().hour < 19:
                    if con.is_connected() == False:
                        con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='real')
                    main()
        else:
            trained =False
            if con.is_connected() == True:
                con.close()
            if first_iteration:
                write_logger_header()
                    
                    
                
        time.sleep((60 * 15) - ((time.time() - starttime) % (60 * 15)))           
    except KeyboardInterrupt:
        print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])

            
    


# In[ ]:



