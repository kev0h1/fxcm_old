
# coding: utf-8

# In[ ]:

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


# In[ ]:

import smtplib
import re
import sys


# In[ ]:

#from sklearn.linear_model import LinearRegression
#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree


# In[ ]:

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
        send_email(
            "Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(sys.exc_info()[2]),
            "Exceptional Error")


# In[ ]:

TOKEN = "338faa8cdbb0066a9236d101a3a0f5a5020bafa5"

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')


# In[ ]:

pairs = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'NZD/USD', 'AUD/JPY', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY',
         'GBP/JPY', 'USD/CHF', 'BTC/USD', "ETH/USD", "LTC/USD"]


# In[ ]:

risk_percentage = 0.5
risk = risk_percentage * 0.01


# In[ ]:

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


# In[ ]:

def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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


def get_data(currency_pair, per, candle_number):
    data = con.get_candles(currency_pair, period=per, number=candle_number)
    ohlc = data.iloc[:, [0, 1, 2, 3, 8]]
    ohlc.columns = ["Open", "Adj Close", "High", "Low", "Volume"]
    ohlc['weekday'] = ohlc.index.dayofweek
    return ohlc


# In[ ]:

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


# In[ ]:

day_of_the_week = -1
data_dictionary = {}


def main():
    open_position = con.get_open_positions()
    for currency_pair in pairs:
        alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD"]
        ohlc = get_data(currency_pair, 'm5', 500)
        equity = con.get_accounts_summary()['equity'][0]
        pos_size = 1

        if currency_pair in alternative_list:
            pos_size = 5

        close = ohlc.iloc[-1]['Adj Close']
        _open = ohlc.iloc[-1]['Open']
        vol = ohlc.iloc[-1]['Volume']
        day = ohlc.iloc[-1].name.to_pydatetime().strftime("%A")
        date = ohlc.iloc[-1].name.to_pydatetime().strftime("%d %b %Y")
        _time = ohlc.iloc[-1].name.to_pydatetime().strftime("%I:%M%p")
        hour = ohlc.iloc[-1].name.hour
        minute = ohlc.iloc[-1].name.minute
        ohlc_df = stochastic(ohlc, 14, 3, 3)
        ohlc_df = SMA(ohlc_df, 100, 200)
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

        sentiment_table = get_net_buy_sell()
        sentiment_table_name = currency_pair
        if currency_pair in alternative_list:
            sentiment_table_name = full_sentiment_dictionary[currency_pair]

        low = ohlc_df['Low'].iloc[-2]
        high = ohlc_df['High'].iloc[-2]

        signal = pivot_with_sentiment_signal(high, low, pivot, close, s1, s2, s3, r1, r2, r3)

        try:
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
                                                                         volume=vol, date=date,day=day, hour=hour,
                                                                         minute=minute
                                                                         )

                        combined_data_frame = add_to_imported_data_frame(test_row_df)

                        win_loss_value = regression_model.predict(combined_data_frame)[-1]

                        if win_loss_value == 1:
                            con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=False, amount=pos_size,
                                           time_in_force='GTC', stop=signal['Support'], limit=signal['Resistance'],
                                           trailing_step=False, order_type='AtMarket')

                            print("New long position initiated for ", currency_pair)

                            message = "New long position initiated for " + currency_pair + "\rEntry: " + str(
                                close) + " \rStop Loss: " + str(signal['Support']) + " \rTake Profit: " + str(
                                signal['Resistance'])
                            send_email(message, "FXCMPY Trade Buy")


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
                                                                         volume=vol, date=date,day=day, hour=hour,
                                                                         minute=minute
                                                                         )

                        combined_data_frame = add_to_imported_data_frame(test_row_df)

                        win_loss_value = regression_model.predict(combined_data_frame)[-1]

                        if win_loss_value == 1:
                            con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=False, amount=pos_size,
                                           time_in_force='GTC', stop=signal['Resistance'], limit=signal['Support'],
                                           trailing_step=False, order_type='AtMarket')
                            print("New short position initiated for ", currency_pair)
                            message = "New short position initiated for " + currency_pair + "\rEntry: " + str(
                                close) + " \rStop Loss: " + str(signal['Resistance']) + " \rTake Profit: " + str(
                                signal['Support'])
                            send_email(message, "FXCMPY Trade Sell")

        except ConnectionError:
            print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
            send_email(
                "Error on Currency " + currency_pair + " Expected error" + str(sys.exc_info()[0]) + "value:" + str(
                    sys.exc_info()[1]) + str(sys.exc_info()[2]),
                "Exceptional Error")


# In[ ]:

def create_prediction_param_dataframe(currency, _open, high, low, close, sentiment, net_long,
                                      net_short, s3, s2, s1, p, r1, r2, r3, stop, limit, order_type,
                                      win_loss, volume, date, day, hour, minute):
    data = {"Date": date,
            "Currency": currency,
            "Day": day,
            "Hour": hour,
            "Minute": minute,
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
            }


# In[ ]:

currency_s_r = {}
for i in pairs:
    currency_s_r[i] = {}


# In[ ]:

def close_all(open_pos, currency):
    if len(open_pos) > 0:
        if len(open_pos[open_pos["currency"] == currency]) == 1:
            con.close_all_for_symbol(currency)


# In[ ]:

def close_if_support_resistances_exceeded(open_pos, _close, currency):
    if (len(open_pos) > 0):
        if (len(open_pos[open_pos["currency"] == currency]) == 1):
            if currency_s_r[currency] != {}:
                if _close <= currency_s_r[currency]['support'] or _close >= currency_s_r[currency]['resistance']:
                    con.close_all_for_symbol(currency)
                currency_s_r[currency] = {}


def close_trade(open_pos, sig, currency, _close, _open, high, low, vol, date, _time, day, hour, minute, sentiment_table,
                sentiment_table_name, pivot, s1, s2, s3, r1, r2, r3,
                currency_pair, close, pos_size):
    if len(open_pos) > 0 and len(open_pos[open_pos["currency"] == currency]) == 1:
        index = open_pos[open_pos["currency"] == currency].index[0]
        if sig['Signal'] == "Sell" and open_pos[open_pos["currency"] == currency]['isBuy'][index] == True:
            con.close_all_for_symbol(currency)
            check_closed_positions(currency)
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
            check_closed_positions(currency)
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


# In[ ]:

def get_currency_name_for_pivot(curr):
    currency_name = curr.replace("/", "-").lower()
    return currency_name


# In[ ]:

def pivot_with_sentiment_signal(high, low, pivot, _close, s1, s2, s3, r1, r2, r3):
    if low < pivot < _close and ((s1 + pivot) / 2) < _close < ((r1 + pivot)):
        return {"Signal": "Buy", "Support": (s1 + pivot) / 2, "Resistance": (r1 + pivot) / 2}
    if low < s1 < _close and ((s1 + s2) / 2) < _close < ((pivot + s1) / 2):
        return {"Signal": "Buy", "Support": (s1 + s2) / 2, "Resistance": (pivot + s1) / 2}
    if low < s2 < _close and ((s2 + s3) / 2) < _close < ((s2 + s1) / 2):
        return {"Signal": "Buy", "Support": (s2 + s3) / 2, "Resistance": (s2 + s1) / 2}
    if high > pivot > _close and ((s1 + pivot) / 2) < _close < ((pivot + r1) / 2):
        return {"Signal": "Sell", "Support": (s1 + pivot) / 2, "Resistance": (pivot + r1) / 2}
    if high > r1 > _close and ((pivot + r1) / 2) < _close < ((r1 + r2) / 2):
        return {"Signal": "Sell", "Support": (pivot + r1) / 2, "Resistance": (r1 + r2) / 2}
    if high > r2 > _close and ((r1 + r2) / 2) < _close < ((r2 + r3) / 2):
        return {"Signal": "Sell", "Support": (r1 + r2) / 2, "Resistance": (r2 + r3) / 2}

    return {}


# In[ ]:

trained =False
count = 0
count2 = 0
starttime = time.time()
timeout = time.time() + 60 * 60 * 100000  # 60 seconds times 60 meaning the script will run for 1 hr
# while time.time() <= timeout:

def machine_learning():
    imported_dataframe = pd.read_csv('output.csv', index_col='Date')
    imported_dataframe.drop('TradeId', axis=1, inplace=True)
    buy_sell_dummies = get_dummies(imported_dataframe)
    X = buy_sell_dummies.drop('Win/Loss', axis=1)
    Y = buy_sell_dummies['Win/Loss']
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.99, random_state=0)
    regression_model = tree.DecisionTreeClassifier()
    regression_model.fit(X_train, Y_train)
    return regression_model


# In[ ]:

regression_model = tree.DecisionTreeClassifier()


# In[ ]:

def create_prediction_param_data_frame(currency, _open, high, low, close, sentiment, net_long,
                                       net_short, s3, s2, s1, p, r1, r2, r3, stop, limit, order_type,
                                       win_loss, volume, date, day, hour, minute):
    data = {0: [date, currency, day, hour, minute, _open, high, low, close, volume,
                sentiment, net_long, net_short, s3, s2, s1, p, r3, r2, r1, stop, limit,
                order_type, win_loss]}

    df = pd.DataFrame.from_dict(data, orient='index', columns=['Date', "Currency",
                                                               "Day", "Hour", "Minute", "Open",
                                                               "High", "Low", "Close", "Volume",
                                                               "Sentiment", "Net Long", "Net Short",
                                                               "S3", "S2", "S1", "P", "R3", "R2", "R1",
                                                               "Stop_loss", "Take_Profit", "Buy/Sell", "Win/Loss",
                                                               ])
    df = df.set_index('Date')
    return df


# In[ ]:

def add_to_imported_data_frame(predicting_param_data_frame):
    read_df = pd.read_csv('output.csv', index_col='Date')
    read_df.drop('TradeId', axis=1, inplace=True)
    joined_df = pd.concat([read_df, predicting_param_data_frame])
    joined_df = get_dummies(joined_df)
    X = joined_df.drop('Win/Loss', axis=1)
    return X


# In[ ]:

def get_dummies(df):
    return pd.get_dummies(df, columns=['Currency', 'Day', 'Sentiment', 'Buy/Sell'])


# In[ ]:

while True:
    robot_message = "pass through at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\r"
    print(robot_message)
    
    try:
        if datetime.today().weekday() != 5 and datetime.today().weekday() != 6 and datetime.now().hour < 22:
            if trained == False:
                regression_model = machine_learning()
                trained == True
            if datetime.now().hour > 20:
                if len(con.get_open_positions()) > 0:
                    if con.is_connected() == True:
                        con.close_all()
                    else:
                        con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                        con.close_all()

                    if len(con.get_open_positions()) > 0:
                        con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                        con.close_all()

                if con.is_connected() == True:
                    con.close()

            if datetime.now().hour < 21:
                if con.is_connected() == False:
                    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                main()

            else:
                if datetime.now().hour > 22:
                    trained =False

    except:
        print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
        send_email(
            "Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(sys.exc_info()[2]),
            "Exceptional Error")

    count += 1
    if count == 6:
        if day_of_the_week != 5 and day_of_the_week != 6:
            send_email(str(robot_message), "FXCMPY Logs")
        count = 0
    time.sleep((60 * 5) - ((time.time() - starttime) % (60 * 5)))  # 5 minute interval between each new execution


# In[ ]:



