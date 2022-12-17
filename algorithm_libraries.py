import smtplib
import sys
from bs4 import BeautifulSoup
import numpy as np
from stocktrends import Renko
import requests
import re
import pandas as pd
import time
# import statsmodels
import copy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import smtplib
import sys
import csv
from timeloop import Timeloop

pairs = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'NZD/USD', 'AUD/JPY', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY',
         'GBP/JPY', 'USD/CHF', 'BTC/USD', "ETH/USD", "LTC/USD", 'GER30', "FRA40"]

alternative_dictionary = {'XAU/USD': 'gold-price', 'XAG/USD': 'silver-prices', 'BTC/USD': 'bitcoin', 'GER30': 'dax-30',
                          'FRA40': 'cac-40', 'ETH/USD': 'ether-eth', 'LTC/USD': 'litecoin-ltc'}

alternative_sentiment_dictionary = {'XAU/USD': 'Gold', 'XAG/USD': 'Silver', 'BTC/USD': 'Bitcoin', 'GER30': 'Germany 30',
                                    'FRA40': 'France 40'}

full_sentiment_dictionary = {"BTC/USD": "Bitcoin", "GER30": "Germany 30",
                             "ETH/USD": "Ethereum", "FRA40": "France 40", "XAU/USD": "Gold",
                             "LTC/USD": "Litecoin", "XAG/USD": "Silver"
                             }
full_pivot_table_dictionary = {"BTC/USD": "Bitcoin", "ETH/USD": "Ethereum",
                               "LTC/USD": "Litecoin", "XAU/USD": "Gold", "XAG/USD": "Silver",
                               "GER30": "Germany30", "FRA40": "France40"}

field_names = ['Date', 'Currency', 'Day', 'Hour', 'Minute', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Sentiment',
               'Net Long', 'Net Short', 'S3', 'S2',
               'S1', 'P', 'R1', 'R2', 'R3', 'Stop_loss', 'Take_Profit', 'Buy/Sell', 'Win/Loss']

field_names2 = ['Date', 'Currency', 'Day', 'Hour', 'Minute', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Sentiment',
                'Net Long', 'Net Short', 'S3', 'S2',
                'S1', 'P', 'R1', 'R2', 'R3', 'Stop_loss', 'Take_Profit', 'Stochastic_K', 'Stochastic_D', 'SMA_slow',
                'SMA_fast', 'Renko_Uptrend', 'Renko_bar_num', 'MACD', 'OBV', 'ATR', 'ADX', 'Buy/Sell', 'Win/Loss']

field_names3 = ['Date', 'Currency', 'Day', 'Hour', 'Minute', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Sentiment',
                'Net Long', 'Net Short', 'S3', 'S2',
                'S1', 'P', 'R1', 'R2', 'R3', 'Stop_loss', 'Take_Profit', 'Stochastic_K', 'Stochastic_D', 'SMA_slow',
                'SMA_fast', 'Renko_Uptrend', 'Renko_bar_num', 'MACD', 'OBV', 'ATR', 'ADX', 'RSI', 'Buy/Sell', 'Win/Loss']


def get_data(currency_pair, per, candle_number, con):
    data = con.get_candles(currency_pair, period=per, number=candle_number)
    ohlc = data.iloc[:, [0, 1, 2, 3, 8]]
    ohlc.columns = ["Open", "Adj Close", "High", "Low", "Volume"]
    ohlc['weekday'] = ohlc.index.dayofweek
    return ohlc


def write_header(file, field_type):
    with open(file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_type)
        writer.writeheader()


def write_to_csv(file, currency, _open, high, low, close, sentiment, net_long,
                 net_short, s3, s2, s1, p, r1, r2, r3, stop, limit, order_type,
                 win_loss, volume, date, day, hour, minute, trade_id, stoch_k, stoch_d, sma_slow, sma_fast,
                 renko_uptrend,
                 renko_bar_num, macd, obv, atr, adx):
    with open(file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names2)
        writer.writerow({"Date": date,
                         "Currency": currency,
                         "Day": day,
                         "Hour": hour,
                         "Minute": minute,
                         "TradeId": trade_id,
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
                         'Stochastic_K': stoch_k,
                         'Stochastic_D': stoch_d,
                         'SMA_slow': sma_slow,
                         'SMA_fast': sma_fast,
                         'Renko_Uptrend': renko_uptrend,
                         'Renko_bar_num': renko_bar_num,
                         'MACD': macd,
                         'OBV': obv,
                         'ATR': atr,
                         'ADX': adx,
                         "Buy/Sell": order_type,
                         "Win/Loss": win_loss,
                         })

def write_to_csv_rsi(file, currency, _open, high, low, close, sentiment, net_long,
                 net_short, s3, s2, s1, p, r1, r2, r3, stop, limit, order_type,
                 win_loss, volume, date, day, hour, minute, trade_id, stoch_k, stoch_d, sma_slow, sma_fast,
                 renko_uptrend,
                 renko_bar_num, macd, obv, atr, adx, rsi):
    with open(file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_names3)
        writer.writerow({"Date": date,
                         "Currency": currency,
                         "Day": day,
                         "Hour": hour,
                         "Minute": minute,
                         "TradeId": trade_id,
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
                         'Stochastic_K': stoch_k,
                         'Stochastic_D': stoch_d,
                         'SMA_slow': sma_slow,
                         'SMA_fast': sma_fast,
                         'Renko_Uptrend': renko_uptrend,
                         'Renko_bar_num': renko_bar_num,
                         'MACD': macd,
                         'OBV': obv,
                         'ATR': atr,
                         'ADX': adx,
                         'RSI': rsi,
                         "Buy/Sell": order_type,
                         "Win/Loss": win_loss,
                         })

def write_logger_header(file, file_header):
    with open(file, 'w', newline='') as f:
        f.write(file_header)


def iterate_through_closed_positions(con, output_file, trainer_trades_file, log_file):
    closed_positions = con.get_closed_positions()
    df = pd.read_csv(trainer_trades_file)
    if len(closed_positions) > 0:
        try:
            for index, data in df.iterrows():
                t_id = data['TradeId']
                trade_id = closed_positions[closed_positions['tradeId'] == str(t_id)]
                if len(trade_id) > 0:
                    if trade_id['grossPL'].iloc[-1] > 0:
                        data['Win/Loss'] = 1
                    else:
                        data['Win/Loss'] = 0

                    write_to_csv(output_file,
                                 data['Currency'],
                                 data['Open'],
                                 data['High'],
                                 data['Low'],
                                 data['Close'],
                                 data['Sentiment'],
                                 data['Net Long'],
                                 data['Net Short'],
                                 data['S3'],
                                 data['S2'],
                                 data['S1'],
                                 data['P'],
                                 data['R1'],
                                 data['R2'],
                                 data['R3'],
                                 data['Stop_loss'],
                                 data['Take_Profit'],
                                 data['Buy/Sell'],
                                 data['Win/Loss'],
                                 data['Volume'],
                                 data['Date'],
                                 data['Day'],
                                 data['Hour'],
                                 data['Minute'],
                                 data['TradeId'],
                                 data["Stochastic_K"],
                                 data["Stochastic_D"],
                                 data["SMA_slow"],
                                 data["SMA_fast"],
                                 data["Renko_Uptrend"],
                                 data["Renko_bar_num"],
                                 data["MACD"],
                                 data["OBV"],
                                 data["ATR"],
                                 data["ADX"]
                                 )

                    index_to_delete = df[df['TradeId'] == t_id].index[-1]
                    df.drop(index_to_delete, inplace=True)
            write_header(trainer_trades_file)
            for index, data in df.iterrows():
                write_to_csv(trainer_trades_file,
                             data['Currency'],
                             data['Open'],
                             data['High'],
                             data['Low'],
                             data['Close'],
                             data['Sentiment'],
                             data['Net Long'],
                             data['Net Short'],
                             data['S3'],
                             data['S2'],
                             data['S1'],
                             data['P'],
                             data['R1'],
                             data['R2'],
                             data['R3'],
                             data['Stop_loss'],
                             data['Take_Profit'],
                             data['Buy/Sell'],
                             data['Win/Loss'],
                             data['Volume'],
                             data['Date'],
                             data['Day'],
                             data['Hour'],
                             data['Minute'],
                             data['TradeId'],
                             data["Stochastic_K"],
                             data["Stochastic_D"],
                             data["SMA_slow"],
                             data["SMA_fast"],
                             data["Renko_Uptrend"],
                             data["Renko_bar_num"],
                             data["MACD"],
                             data["OBV"],
                             data["ATR"],
                             data["ADX"]
                             )
        except:
            message = "From logger -Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(
                sys.exc_info()[2])
            logger(message, log_file=log_file)


def get_currency_name_for_pivot(curr):
    currency_name = curr.replace("/", "-").lower()
    return currency_name


def logger(message, log_file):
    with open(log_file, 'a', newline='') as f:
        f.write("\n" + message)


def send_email(message, subject):
    password = "Roronoa_12"
    subject = subject
    to = "kmaingi12@yahoo.com"
    FROM = "kmaingi12@hotmail.com"
    try:
        server = smtplib.SMTP(host='smtp-mail.outlook.com', port=587)
        server.starttls()
        server.login(FROM, password)
        BODY = "\r\n".join((
            "From: %s" % FROM,
            "To: %s" % to,
            "Subject: %s" % subject,
            "",
            message + "\r :)"
        ))
        server.sendmail(FROM, [to], BODY)
        server.quit()
    except:
        print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


def get_net_buy_sell():
    currency_dictionary_full = {}
    url = "https://www.dailyfx.com/sentiment-report"
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    currency_information = soup.find_all("tr")
    for index in range(1, len(currency_information)):
        currency_dictionary_full[currency_information[index].find_all("span")[0].contents[0]] = {
            "Sentiment": currency_information[index].find_all("span")[1].contents[0],
            "Net Long": currency_information[index].find_all("span")[2].contents[0],
            "Net Short": currency_information[index].find_all("span")[3].contents[0]
        }
    return currency_dictionary_full


def p2f(x):
    return float(x.strip('%'))


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
    previous_sp = ""
    if len(currency_support_resistances) > 0:
        for info in currency_support_resistances:
            index = 0
            for item in info.find_all('span'):
                if item.string.startswith("S") or item.string.startswith("R"):
                    previous_sp = item.string
                else:
                    support_resistance[previous_sp] = [float(item.string), "", index]
                    index += 1

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


def full_pivot_table():
    url = "https://www.dailyfx.com/pivot-points"
    page = requests.get(url)
    page_content = page.content
    soup = BeautifulSoup(page_content, 'html.parser')
    pivots = soup.find_all("div", {
        "class": "dfx-pivotPointsTable dfx-pivotPointsWidget__table container pt-2 pt-md-0 dfx-font-size-3"})
    pivot_dictionary = {}
    for pivot_element in range(0, len(pivots)):
        try:
            spans = pivots[pivot_element].find_all("span")
            pivot_dictionary[re.sub('\s+', '', pivots[pivot_element].a.contents[0])] = {
                re.sub('\s+', '', spans[0].contents[0]): float(re.sub('\s+', '', spans[1].contents[0])),
                re.sub('\s+', '', spans[2].contents[0]): float(re.sub('\s+', '', spans[3].contents[0])),
                re.sub('\s+', '', spans[4].contents[0]): float(re.sub('\s+', '', spans[5].contents[0])),
                re.sub('\s+', '', spans[6].contents[0]): float(re.sub('\s+', '', spans[7].contents[0])),
                re.sub('\s+', '', spans[8].contents[0]): float(re.sub('\s+', '', spans[9].contents[0])),
                re.sub('\s+', '', spans[10].contents[0]): float(re.sub('\s+', '', spans[11].contents[0])),
                re.sub('\s+', '', spans[12].contents[0]): float(re.sub('\s+', '', spans[13].contents[0]))

            }
        except IndexError:
            print("index error")
    return pivot_dictionary


def supports_less_than_close(support_resistances, close):
    supports_to_consider = []
    if support_resistances['S1'][0] < close:
        supports_to_consider.append('S1')
    if support_resistances['S2'][0] < close:
        supports_to_consider.append('S2')
    if support_resistances['S3'][0] < close:
        supports_to_consider.append('S3')
    return supports_to_consider


def resistances_greater_than_close(support_resistances, close):
    resistances_to_consider = []
    if support_resistances['R1'][0] > close:
        resistances_to_consider.append('R1')
    if support_resistances['R2'][0] > close:
        resistances_to_consider.append('R2')
    if support_resistances['R3'][0] > close:
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


def stochastic(df, a, b, c):
    """function to calculate stochastic"""
    df['k'] = ((df['Adj Close'] - df['Low'].rolling(a).min()) / (
            df['High'].rolling(a).max() - df['Low'].rolling(a).min())) * 100
    df['K'] = df['k'].rolling(b).mean()
    df['D'] = df['K'].rolling(c).mean()
    return df


def get_obv(data_frame):
    """function to calculate On Balance Volume"""
    df = data_frame.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret'] >= 0, 1, -1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']


def get_sma(df, a, b):
    """function to calculate stochastic"""
    df['sma_fast'] = df['Adj Close'].rolling(a).mean()
    df['sma_slow'] = df['Adj Close'].rolling(b).mean()
    return df


def get_macd(data_frame, a, b, c):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = data_frame.copy()
    df["MA_Fast"] = df["Adj Close"].ewm(span=a, min_periods=a).mean()
    df["MA_Slow"] = df["Adj Close"].ewm(span=b, min_periods=b).mean()
    df["MACD"] = df["MA_Fast"] - df["MA_Slow"]
    df["Signal"] = df["MACD"].ewm(span=c, min_periods=c).mean()
    df.dropna(inplace=True)
    return df["MACD"], df["Signal"]


def get_atr(data_frame, n):
    """function to calculate True Range and Average True Range"""
    df = data_frame.copy()
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    # df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return df2


def slope(ser, n):
    """function to calculate the slope of n consecutive points on a plot"""
    slopes = [index * 0 for index in range(n - 1)]
    for index in range(n, len(ser) + 1):
        y = ser[index - n:index]
        x = np.array(range(n))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = (x - x.min()) / (x.max() - x.min())
        x_scaled = sm.api.add_constant(x_scaled)
        model = sm.api.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)


def get_renko(data_frame):
    """function to convert ohlc data into renko bricks"""
    df = data_frame.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:, [0, 1, 2, 3, 4, 5]]
    df.columns = ["date", "open", "close", "high", "low", "volume"]
    df2 = Renko(df)
    df2.brick_size = round(get_atr(data_frame, 120)["ATR"][-1], 4)
    renko_df = df2.get_ohlc_data()
    renko_df["bar_num"] = np.where(renko_df["uptrend"] == True, 1, np.where(renko_df["uptrend"] == False, -1, 0))
    for index in range(1, len(renko_df["bar_num"])):
        if renko_df["bar_num"][index] > 0 and renko_df["bar_num"][index - 1] > 0:
            renko_df["bar_num"][index] += renko_df["bar_num"][index - 1]
        elif renko_df["bar_num"][index] < 0 and renko_df["bar_num"][index - 1] < 0:
            renko_df["bar_num"][index] += renko_df["bar_num"][index - 1]
    renko_df.drop_duplicates(subset="date", keep="last", inplace=True)
    return renko_df


def renko_merge(data_frame):
    """function to merging renko df with original ohlc df"""
    df = data_frame.copy()
    df["Date"] = df.index
    renko = get_renko(df)
    renko.columns = ["Date", "open", "high", "low", "close", "uptrend", "bar_num"]
    merged_df = df.merge(renko.loc[:, ["Date", "bar_num"]], how="outer", on="Date")
    merged_df["bar_num"].fillna(method='ffill', inplace=True)
    merged_df["macd"] = get_macd(merged_df, 12, 26, 9)[0]
    merged_df["macd_sig"] = get_macd(merged_df, 12, 26, 9)[1]
    merged_df["macd_slope"] = slope(merged_df["macd"], 5)
    merged_df["macd_sig_slope"] = slope(merged_df["macd_sig"], 5)
    return merged_df


def get_adx(data_frame, n):
    """function to calculate ADX"""
    df2 = data_frame.copy()
    df2['TR'] = get_atr(df2, n)[
        'TR']  # the period parameter of ATR function does not matter because period does not influence TR calculation
    df2['DMplus'] = np.where((df2['High'] - df2['High'].shift(1)) > (df2['Low'].shift(1) - df2['Low']),
                             df2['High'] - df2['High'].shift(1), 0)
    df2['DMplus'] = np.where(df2['DMplus'] < 0, 0, df2['DMplus'])
    df2['DMminus'] = np.where((df2['Low'].shift(1) - df2['Low']) > (df2['High'] - df2['High'].shift(1)),
                              df2['Low'].shift(1) - df2['Low'], 0)
    df2['DMminus'] = np.where(df2['DMminus'] < 0, 0, df2['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for index in range(len(df2)):
        if index < n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif index == n:
            TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
        elif index > n:
            TRn.append(TRn[index - 1] - (TRn[index - 1] / n) + TR[index])
            DMplusN.append(DMplusN[index - 1] - (DMplusN[index - 1] / n) + DMplus[index])
            DMminusN.append(DMminusN[index - 1] - (DMminusN[index - 1] / n) + DMminus[index])
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN'] = 100 * (df2['DMplusN'] / df2['TRn'])
    df2['DIminusN'] = 100 * (df2['DMminusN'] / df2['TRn'])
    df2['DIdiff'] = abs(df2['DIplusN'] - df2['DIminusN'])
    df2['DIsum'] = df2['DIplusN'] + df2['DIminusN']
    df2['DX'] = 100 * (df2['DIdiff'] / df2['DIsum'])
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2 * n - 1:
            ADX.append(np.NaN)
        elif j == 2 * n - 1:
            ADX.append(df2['DX'][j - n + 1:j + 1].mean())
        elif j > 2 * n - 1:
            ADX.append(((n - 1) * ADX[j - 1] + DX[j]) / n)
    df2['ADX'] = np.array(ADX)
    return df2['ADX']


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


currency_s_r = {}
for i in pairs:
    currency_s_r[i] = {}


# In[17]:

def close_all(open_pos, currency, con):
    if len(open_pos) > 0:
        if len(open_pos[open_pos["currency"] == currency]) == 1:
            con.close_all_for_symbol(currency)


# In[18]:

def close_if_support_resistances_exceeded(open_pos, _close, currency, con):
    if len(open_pos) > 0:
        if len(open_pos[open_pos["currency"] == currency]) == 1:
            if currency_s_r[currency] != {}:
                if _close <= currency_s_r[currency]['support'] or _close >= currency_s_r[currency]['resistance']:
                    con.close_all_for_symbol(currency)
                currency_s_r[currency] = {}
