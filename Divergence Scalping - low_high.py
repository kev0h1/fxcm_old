#!/usr/bin/env python
# coding: utf-8

# In[1]:


from algorithm_libraries import *
import fxcmpy
import pandas as pd
from random import randint
from finta import TA
import pickle
import talib

# D25833410

# 9lRju

TOKEN = "4be79efa5f283d34aad9f97a90e510031758471e"

output_file_name = '../../Downloads/mcs.csv'

log_file = "mlb2.log"

trace_file = "trace2.txt"

risk = 0.5

pairs_forex = ["AUD/USD", "EUR/USD", "GBP/USD", "USD/CHF", "USD/JPY", "AUD/JPY", "EUR/GBP", "EUR/JPY", "NZD/USD", "NZD/JPY", "USD/CAD"]

headers = {
    "x-requested-with": "XMLHttpRequest",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36",
    "referer": "https://www.myfxbook.com/",
}

option_data = {
    "colSymbols": "8,9,10,6,7,1,4,2,5,3",
    "rowSymbols": "8,47,9,10,1234,11,103,12,46,1245,6,13,14,15,17,18,7,2114,19,20,21,22,1246,23,1,1233,107,24,25,4,2872,137,48,1236,1247,2012,2,1863,3240,26,49,27,28,2090,131,5,29,5779,31,34,3,36,37,38,2076,40,41,42,43,45,3005,3473,50,2115,2603,2119,1815,2521,51,12755,5435,5079,10064,1893",
    # Change the value of `timeScale` to get the different Timeframe, the value should be by minutes, for example to get 1 week, use `10080`
    "timeScale": "5",
    "z": "0.6367404250506281",
}


# CAD/CHF, CAD/JPY, EUR/AUD(M) EUR/CHF(m) "EUR/NZD",GBP/AUD, NZD/JPY AUD/NZD
# pairs_forex = ["CAD/CHF","GBP/CAD", "EUR/NZD"]
class rsi_low_high:
    def __init__(self, _price, _rsi, _trend):
        self.price = _price
        self.rsi = _rsi
        self.trend = _trend


def reset_files():
    for currency_pair in pairs_forex:
        alternative_pair_name = currency_pair.replace("/", "")
        rsi_values_list = []
        signal_counter = -1
        signal = ""
        serialise_object("DivPip/LowSpread/" + alternative_pair_name + "/list", rsi_values_list)


def ten_pip_value(currency_pair):
    if "JPY" in currency_pair:
        return 0.1
    return 0.001


def dochian_channel_formal(_ohlc):
    _df = _ohlc.copy()
    _df.columns = ["open", "close", "high", "low", "Ask close", "volume", "weekday"]
    return _df


def calculate_lotsize(risk_percentage, stoploss_in_pips, equity):
    return ((risk_percentage / 100) * equity) / (stoploss_in_pips * 10)


def get_market_data(_currency_pair, per, candle_number, _con):
    _data = _con.get_candles(_currency_pair, period=per, number=candle_number)
    _ohlc = _data.iloc[:, [0, 1, 2, 3, 5, 8]]
    _ohlc.columns = ["Open", "Adj Close", "High", "Low", "Ask Close", "Volume"]
    _ohlc['weekday'] = _ohlc.index.dayofweek
    return _ohlc


def serialise_object(path, obj):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def unpack_object(path):
    with open(path, "rb") as fp:
        b = pickle.load(fp)
    return b


restart = False


def insert_slash(string, index):
    return string[:3] + '/' + string[3:]


def main():
    response = requests.post(
        "https://www.myfxbook.com/updateCorrelationSymbolMenu.json",
        headers=headers,
        data=option_data,
    ).json()

    table_html = response["content"]["marketCorrelationTable"]
    soup = BeautifulSoup(table_html, "html.parser")
    data_frame = pd.read_html(str(soup))[0]
    data_frame.set_index("Currency", inplace=True)
    for column in data_frame:
        data_frame[column] = data_frame[column].str.rstrip('%').astype(float)

    for currency_pair in pairs_forex:
        try:
            alternative_pair_name = currency_pair.replace("/", "")
            df_filter = data_frame[data_frame.index == alternative_pair_name]
            df_2 = df_filter[((abs(df_filter)) > 85) & ((abs(df_filter)) < 100)]
            min_currency = ""
            max_currency = ""
            if type(df_2.min().idxmin()) is not float:
                if df_2.min().min() < 0:
                    min_currency = insert_slash(df_2.min().idxmin(), 3)
            if type(df_2.max().idxmax()) is not float:
                if df_2.max().max() > 0:
                    max_currency = insert_slash(df_2.max().idxmax(), 3)

            ohlc = get_market_data(currency_pair, "m5", 30, con)
            d2 = dochian_channel_formal(ohlc)
            ohlc_obv = get_obv(ohlc)
            ohlc_obv_df = pd.DataFrame({'index': ohlc_obv.index, 'OBV': ohlc_obv.values})
            ohlc_obv_df.dropna(inplace=True)
            ohlc_obv_df['MA'] = ohlc_obv_df['OBV'].rolling(14).mean()

            ma_slow = TA.SMA(d2, 14)[-1]
            equity = con.get_accounts_summary()['equity'][0]
            adx_row = TA.ADX(d2)[-1]
            signal = ""

            spread = abs(ohlc.iloc[-1]["Adj Close"] - ohlc.iloc[-1]["Ask Close"])

            obv_1_row = ohlc_obv[-2]
            ohlc_row = ohlc.iloc[-1]
            ohlc_1_row = ohlc.iloc[-2]
            ohlc_2_row = ohlc.iloc[-3]
            obv_ma_row = ohlc_obv_df.iloc[-1]
            
            rsi = TA.RSI(d2)[-2]
            
            pip_min = 2.5

            # even though the list is called rsi, obv divergence is being used

            if ohlc_row['Adj Close'] > ma_slow:
                trend = "up"
            else:
                trend = "down"

            if trend == "up":
                if ohlc_row['Low'] > ohlc_1_row['Low'] and ohlc_1_row['Low'] < ohlc_2_row['Low'] and ohlc_row['Open']< ohlc_row['Adj Close']:
                    signal = "buy"

            else:
                if ohlc_row['High'] < ohlc_1_row['High'] and ohlc_1_row['High'] > ohlc_2_row['High'] and ohlc_row['Open']> ohlc_row['Adj Close']:
                    signal = "sell"

            if "JPY" in currency_pair:
                scalar = 100
            else:
                scalar = 10000

            if signal == "buy" and spread < pip_min and adx_row > 25 and obv_ma_row['MA'] < obv_ma_row['OBV']:
                stop_loss = (abs(ohlc_row['Ask Close'] - ohlc_1_row['Low']) * scalar) + 1
                if stop_loss > 10:
                    stop_loss = 10
                if max_currency != "":
                    found = False
                    corr_ohlc = get_market_data(max_currency, "m5", 10, con)
                    correlation_index = -1
                    for index, correlation_data in corr_ohlc[::-1].iterrows():
                        _ohlc_1_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index - 1].name]
                        if abs(correlation_index - 1) == len(ohlc):
                            break
                        _ohlc_2_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index - 2].name]
                        if correlation_data['Low'] > _ohlc_1_row['Low'][-1] and _ohlc_1_row['Low'][-1] < _ohlc_2_row['Low'][-1]:
                            found = True
                            break
                        correlation_index -= 1
                        
                    _ohlc_1_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index].name]
                    pips = abs(corr_ohlc.iloc[-1]["Ask Close"] - _ohlc_1_row["Ask Close"][-1]) * scalar
                    if pips > pip_min and found == True:
                        pos_size = round(0.01 * risk * equity)
                        logger("buy initiated for " + currency_pair, log_file)
                        logger("stop loss :" + str(stop_loss), log_file)
                        logger("take profit b1:" + str(pips), log_file)
                        if pips > 20:
                            pips = 20
                            logger("pips set to 20", log_file)
                            
                        con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=True, amount=pos_size,
                                       time_in_force='GTC', stop=-stop_loss, limit=pips,
                                       trailing_step=False, order_type='AtMarket')

                if min_currency != "":
                    found = False
                    corr_ohlc = get_market_data(min_currency, "m5", 10, con)
                    correlation_index = -1
                    for index, correlation_data in corr_ohlc[::-1].iterrows():
                        _ohlc_1_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index - 1].name]
                        if abs(correlation_index - 1) == len(ohlc):
                            break
                        _ohlc_2_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index - 2].name]
                        if correlation_data['High'] < _ohlc_1_row['High'][-1] and _ohlc_1_row['High'][-1] > _ohlc_2_row['High'][-1]:
                            found = True
                            break
                        correlation_index -= 1
                        
                    _ohlc_1_row = ohlc[ohlc.index == ohlc.iloc[correlation_index].name]
                    pips = abs(ohlc.iloc[-1]["Adj Close"] - _ohlc_1_row["Adj Close"][-1]) * scalar
                    if pips > pip_min and found == True:
                        pos_size = round(0.01 * risk * equity)
                        logger("buy initiated for " + currency_pair, log_file)
                        logger("stop loss :" + str(stop_loss), log_file)
                        logger("take profit b2:" + str(pips), log_file)
                        if pips > 20:
                            pips = 20
                            logger("pips set to 20", log_file)
                        con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=True, amount=pos_size,
                                       time_in_force='GTC', stop=-stop_loss, limit=pips,
                                       trailing_step=False, order_type='AtMarket')

            elif signal == "sell" and spread < pip_min and adx_row > 25 and obv_ma_row['MA'] > obv_ma_row['OBV']:
                stop_loss = (abs(ohlc_row['Adj Close'] - ohlc_1_row['High']) * scalar) + 1
                if stop_loss > 10:
                    stop_loss = 10
                if max_currency != "":
                    found = False
                    corr_ohlc = get_market_data(max_currency, "m5", 10, con)
                    correlation_index = -1
                    for index, correlation_data in corr_ohlc[::-1].iterrows():
                        _ohlc_1_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index - 1].name]
                        if abs(correlation_index - 1) == len(ohlc):
                            break
                        _ohlc_2_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index - 2].name]
                        if correlation_data['High'] < _ohlc_1_row['High'][-1] and _ohlc_1_row['High'][-1] > _ohlc_2_row['High'][-1]:
                            found = True
                            break
                        correlation_index -= 1

                    _ohlc_1_row = ohlc[ohlc.index == ohlc.iloc[correlation_index].name]
                    pips = abs(ohlc.iloc[-1]["Adj Close"] - _ohlc_1_row["Adj Close"][-1]) * scalar
                    if pips > pip_min and found == True:
                        pos_size = round(0.01 * risk * equity)
                        logger("sell initiated for " + currency_pair, log_file)
                        logger("stop loss :" + str(stop_loss), log_file)
                        logger("take profit s1:" + str(pips), log_file)
                        if pips > 20:
                            pips = 20
                            logger("pips set to 20", log_file)
                        con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=True,
                                       amount=pos_size,
                                       time_in_force='GTC', stop=-stop_loss, limit=pips,
                                       trailing_step=False, order_type='AtMarket')

                if min_currency != "":
                    found = False
                    corr_ohlc = get_market_data(min_currency, "m5", 10, con)
                    correlation_index = -1
                    for index, correlation_data in corr_ohlc[::-1].iterrows():
                        _ohlc_1_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index - 1].name]
                        if abs(correlation_index - 1) == len(ohlc):
                            break
                        _ohlc_2_row = corr_ohlc[corr_ohlc.index == corr_ohlc.iloc[correlation_index - 2].name]
                        if correlation_data['Low'] > _ohlc_1_row['Low'][-1] and _ohlc_1_row['Low'][-1] < _ohlc_2_row['Low'][-1]:
                            found = True
                            break
                        correlation_index -= 1

                    _ohlc_1_row = ohlc[ohlc.index == ohlc.iloc[correlation_index].name]
                    pips = abs(ohlc.iloc[-1]["Ask Close"] - _ohlc_1_row["Ask Close"][-1]) * scalar
                    if pips > pip_min and found == True:
                        pos_size = round(0.01 * risk * equity)
                        logger("sell initiated for " + currency_pair, log_file)                       
                        logger("stop loss :" + str(stop_loss), log_file)
                        logger("take profit s2:" + str(pips), log_file)
                        if pips > 20:
                            pips = 20
                            logger("pips set to 20", log_file)
                        con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=True,
                                       amount=pos_size,
                                       time_in_force='GTC', stop=-stop_loss, limit=pips,
                                       trailing_step=False, order_type='AtMarket')


        except Exception as exception:
            if type(exception) is ValueError:
                trace_data = traceback.format_exc()
                logger("Error with " + currency_pair + " " + str(datetime.now()), log_file)
                logger("Error at time" + str(datetime.now()), log_file)
                logger(currency_pair, trace_file)
                logger(str(datetime.now()) + "\n" + trace_data + "\n", trace_file)
            else:
                trace_data = traceback.format_exc()
                logger("Error at time" + str(datetime.now()), log_file)
                logger(currency_pair, trace_file)
                logger(str(datetime.now()) + "\n" + trace_data + "\n", trace_file)
#                 return


# In[ ]:


first_iteration = True
while True:
    if datetime.today().minute % 5 == 0 and datetime.today().second == 1:
        start_time = time.time()
        try:
            message = "pass through at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            logger(message, log_file)
            print(message)
            if datetime.today().weekday() == 0 or datetime.today().weekday() == 1 or datetime.today().weekday() == 2 or datetime.today().weekday() == 6 and datetime.now().hour > 20 or datetime.today().weekday() == 3 or datetime.today().weekday() == 4:
                if not first_iteration:
                    first_iteration = True
                con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='demo',
                                    log_file='debug_2.txt')
                if datetime.now().hour > 3 and datetime.now().hour < 22:
                    main()
                
                if datetime.now().hour == 21 and datetime.today().minute >=50:
                    con.close_all()
                con.close()
                del con
            if datetime.now().minute == 0 and datetime.now().hour == 22:
                trained = False
                if first_iteration:
                    write_logger_header(log_file, 'Real Logger')
                    write_logger_header(trace_file, 'Trace Logger')
                    first_iteration = False

        except Exception as e:
            if type(e) is KeyboardInterrupt:
                trace = traceback.format_exc()
                logger("\n" + str(datetime.now()) + "\n" + trace + "\n", trace_file)
                break
            else:
                trace = traceback.format_exc()
                logger("restart failed", log_file)
                try:
                    a_del = []
                    for module in sys.modules.keys():
                        if 'fxcm' in module:
                            a_del.append(module)

                    for module in a_del:
                        del sys.modules[module]

                    del fxcmpy
                    import fxcmpy
                except:
                    trace = traceback.format_exc()
                    logger("fxcm restart failed", log_file)
                    logger("\n" + str(datetime.now()) + "\n" + trace + "\n", trace_file)

        time.sleep((60 * 4.5) - ((time.time() - start_time) % (60 * 4.5)))


# In[ ]:




