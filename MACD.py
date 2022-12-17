
# coding: utf-8

# In[1]:

from algorithm_libraries import *
import fxcmpy
import talib


# In[2]:

#D25749100 - 7002

TOKEN="28f34e11abcbdb3bc17924db7d41770dafee14a5"

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')

trades_file_name = "trainer_macd.csv"

output_file = "output_macd.csv"

log_file = "log_macd.txt"

field_name_number = 4


# In[3]:

day_of_the_week = -1
data_dictionary = {}


def execute_trade():
    sentiment_table = pd.read_csv('net_buy.txt')
    pivot_tables = pd.read_csv("pivot_tables.txt") 
    supports_and_resistances = pd.read_csv('support_and_resistance.txt')
    trend = pd.read_csv("trend.txt")                   
    for currency_pair in pairs_refined:
        try:
            if not con.is_connected():
                con.connect()
            alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD", "UK100", "US30", "SPX500",
         "USOil"]
    #             ohlc = get_data(currency_pair, 'm15', 500, con)
            data = con.get_candles(currency_pair, period='m15', number=250)
            ohlc = data.iloc[:, [0, 1, 2, 3, 8]]
            ohlc.columns = ["Open", "Adj Close", "High", "Low", "Volume"]
            ohlc['weekday'] = ohlc.index.dayofweek
            pos_size = 1
            previous_close = ohlc.iloc[-2]['Adj Close']
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
            ohlc_df = get_sma(ohlc, 5, 13).dropna()
            sma_slow = ohlc_df.iloc[-1]['sma_slow']
            sma_fast = ohlc_df.iloc[-1]['sma_fast']
    #             mac = get_macd(ohlc, 12, 26, 9)[-1][-1]
            macd= talib.MACD(ohlc['Adj Close'].values, 12, 26, 9)
            macd_signal = macd[1][-1]
            previous_macd_signal = macd[1][-2]
            mac = macd[0][-1]
            previous_mac = macd[0][-2] 
            ema= talib.EMA(ohlc['Adj Close'].values, 200)[-1]
            obv = get_obv(ohlc).iloc[-1]
            atr = get_atr(ohlc, 10).dropna().iloc[-1]['ATR']
            adx = get_adx(ohlc, 14).dropna().iloc[-1]
    #             renko = get_renko(ohlc)
            uptrend = False
            renko_bar_num = 1

            if len(pivot_tables[pivot_tables["Currency"]==currency_pair])==0:
                continue

            pivot = pivot_tables[pivot_tables["Currency"]==currency_pair]["P"].iloc[-1]
            s1 = pivot_tables[pivot_tables["Currency"]==currency_pair]["S1"].iloc[-1]
            r1 = pivot_tables[pivot_tables["Currency"]==currency_pair]["R1"].iloc[-1]
            s2 = pivot_tables[pivot_tables["Currency"]==currency_pair]["S2"].iloc[-1]
            r2 = pivot_tables[pivot_tables["Currency"]==currency_pair]["R2"].iloc[-1]
            s3 = pivot_tables[pivot_tables["Currency"]==currency_pair]["S3"].iloc[-1]
            r3 = pivot_tables[pivot_tables["Currency"]==currency_pair]["R3"].iloc[-1]

            first_entry = ohlc.iloc[0]
            last_entry = ohlc.iloc[-1]

            price_min = ohlc['Adj Close'].min()
            min_df = ohlc[ohlc['Adj Close']==price_min]
            price_max = ohlc['Adj Close'].max()
            max_df = ohlc[ohlc['Adj Close']==price_max]
            diff = price_max - price_min
            level1 = price_max - 0.236 * diff
            level2 = price_max - 0.382 * diff
            level3 = price_max - 0.5 * diff
            level4 = price_max - 0.618 * diff
            level5 = price_max - 0.789 * diff

            if min_df.index[-1] != first_entry.index[-1] and min_df.index[-1] != last_entry.index[-1] and max_df.index[-1] != first_entry.index[-1] and max_df.index[-1] != last_entry.index[-1]:        
                sentiment_table_name = currency_pair
                if currency_pair in alternative_list:
                    sentiment_table_name = full_sentiment_dictionary[currency_pair]

                low = ohlc_df['Low'].iloc[-2]
                high = ohlc_df['High'].iloc[-2]

                if len(supports_and_resistances[supports_and_resistances["Currency"]==currency_pair])==0:
                    continue

                resistance_1 = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["R1"].iloc[-1]
                resistance_1_strength = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["R1 strength"].iloc[-1]
                resistance_2 = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["R2"].iloc[-1]
                resistance_2_strength = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["R2 strength"].iloc[-1]
                resistance_3 = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["R3"].iloc[-1]
                resistance_3_strength = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["R3 strength"].iloc[-1]

                support_1 = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["S1"].iloc[-1]
                support_1_strength = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["S1 strength"].iloc[-1]
                support_2 = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["S2"].iloc[-1]
                support_2_strength = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["S2 strength"].iloc[-1]
                support_3 = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["S3"].iloc[-1]
                support_3_strength = supports_and_resistances[supports_and_resistances['Currency']==currency_pair]["S3 strength"].iloc[-1]

                net_long = sentiment_table[sentiment_table['Currency']==sentiment_table_name]["Net Long"].iloc[-1]
                net_short =sentiment_table[sentiment_table['Currency']==sentiment_table_name]["Net Short"].iloc[-1]
                sentiment =sentiment_table[sentiment_table['Currency']==sentiment_table_name]["Sentiment"].iloc[-1]

                trend_direction = trend[trend['Currency']==sentiment_table_name]["Trend"].iloc[-1]

                signal = pivot_with_sentiment_signal(close, macd_signal, previous_macd_signal, mac, previous_mac, ema)

                if signal != {}:
                    if signal["Signal"] == "Buy":
                        take_profit = ohlc.iloc[-1]['Adj Close'] + 2.25 * atr
                        stop_loss = ohlc.iloc[-1]['Adj Close'] - 1 * atr
                        if datetime.now().hour < 21:
                            token = con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=False, amount=pos_size,
                                           time_in_force='GTC', stop=stop_loss, limit=take_profit,
                                           trailing_step=False, order_type='AtMarket')

                            if type(token) is fxcmpy.fxcmpy_order:
                                currency_trade_id = token.get_tradeId()
                                write_to_csv(field_name_num=field_name_number, file_name=trades_file_name, currency=currency_pair, _open=_open,
                                             high=high,
                                             low=low, close=close,
                                             sentiment=sentiment,
                                             net_long=p2f(net_long),
                                             net_short=p2f(net_short),
                                             s3=s3,
                                             s2=s2, s1=s1, p=pivot, r1=r1, r2=r2, r3=r3, stop=stop_loss,
                                             limit=take_profit, order_type=signal["Signal"],
                                             win_loss=-1, volume=vol, date=date, day=day, hour=hour, minute=minute,
                                             trade_id=currency_trade_id, stoch_k=stoch_k, stoch_d=stoch_d,
                                             sma_slow=sma_slow, sma_fast=sma_fast,
                                             renko_uptrend=uptrend,
                                             renko_bar_num=renko_bar_num, macd=mac, obv=obv, atr=atr, adx=adx,
                                             resistance_1=resistance_1, resistance_1_strength=resistance_1_strength,
                                             resistance_2=resistance_2,
                                             resistance_2_strength=resistance_2_strength,
                                             resistance_3=resistance_3, resistance_3_strength=resistance_3_strength,
                                             support_1=support_1,
                                             support_1_strength=support_1_strength,
                                             support_2=support_2, support_2_strength=support_2_strength,
                                             support_3=support_3, support_3_strength=support_3_strength, 
                                             trend=trend_direction, price_min=price_min, price_max=price_max, level1=level1, 
                                             level2=level2, level3=level3, level4=level4, level5=level5)

                    elif signal["Signal"] == "Sell":
                        take_profit = ohlc.iloc[-1]['Adj Close'] - 2.25 * atr
                        stop_loss = ohlc.iloc[-1]['Adj Close'] + 1 * atr
                        if datetime.now().hour < 21:
                            token = con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=False, amount=pos_size,
                                           time_in_force='GTC', stop=stop_loss, limit=take_profit,
                                           trailing_step=False, order_type='AtMarket')
                            if type(token) is fxcmpy.fxcmpy_order:
                                currency_trade_id = token.get_tradeId()
                                write_to_csv(field_name_num=field_name_number, file_name=trades_file_name, currency=currency_pair, _open=_open,
                                             high=high,
                                             low=low, close=close,
                                             sentiment=sentiment,
                                             net_long=p2f(net_long),
                                             net_short=p2f(net_short),
                                             s3=s3,
                                             s2=s2, s1=s1, p=pivot, r1=r1, r2=r2, r3=r3, stop=stop_loss,
                                             limit=take_profit, order_type=signal["Signal"],
                                             win_loss=-1, volume=vol, date=date, day=day, hour=hour, minute=minute,
                                             trade_id=currency_trade_id, stoch_k=stoch_k, stoch_d=stoch_d,
                                             sma_slow=sma_slow, sma_fast=sma_fast,
                                             renko_uptrend=uptrend,
                                             renko_bar_num=renko_bar_num, macd=mac, obv=obv, atr=atr, adx=adx,
                                             resistance_1=resistance_1, resistance_1_strength=resistance_1_strength,
                                             resistance_2=resistance_2,
                                             resistance_2_strength=resistance_2_strength,
                                             resistance_3=resistance_3, resistance_3_strength=resistance_3_strength,
                                             support_1=support_1,
                                             support_1_strength=support_1_strength,
                                             support_2=support_2, support_2_strength=support_2_strength,
                                             support_3=support_3, support_3_strength=support_3_strength, 
                                             trend=trend_direction, price_min=price_min, price_max=price_max, level1=level1, 
                                             level2=level2, level3=level3, level4=level4, level5=level5)


        except Exception as e:
            if type(e) is OSError:
                error_message = "Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(
                    sys.exc_info()[2])
                print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
                logger(error_message, log_file)
                return
            else:
                error_message = "Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(
                        sys.exc_info()[2])
                print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
                logger(error_message, log_file)


# In[4]:

# def pivot_with_sentiment_signal(high, low, _close, price_min, price_max, level1, level2, level3, level4, level5):
    
#     if low < level1 < _close:
#         return {"Signal": "Buy", "Support": (level1+level2)/2, "Resistance": price_max}
#     if low < level2 < _close:
#         return {"Signal": "Buy", "Support": (level2+level3)/2, "Resistance": level1}
#     if low < level3 < _close:
#         return {"Signal": "Buy", "Support": (level3+level4)/2, "Resistance": level2}
#     if low < level4 < _close:
#         return {"Signal": "Buy", "Support": (level4+level5)/2, "Resistance": level3}
#     if low < level5 < _close:
#         return {"Signal": "Buy", "Support": (level5+price_min)/2, "Resistance": level4}
    
#     if high > level1 > _close:
#         return {"Signal": "Sell", "Support": level2, "Resistance": (level1 + price_max)/2}
#     if high > level2 > _close:
#         return {"Signal": "Sell", "Support": level3, "Resistance": (level1 + level2)/2}
#     if high > level3 > _close:
#         return {"Signal": "Sell", "Support": level4, "Resistance": (level2+level3)/2}
#     if high > level4 > _close:
#         return {"Signal": "Sell", "Support": level5, "Resistance": (level3+level4)/2}
#     if high > level5 > _close:
#         return {"Signal": "Sell", "Support": price_min, "Resistance": (level4+level5)/2}


#     return {}

def pivot_with_sentiment_signal(_close, macd_signal, previous_macd_signal, mac, previous_mac, ema):
    
    if _close > ema and previous_macd_signal > previous_mac and macd_signal < mac and mac < 0:
         return {"Signal": "Buy"}
    
    if _close < ema and previous_macd_signal < previous_mac and macd_signal > mac and mac > 0:
         return {"Signal": "Sell"}
        
    return {}


# In[ ]:

count = 0

first_iteration = True
write_closed_positions = True
while True:
    if datetime.today().minute is 15 or datetime.today().minute is 30 or datetime.today().minute is 45 or datetime.today().minute is 0:
        start_time = time.time()
        try:
            robot_message = "pass through at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\r"
            print(robot_message)
            logger(robot_message, log_file)
            if datetime.today().weekday() != 5 and datetime.today().weekday() != 6 and 0 < datetime.now().hour < 22:
                if datetime.now().hour > 18 and datetime.now().minute >= 45 or datetime.now().hour > 19:
                    if len(con.get_open_positions()) > 0:
                        if con.is_connected():
                            open_positions = con.get_open_positions()
                            write_open_positions_to_output_file(output_file=output_file, trainer_trades_file=trades_file_name,
                                                 log_file=log_file, field_name_num=field_name_number, open_positions=open_positions)
                            con.close_all()

                if datetime.now().hour < 19:
                    if not con.is_connected():
                        con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                    execute_trade()
                closed_positions = con.get_closed_positions() 
                iterate_through_closed_positions(output_file=output_file, trainer_trades_file=trades_file_name,
                                                 log_file=log_file, field_name_num=field_name_number, closed_positions=closed_positions)
            else:
                write_closed_positions = True
                if datetime.now().hour == 0 and datetime.now().minute < 15:
                    first_iteration = True
                if first_iteration:
                    write_header(trades_file_name, field_names4)
                    write_logger_header(log_file, 'New Trainer logger')
                    first_iteration = False
        except Exception as e:
            if type(e) is OSError:
                error_message = "Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(
                    sys.exc_info()[2])
                print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
                logger(error_message, log_file)

            elif type(e) is KeyboardInterrupt:
                print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
                break

            else:
                error_message = "Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(
                        sys.exc_info()[2])
                print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
                logger(error_message, log_file)
                try:
                    a_del=[]
                    for module in sys.modules.keys():
                        if 'fxcm' in module:
                            a_del.append(module)

                    for module in a_del:
                        del sys.modules[module]

                    del fxcmpy
                    import fxcmpy
                    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                except:
                    logger("could not connect", log_file)

        time.sleep((60 * 15) - ((time.time() - start_time) % (60 * 15)))



# In[ ]:



