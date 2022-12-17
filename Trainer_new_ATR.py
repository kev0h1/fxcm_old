
# coding: utf-8

# In[1]:

from algorithm_libraries import *
import fxcmpy


# In[2]:

#D25814412

#r1eiB

TOKEN = "dfed7c226f04de4afa1207a8da7143e210b98b9e"

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')

field_name_number = 2

trades_file_name = 'trainer_trades_1.csv'

output_file = 'output3_1.csv'

log_file = 'log_trainer_new_1.txt'


# In[3]:

day_of_the_week = -1
data_dictionary = {}


def execute_trade():
    sentiment_table = pd.read_csv('net_buy.txt')
    pivot_tables = pd.read_csv("pivot_tables.txt")
    for currency_pair in pairs_refined:
        try:
            if not con.is_connected():
                con.connect()
                
            alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD", "NAS100",
                                "UK100", "US30", "SPX500", "USOil"]
#             ohlc = get_data(currency_pair, 'm15', 250, con)
            data = con.get_candles(currency_pair, period='m15', number=250)
            ohlc = data.iloc[:, [0, 1, 2, 3, 8]]
            ohlc.columns = ["Open", "Adj Close", "High", "Low", "Volume"]
            ohlc['weekday'] = ohlc.index.dayofweek
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
            ohlc_df = get_sma(ohlc, 5, 13).dropna()
            sma_slow = ohlc_df.iloc[-1]['sma_slow']
            sma_fast = ohlc_df.iloc[-1]['sma_fast']
            mac = get_macd(ohlc, 12, 26, 9)[-1][-1]
            obv = get_obv(ohlc).iloc[-1]
            atr = get_atr(ohlc, 10).dropna().iloc[-1]['ATR']
            adx = get_adx(ohlc, 14).dropna().iloc[-1]
#             renko = get_renko(ohlc)
            uptrend = False
            renko_bar_num = -1

            if len(pivot_tables[pivot_tables["Currency"] == currency_pair]) == 0:
                continue

            pivot = pivot_tables[pivot_tables["Currency"] == currency_pair]["P"].iloc[-1]
            s1 = pivot_tables[pivot_tables["Currency"] == currency_pair]["S1"].iloc[-1]
            r1 = pivot_tables[pivot_tables["Currency"] == currency_pair]["R1"].iloc[-1]
            s2 = pivot_tables[pivot_tables["Currency"] == currency_pair]["S2"].iloc[-1]
            r2 = pivot_tables[pivot_tables["Currency"] == currency_pair]["R2"].iloc[-1]
            s3 = pivot_tables[pivot_tables["Currency"] == currency_pair]["S3"].iloc[-1]
            r3 = pivot_tables[pivot_tables["Currency"] == currency_pair]["R3"].iloc[-1]

            sentiment_table_name = currency_pair
            if currency_pair in alternative_list:
                sentiment_table_name = full_sentiment_dictionary[currency_pair]

            low = ohlc_df['Low'].iloc[-2]
            high = ohlc_df['High'].iloc[-2]

            net_long = sentiment_table[sentiment_table['Currency'] == sentiment_table_name]["Net Long"].iloc[-1]
            net_short = sentiment_table[sentiment_table['Currency'] == sentiment_table_name]["Net Short"].iloc[-1]
            sentiment = sentiment_table[sentiment_table['Currency'] == sentiment_table_name]["Sentiment"].iloc[-1]

            signal = pivot_with_sentiment_signal(high, low, pivot, close, s1, s2, s3, r1, r2, r3)

            if signal != {}:
                if signal["Signal"] == "Buy":
                    take_profit = ohlc.iloc[-1]['Adj Close'] + 3 * atr
                    stop_loss = ohlc.iloc[-1]['Adj Close'] - 1.25 * atr
                    if datetime.now().hour < 21:
                        token = con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=False, amount=pos_size,
                                       time_in_force='GTC', stop=stop_loss, limit=take_profit,
                                       trailing_step=False, order_type='AtMarket')
                      
                        if type(token) is fxcmpy.fxcmpy_order:
                            currency_trade_id = token.get_tradeId()

                            write_to_csv(field_name_num=field_name_number, file_name=trades_file_name, currency=currency_pair,
                                             _open=_open, high=high,
                                             low=low, close=close,
                                             sentiment=sentiment,
                                             net_long=p2f(net_long),
                                             net_short=p2f(net_short),
                                             s3=s3,
                                             s2=s2, s1=s1, p=pivot, r1=r1, r2=r2, r3=r3, stop=stop_loss,
                                             limit=take_profit,
                                             order_type=signal['Signal'],
                                             win_loss=-1, volume=vol, date=date, day=day, hour=hour, minute=minute,
                                             trade_id=currency_trade_id, stoch_k=stoch_k, stoch_d=stoch_d,
                                             sma_slow=sma_slow, sma_fast=sma_fast,
                                             renko_uptrend=uptrend,
                                                     renko_bar_num=renko_bar_num, macd=mac, obv=obv, atr=atr, adx=adx)

                elif signal["Signal"] == "Sell":
                    take_profit = ohlc.iloc[-1]['Adj Close'] - 3 * atr
                    stop_loss = ohlc.iloc[-1]['Adj Close'] + 1.25 * atr
                    if datetime.now().hour < 21:
                        token = con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=False, amount=pos_size,
                                       time_in_force='GTC', stop=stop_loss, limit=take_profit,
                                       trailing_step=False, order_type='AtMarket')
                        if type(token) is fxcmpy.fxcmpy_order:
                            currency_trade_id = token.get_tradeId()

                            write_to_csv(field_name_num=field_name_number, file_name=trades_file_name, currency=currency_pair,
                                         _open=_open,
                                         high=high,
                                         low=low, close=close,
                                         sentiment=sentiment,
                                         net_long=p2f(net_long),
                                         net_short=p2f(net_short),
                                         s3=s3,
                                         s2=s2, s1=s1, p=pivot, r1=r1, r2=r2, r3=r3, stop=stop_loss,
                                         limit=take_profit, 
                                         order_type=signal['Signal'],
                                         win_loss=-1, 
                                         volume=vol, date=date, day=day, hour=hour, minute=minute,
                                         trade_id=currency_trade_id, stoch_k=stoch_k, stoch_d=stoch_d,
                                         sma_slow=sma_slow, sma_fast=sma_fast,
                                         renko_uptrend=uptrend,
                                         renko_bar_num=renko_bar_num, macd=mac, obv=obv, atr=atr, adx=adx)
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


# In[ ]:

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
            if datetime.today().weekday() != 5 and datetime.today().weekday() != 6 and datetime.now().hour < 22:
                if datetime.now().hour > 18 and datetime.now().minute >= 45 or datetime.now().hour > 19:
                    if len(con.get_open_positions()) > 0:
                        if con.is_connected():
                            open_positions = con.get_open_positions()
                            write_open_positions_to_output_file(output_file=output_file,
                                                                trainer_trades_file=trades_file_name,
                                                                log_file=log_file, field_name_num=field_name_number, open_positions=open_positions)

                            con.close_all()

                if datetime.now().hour < 19:
                    if not con.is_connected():
                        con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                    execute_trade()

                closed_positions = con.get_closed_positions() 
                iterate_through_closed_positions(output_file=output_file,
                                                 trainer_trades_file=trades_file_name, log_file=log_file,
                                                 field_name_num=field_name_number, closed_positions=closed_positions)
            else:
                if con.is_connected():
                    con.close()
                write_closed_positions = True
                if datetime.now().hour == 0 and datetime.now().minute < 15:
                    first_iteration = True
                if first_iteration:
                    write_header(trades_file_name, field_names2)
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

