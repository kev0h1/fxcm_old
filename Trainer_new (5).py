# coding: utf-8

# In[1]:

from algorithm_libraries import *
import fxcmpy

# In[2]:

# D25782575

# 5301

TOKEN = "0401e2db9d68e3f5e515e6d599bae9d7afae17b9"

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')

field_name_number = 2

trades_file_name = 'trainer_trades.csv'

output_file = 'output3.csv'

log_file = 'trainer_new_log.txt',

# In[3]:

day_of_the_week = -1
data_dictionary = {}


def execute_trade():
    sentiment_table = pd.read_csv('net_buy.txt')
    pivot_tables = pd.read_csv("pivot_tables.txt")
    for currency_pair in pairs_refined:
        try:
            if not con.is_connected():
                return
            alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD", "NAS100",
                                "UK100", "US30", "SPX500", "USOil"]
            ohlc = get_data(currency_pair, 'm15', 250, con)
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
            renko = get_renko(ohlc)
            uptrend = renko.iloc[-1]['uptrend']
            renko_bar_num = renko.iloc[-1]['bar_num']

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
                    if datetime.now().hour < 21:
                        con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=False, amount=pos_size,
                                       time_in_force='GTC', stop=signal['Support'], limit=signal['Resistance'],
                                       trailing_step=False, order_type='AtMarket')
                        open_position = con.get_open_positions()
                        if len(open_position) > 0:
                            if len(open_position[open_position['currency'] == currency_pair]) != 0:
                                currency_trade_id = open_position[open_position['currency'] == currency_pair]['tradeId'].iloc[-1]
                                #                                 print(signal["Signal"])
                            write_to_csv(field_name_num=field_name_number, file=trades_file_name, currency=currency_pair,
                                             _open=_open,
                                             high=high,
                                             low=low, close=close,
                                             sentiment=sentiment,
                                             net_long=p2f(net_long),
                                             net_short=p2f(net_short),
                                             s3=s3,
                                             s2=s2, s1=s1, p=pivot, r1=r1, r2=r2, r3=r3, stop=signal['Support'],
                                             limit=signal['Resistance'], order_type=signal["Signal"],
                                             win_loss=-1, volume=vol, date=date, day=day, hour=hour, minute=minute,
                                             trade_id=currency_trade_id, stoch_k=stoch_k, stoch_d=stoch_d,
                                             sma_slow=sma_slow, sma_fast=sma_fast,
                                             renko_uptrend=uptrend,
                                             renko_bar_num=renko_bar_num, macd=mac, obv=obv, atr=atr, adx=adx)

                elif signal["Signal"] == "Sell":
                    if datetime.now().hour < 21:
                        con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=False, amount=pos_size,
                                       time_in_force='GTC', stop=signal['Resistance'], limit=signal['Support'],
                                       trailing_step=False, order_type='AtMarket')
                        open_position = con.get_open_positions()
                        if len(open_position) > 0:
                            if len(open_position[open_position['currency'] == currency_pair]) != 0:
                                currency_trade_id = open_position[open_position['currency'] == currency_pair]['tradeId'].iloc[-1]
                                #                                 print(signal["Signal"])
                                write_to_csv(field_name_num=field_name_number, file=trades_file_name, currency=currency_pair,
                                             _open=_open,
                                             high=high,
                                             low=low, close=close,
                                             sentiment=sentiment,
                                             net_long=p2f(net_long),
                                             net_short=p2f(net_short),
                                             s3=s3,
                                             s2=s2, s1=s1, p=pivot, r1=r1, r2=r2, r3=r3, stop=signal['Resistance'],
                                             limit=signal['Support'], order_type=signal["Signal"],
                                             win_loss=-1, volume=vol, date=date, day=day, hour=hour, minute=minute,
                                             trade_id=currency_trade_id, stoch_k=stoch_k, stoch_d=stoch_d,
                                             sma_slow=sma_slow, sma_fast=sma_fast,
                                             renko_uptrend=uptrend,
                                             renko_bar_num=renko_bar_num, macd=mac, obv=obv, atr=atr, adx=adx)
        except:
            message = "From main - Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(
                sys.exc_info()[2])
            print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
            logger(message, log_file)

        # In[4]:


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
start_time = time.time()
first_iteration = True
write_closed_positions = True
while True:
    try:
        robot_message = "pass through at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\r"
        print(robot_message)
        logger(robot_message, log_file)
        if datetime.today().weekday() != 5 and datetime.today().weekday() != 6 and datetime.now().hour < 22:
            if not first_iteration:
                first_iteration = True
            if datetime.now().hour > 18 and datetime.now().minute >= 45 or datetime.now().hour > 19:
                if len(con.get_open_positions()) > 0:
                    if con.is_connected():
                        write_open_positions_to_output_file(con=con, output_file=output_file,
                                                            trainer_trades_file=trades_file_name,
                                                            log_file=log_file, field_name_num=field_name_number)

                        con.close_all()

            if datetime.now().hour < 19:
                if not con.is_connected():
                    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')
                execute_trade()
            iterate_through_closed_positions(con=con, output_file=output_file,
                                             trainer_trades_file=trades_file_name, log_file=log_file,
                                             field_name_num=field_name_number)
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

        time.sleep((60 * 15) - ((time.time() - start_time) % (60 * 15)))
    except KeyboardInterrupt:
        print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])

# In[ ]:



