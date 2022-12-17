
# coding: utf-8

# In[2]:

from algorithm_libraries_strict import *
import fxcmpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from random import randint

TOKEN = "11d570560cb3036f2e2dd229fee34bc5df1ff0d4"

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='real', log_file='real_fxcm.txt')

output_file_name = 'output3.csv'

trades_file_name = "real_trades.csv"

log_file = 'real_log.txt'

output_file_dummy = "real_output.csv"

k_scaler = 1000

leverage = {'EUR/USD': 30, 'USD/JPY': 30, 'AUD/USD': 30, 'GBP/USD': 30, 'USD/CAD': 30,
            'NZD/USD': 30, 'AUD/JPY': 20, 'EUR/CHF': 20, 'EUR/GBP': 20, 'EUR/JPY': 20,
            'GBP/JPY': 20, 'USD/CHF': 30, 'BTC/USD': 2, "ETH/USD": 2, "LTC/USD": 2, 'GER30': 20, 'FRA40': 20,
                   "UK100": 20, "US30": 20, "SPX500":20, "USOil": 20}

lot_size_scaler = {'EUR/USD': 10000, 'USD/JPY': 100, 'AUD/USD': 10000, 'GBP/USD': 10000, 'USD/CAD': 10000,
                   'NZD/USD': 10000, 'AUD/JPY': 100, 'EUR/CHF': 10000, 'EUR/GBP': 10000, 'EUR/JPY': 100,
                   'GBP/JPY': 100, 'USD/CHF': 10000, 'BTC/USD': 1, "ETH/USD": 1, "LTC/USD": 100, 'GER30': 1, 'FRA40': 1,
                   "UK100": 1, "US30": 1, "SPX500":1, "USOil": 100}

non_farm_payroll = {'January':10 , 'February':7, 'March':6, 'April':3, 'May':8, 'June':5, 'July':2, 'August':7, 
                    'September':4, 'October':2, 'November':6, 'December':4
                   }

day_of_the_week = -1
data_dictionary = {}

risk = 2.5
risk_percentage = risk / 100


def main():
    sentiment_table = pd.read_csv('net_buy.txt')
    pivot_tables = pd.read_csv("pivot_tables.txt") 
    supports_and_resistances = pd.read_csv('support_and_resistance.txt')
    open_position = con.get_open_positions()
            
    for currency_pair in pairs:
        try:
            if not con.is_connected():
                con.connect()
            alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD", "UK100", "US30", "SPX500",
         "USOil"]

            if datetime.today().day == non_farm_payroll[datetime.now().strftime("%B")]:
                if 'USD' in currency_pair or currency_pair == "USOil" or currency_pair == "US30" or currency_pair == "SPX500":
                    continue

#             ohlc = get_data(currency_pair, 'm15', 250, con)
            data = con.get_candles(currency_pair, period='m15', number=250)
            ohlc = data.iloc[:, [0, 1, 2, 3, 8]]
            ohlc.columns = ["Open", "Adj Close", "High", "Low", "Volume"]
            ohlc['weekday'] = ohlc.index.dayofweek
            
            equity = con.get_accounts_summary()['equity'][0]
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

            if len(pivot_tables[pivot_tables["Currency"]==currency_pair])==0:
                continue

            pivot = pivot_tables[pivot_tables["Currency"]==currency_pair]["P"].iloc[-1]
            s1 = pivot_tables[pivot_tables["Currency"]==currency_pair]["S1"].iloc[-1]
            r1 = pivot_tables[pivot_tables["Currency"]==currency_pair]["R1"].iloc[-1]
            s2 = pivot_tables[pivot_tables["Currency"]==currency_pair]["S2"].iloc[-1]
            r2 = pivot_tables[pivot_tables["Currency"]==currency_pair]["R2"].iloc[-1]
            s3 = pivot_tables[pivot_tables["Currency"]==currency_pair]["S3"].iloc[-1]
            r3 = pivot_tables[pivot_tables["Currency"]==currency_pair]["R3"].iloc[-1]

            sentiment_table_name = currency_pair
            if currency_pair in alternative_list:
                sentiment_table_name = full_sentiment_dictionary[currency_pair]

            low = ohlc_df['Low'].iloc[-2]
            high = ohlc_df['High'].iloc[-2]

            net_long = sentiment_table[sentiment_table['Currency']==sentiment_table_name]["Net Long"].iloc[-1]
            net_short =sentiment_table[sentiment_table['Currency']==sentiment_table_name]["Net Short"].iloc[-1]
            sentiment =sentiment_table[sentiment_table['Currency']==sentiment_table_name]["Sentiment"].iloc[-1]  

            signal = pivot_with_sentiment_signal(high, low, pivot, close, s1, s2, s3, r1, r2, r3)

            if signal != {}:

                can_trade = True
                if signal["Signal"] == "Buy":
                    if datetime.now().hour < 21:
                        take_profit = signal['Resistance']
                        stop_loss = signal['Support']
                        if len(open_position)>0:
                            if len(open_position[open_position['currency']==currency_pair]) == 2: 
                                can_trade = False

                        if can_trade is True:
                            test_row_df = create_prediction_param_data_frame(currency=currency_pair, _open=_open,
                                                                             high=high, low=low, close=close, 
                                                                             sentiment=sentiment,
                                                                             net_long=p2f(net_long),
                                                                             net_short=p2f(net_short),
                                                                             s3=s3, s2=s2, s1=s1, p=pivot, r1=r1, r2=r2,
                                                                             r3=r3, stop=stop_loss,
                                                                             limit=take_profit,
                                                                             order_type=signal["Signal"], win_loss=0,
                                                                             volume=vol, date=date, day=day, hour=hour,
                                                                             minute=minute, stoch_k=stoch_k,
                                                                             stoch_d=stoch_d,
                                                                             sma_slow=sma_slow, sma_fast=sma_fast,
                                                                             renko_uptrend=uptrend,
                                                                             renko_bar_num=renko_bar_num,
                                                                             macd=mac, obv=obv, atr=atr, adx=adx)

                            combined_data_frame = add_to_imported_data_frame(test_row_df)

                            win_loss_value = regression_model.predict(combined_data_frame)[-1]

                            pos_size = ((risk_percentage * equity) / leverage[currency_pair]) / (
                                (close - signal['Support']) * lot_size_scaler[currency_pair] * 10)

                            pos_size = round(pos_size * k_scaler)

                            if pos_size >= 1 and win_loss_value == 1:
                                token = con.open_trade(symbol=currency_pair, is_buy=True, is_in_pips=False, amount=pos_size,
                                               time_in_force='GTC', stop=stop_loss, limit=take_profit,
                                               trailing_step=False, order_type='AtMarket')

                                if type(token) is fxcmpy.fxcmpy_order:
                                    
                                    currency_trade_id = token.get_tradeId()

                                    write_to_csv(file_name=trades_file_name, currency=currency_pair, _open=_open,
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
                                                 mid_stop=signal['Mid Stop'], mid_stop_passed=False, pos_size=pos_size)

                elif signal["Signal"] == "Sell":
                    take_profit = signal['Support']
                    stop_loss = signal['Resistance']
                    if datetime.now().hour < 21: 
                        if len(open_position)>0:
                            if len(open_position[open_position['currency']==currency_pair]) == 2: 
                                can_trade = False

                        if can_trade is True:
                            test_row_df = create_prediction_param_data_frame(currency=currency_pair, _open=_open,
                                                                             high=high, low=low, close=close, 
                                                                             sentiment=sentiment,
                                                                             net_long=p2f(net_long),
                                                                             net_short=p2f(net_short),
                                                                             s3=s3, s2=s2, s1=s1, p=pivot, r1=r1, r2=r2,
                                                                             r3=r3, stop=stop_loss,
                                                                             limit=take_profit,
                                                                             order_type=signal["Signal"], win_loss=0,
                                                                             volume=vol, date=date, day=day, hour=hour,
                                                                             minute=minute, stoch_k=stoch_k,
                                                                             stoch_d=stoch_d,
                                                                             sma_slow=sma_slow, sma_fast=sma_fast,
                                                                             renko_uptrend=uptrend,
                                                                             renko_bar_num=renko_bar_num,
                                                                             macd=mac, obv=obv, atr=atr, adx=adx)

                            combined_data_frame = add_to_imported_data_frame(test_row_df)

                            win_loss_value = regression_model.predict(combined_data_frame)[-1]

                            pos_size = ((risk_percentage * equity) / leverage[currency_pair]) / (
                                (signal['Resistance'] - close) * lot_size_scaler[currency_pair] * 10)

                            pos_size = round(pos_size * k_scaler)

                            if pos_size >= 1 and win_loss_value == 1:
                                token = con.open_trade(symbol=currency_pair, is_buy=False, is_in_pips=False, amount=pos_size,
                                               time_in_force='GTC', stop=stop_loss, limit=take_profit,
                                               trailing_step=False, order_type='AtMarket')
                                
                                if type(token) is fxcmpy.fxcmpy_order:
                                    
                                    currency_trade_id = token.get_tradeId()

                                    write_to_csv(file_name=trades_file_name, currency=currency_pair, _open=_open,
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
                                                 mid_stop=signal['Mid Stop'], mid_stop_passed=False, pos_size=pos_size)

            iterate_through_open_positions(con, trades_file_name, log_file, close, currency_pair, open_position)


        except Exception as e:
            error_message = "Expected error" + str(sys.exc_info()[0]) + "value:" + str(sys.exc_info()[1]) + str(
                    sys.exc_info()[2])
            print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])
            logger(error_message, log_file)

def pivot_with_sentiment_signal(high, low, pivot, _close, s1, s2, s3, r1, r2, r3):
    s1_pivot = (s1 + pivot) / 2
    s1_s2 = (s1 + s2) / 2
    s2_s3 = (s2 + s3) / 2

    r1_pivot = (r1 + pivot) / 2
    r1_r2 = (r1 + r2) / 2
    r2_r3 = (r2 + r3) / 2

    if low < pivot < _close and s1_pivot < _close < r1_pivot:
        return {"Signal": "Buy", "Support": s1_pivot, "Resistance": r1, "Mid Stop": r1_pivot}
    if low < s1 < _close and s1_s2 < _close < s1_pivot:
        return {"Signal": "Buy", "Support": s1_s2, "Resistance": pivot, "Mid Stop": s1_pivot}
    if low < s2 < _close and s2_s3 < _close < s1_s2:
        return {"Signal": "Buy", "Support": s2_s3, "Resistance": s1, "Mid Stop": s1_s2}
    if high > pivot > _close and s1_pivot < _close < r1_pivot:
        return {"Signal": "Sell", "Support": s1, "Resistance": r1_pivot, "Mid Stop": s1_pivot}
    if high > r1 > _close and r1_pivot < _close < r1_r2:
        return {"Signal": "Sell", "Support": pivot, "Resistance": r1_r2, "Mid Stop": r1_pivot}
    if high > r2 > _close and r1_r2 < _close < r2_r3:
        return {"Signal": "Sell", "Support": r1, "Resistance": r2_r3, "Mid Stop": r1_r2}

    return {}

trained = False
def machine_learning():
    random_number_0 = randint(0, 50)
    random_number = randint(0, 50)
    imported_dataframe = pd.read_csv(output_file_name, index_col='Date')
    imported_dataframe.drop('TradeId', axis=1, inplace=True)
    buy_sell_dummies = get_dummies(imported_dataframe)
    X = buy_sell_dummies.drop('Win/Loss', axis=1)
    Y = buy_sell_dummies['Win/Loss']
    X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.95)
    r_model = RandomForestClassifier()
    r_model.fit(X_train, Y_train)
#     %pylab inline
#     pylab.rcParams['figure.figsize']=(15,6)
#     plt.plot(y_pred, label='Predicted')
#     plt.plot(y_test.values,, label='Actual')
#     plt.ylabel('Win/lose')
#     plt.legend()
#     plt.show()
    
    return r_model


regression_model = tree.DecisionTreeClassifier()

def create_prediction_param_data_frame(currency, _open, high, low, close, sentiment, net_long,
                                       net_short, s3, s2, s1, p, r1, r2, r3, stop, limit, order_type,
                                       win_loss, volume, date, day, hour, minute, stoch_k, stoch_d, sma_slow, sma_fast,
                                       renko_uptrend,
                                       renko_bar_num, macd, obv, atr, adx):
    data = {0: [date, currency, day, hour, minute, _open, high, low, close, volume,
                sentiment, net_long, net_short, s3, s2, s1, p, r1, r2, r3, stop, limit,
                stoch_k, stoch_d, sma_slow, sma_fast, renko_uptrend,
                renko_bar_num, macd, obv, atr, adx, order_type, win_loss]}

    df = pd.DataFrame.from_dict(data, orient='index', columns=['Date', "Currency",
                                                               "Day", "Hour", "Minute", "Open",
                                                               "High", "Low", "Close", "Volume",
                                                               "Sentiment", "Net Long", "Net Short",
                                                               "S3", "S2", "S1", "P", "R1", "R2", "R3",
                                                               "Stop_loss", "Take_Profit",
                                                               "Stochastic_K", "Stochastic_D", "SMA_slow", "SMA_fast",
                                                               "Renko_Uptrend",
                                                               "Renko_bar_num", "MACD", "OBV", "ATR", "ADX", "Buy/Sell",
                                                               "Win/Loss"])
    df = df.set_index('Date')
    return df


def add_to_imported_data_frame(predicting_param_data_frame):
    read_df = pd.read_csv(output_file_name, index_col='Date')
    read_df.drop('TradeId', axis=1, inplace=True)
    joined_df = pd.concat([read_df, predicting_param_data_frame])
    joined_df = get_dummies(joined_df)
    X = joined_df.drop('Win/Loss', axis=1)
    return X


def get_dummies(df):
#     return pd.get_dummies(df, columns=['Currency', 'Day', 'Sentiment', 'Buy/Sell',
#                                        'Renko_Uptrend', 'Resistance1Strength', 'Resistance2Strength', 
#                                       'Resistance3Strength', 'Support1Strength', 'Support2Strength', 
#                                       'Support3Strength', 'Trend'])
    return pd.get_dummies(df, columns=['Currency', 'Day', 'Sentiment', 'Buy/Sell',
                                       'Renko_Uptrend'])


# In[ ]:

first_iteration = True
while True:
    if datetime.today().minute is 15 or datetime.today().minute is 30 or datetime.today().minute is 45 or datetime.today().minute is 0:
        start_time = time.time()
        try:
            message = "pass through at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            logger(message, log_file)
            print(message)
            if datetime.today().weekday() != 5 and datetime.today().weekday() != 6 and datetime.now().hour < 22:
                if not first_iteration:
                    first_iteration = True
                if not trained:
                    regression_model = machine_learning()
                    trained = True
                if datetime.now().hour < 19:
                    if not con.is_connected():
                        con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='real', log_file='real_fxcm.txt')
                    main()
                if datetime.now().hour > 18 and datetime.now().minute >= 45:
                    if len(con.get_open_positions()) > 0:
                        if con.is_connected():
                            con.close_all()

                closed_positions = con.get_closed_positions()   
                iterate_through_closed_positions(con=con, output_file=output_file_dummy, trainer_trades_file=trades_file_name,
                                                 log_file=log_file, closed_positions=closed_positions)
            else:
                trained = False
                if con.is_connected():
                    con.close()
                if first_iteration:
                    write_header(output_file_dummy, field_names)
                    write_header(trades_file_name, field_names)
                    write_logger_header(log_file, 'Real Logger')
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
                    con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server='real', log_file='real_fxcm.txt')
                except:
                    logger("could not connect", log_file)
        time.sleep((60 * 15) - ((time.time() - start_time) % (60 * 15)))


# In[ ]:



