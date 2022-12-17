#!/usr/bin/env python
# coding: utf-8

# In[1]:


from algorithm_libraries import *
from finta import TA
from random import randint
import fxcmpy
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import talib


# In[2]:


#D25749100 - 7002

TOKEN="207f1613f32cd6003c154df38990814e0c08fa2f"

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')


# In[14]:


class Equity:
    def __init__(self):
        self.equity = 500
        self.total_margin = 0
        
class Order:
    def __init__(self, currency_pair, order_type, entry_price, stop_loss, take_profit, lotsize, spread, margin, sl_pips, market_conditions):
        self.currency_pair = currency_pair
        self.order_type = order_type
        self.entry_price = entry_price
        self.close_price = 0
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.lotsize = lotsize
        self.trade_id = randint(1000000, 1019999)
        self.pip_value_difference = 0
        self.market_conditions = market_conditions
        self.trade_earnings = 0
        self.win_loss = ""
        self.order_status = "open"
        self.spread = spread
        self.rolling_cost= 0
        self.margin = margin
        self.sl_pips = sl_pips
        
    def order_sent(self):
        print("{} Order {} for {} placed at: {}, with stoploss: {} and take profit: {}".format(self.order_type,
                                                                                                self.trade_id,
                                                                                            self.currency_pair,
                                                                                            self.entry_price,
                                                                                            self.stop_loss, 
                                                                                          self.take_profit))
        
class MarketConditions:
    def __init__(self, close, _open, vol, day, date, _time, hour,
                minute, stoch_k, stoch_d, sma_slow, sma_fast, macd, signal,
                obv, atr, adx, dochian_low, dochian_up, high, low, obv_angle,
                bb_upper, bb_middle, bb_lower, chaikin, adl, adl_angle, tenkan, 
                senkou_a, senkou, kijun, chikou, candle, candle_value, candle_sentiment):
        self.close=close
        self._open= _open
        self.vol = vol
        self.day = day
        self.date = date
        self._time = _time 
        self.hour = hour
        self.minute = minute
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.sma_slow = sma_slow
        self.sma_fast = sma_fast
        self.macd = macd
        self.signal = signal
        self.obv = obv
        self.obv_angle = obv_angle
        self.atr = atr 
        self.adx = adx
        self.dochian_low = dochian_low 
        self.dochian_up = dochian_up
        self.high = high
        self.low = low
        self.bb_upper = bb_upper
        self.bb_middle = bb_middle
        self.bb_lower = bb_lower
        self.chaikin = chaikin
        self.adl = adl
        self.adl_angle = adl_angle
        self.tenkan = tenkan
        self.senkou_a = senkou_a
        self.senkou = senkou
        self.kijun = kijun
        self.chikou = chikou
        self.candle = candle
        self.candle_value = candle_value
        self.candle_sentiment = candle_sentiment

def calculate_lotsize(risk_percentage, stoploss_in_pips, equity):
    return (((risk_percentage/100)* equity.equity)/(stoploss_in_pips*10))

def create_market_conditions(ohlc, ohlc_stoch, ohlc_stochd, ohlc_ma_slow, ohlc_ma_fast,
                                                 ohlc_mac, ohlc_obv_df, ohlc_atr, 
                                                 ohlc_adx, chaikin, bbBands, adl,
                                                 ichi, pivots, dochian_low, dochian_up, index, candle, candle_value, candle_sentiment):
    close = ohlc.iloc[index]['Adj Close']
    _open = ohlc.iloc[index]['Open']
    high = ohlc.iloc[index]['High']
    low = high = ohlc.iloc[index]['Low']
    vol = ohlc.iloc[index]['Volume']
    day = ohlc.iloc[index].name.to_pydatetime().strftime("%A")
    date = ohlc.iloc[index].name.to_pydatetime().strftime("%d %b %Y")
    _time = ohlc.iloc[index].name.to_pydatetime().strftime("%I:%M%p")
    hour = ohlc.iloc[index].name.hour
    minute = ohlc.iloc[index].name.minute   
    stoch_k = ohlc_stoch[index]
    stoch_d = ohlc_stochd[index]  
    sma_slow = ohlc_ma_slow[index]
    sma_fast =ohlc_ma_fast[index]
    macd = ohlc_mac['MACD'][index]
    signal = ohlc_mac['SIGNAL'][index]
    obv = ohlc_obv_df['OBV'][index]
    obv_angle = ohlc_obv_df['angle'][index]
    atr = ohlc_atr[index]
    adx = ohlc_adx[index]
    chaikin_value = chaikin[index]
    bb_upper = bbBands["BB_UPPER"][index]
    bb_middle = bbBands["BB_MIDDLE"][index]
    bb_lower= bbBands["BB_LOWER"][index]
    adl_value = adl['ADL'][index]
    adl_angle = adl['angle'][index]
    tenkan = ichi["TENKAN"][index]
    kijun = ichi["KIJUN"][index]
    senkou_a = ichi["senkou_span_a"][index]
    senkou = ichi["SENKOU"][index]
    chikou = ichi["CHIKOU"][index]
#     p = pivots["pivot"][-1]
#     s1 = pivots["s1"][-1]
#     s2 = pivots["s2"][-1]
#     s3 = pivots["s3"][-1]
#     s4 = pivots["s4"][-1]
#     r1 = pivots["r1"][-1]
#     r2 = pivots["r2"][-1]
#     r3 = pivots["r3"][-1]
#     r4 = pivots["r4"][-1]
    
    
    return MarketConditions(close, _open, vol, day, date, _time, hour,
                minute, stoch_k, stoch_d, sma_slow, sma_fast, macd, signal,
                obv, atr, adx, dochian_low, dochian_up, high, low,
                           obv_angle,
                bb_upper, bb_middle, bb_lower, chaikin_value, adl_value, adl_angle, tenkan, 
                senkou_a, senkou, kijun, chikou, candle, candle_value, candle_sentiment)

def get_pip_value(lot_size, currency_pair):
    return 0.0001 * lot_size * 100000
    
def dochian_channel_formal(ohlc):
    df = ohlc.copy()
    df.columns = ["open", "close", "high", "low", "Ask close", "volume", "weekday"]
    return df

def create_market_order(order_type, currency_pair, data, ohlc, ohlc_stoch, ohlc_stochd, ohlc_ma_slow, ohlc_ma_fast, ohlc_mac, ohlc_obv_df, ohlc_atr, ohlc_adx, 
                        chaikin, bbBands, adl, ichi, pivots, dochian_row_lower, dochian_row_upper, i, equity, stop, limit,
                        candle, candle_value, candle_sentiment):
    market_conditions = create_market_conditions(ohlc=ohlc, ohlc_stoch=ohlc_stoch, ohlc_stochd=ohlc_stochd, ohlc_ma_slow=ohlc_ma_slow, ohlc_ma_fast=ohlc_ma_fast,
                                                 ohlc_mac=ohlc_mac, ohlc_obv_df=ohlc_obv_df, ohlc_atr=ohlc_atr, 
                                                 ohlc_adx=ohlc_adx, chaikin=chaikin, bbBands=bbBands, adl=adl,
                                                 ichi=ichi, pivots=pivots, dochian_low=dochian_row_lower, 
                                                 dochian_up=dochian_row_upper, index=i, candle=candle, candle_value=candle_value,
                                                 candle_sentiment=candle_sentiment)
    if order_type == "Buy":
        stop_loss = data["Adj Close"] - 0.5 * market_conditions.atr
#         stop_loss = stop
        take_profit = data["Adj Close"] + 2 * market_conditions.atr
    else:      
        stop_loss = data["Adj Close"] + 0.5 * market_conditions.atr
#         stop_loss = stop
        take_profit = data["Adj Close"] - 2* market_conditions.atr
        
    spread = abs(data["Adj Close"] - data["Ask Close"])
    
    if "JPY" in currency_pair:
        scaler = 100
    else:
        scaler = 10000
        
    sl_pips = abs(stop_loss - data["Adj Close"]) * scaler
    lotsize = calculate_lotsize(1, sl_pips, equity)
    margin = data["Adj Close"] * lotsize * scaler * leverage
    
    order = Order(currency_pair=currency_pair, order_type=order_type, entry_price=data["Adj Close"], 
                  stop_loss=stop_loss, take_profit=take_profit, lotsize=lotsize, 
                  spread=spread, margin=margin, sl_pips=sl_pips, market_conditions=market_conditions)
    return order
    
    
def close_trade_loop(close_price, order_type, equity, date):
    for trade in trades:
        close_trade(trade,close_price, order_type, equity, date)
            
def close_trade(trade,close_price, order_type, equity, date):
    if trade.order_status == "open" and trade.order_type == order_type:
        trade.close_price = close_price
        if "JPY" in trade.currency_pair:
            scaler = 100
        else:
            scaler = 10000
            
        if order_type == "Buy":
            if trade.entry_price > trade.close_price:
                trade.pip_value_difference = -abs(trade.entry_price - trade.close_price) * scaler
            if trade.entry_price < trade.close_price:
                trade.pip_value_difference = abs(trade.entry_price - trade.close_price) * scaler
                
        if order_type == "Sell":
            if trade.entry_price > trade.close_price:
                trade.pip_value_difference = abs(trade.entry_price - trade.close_price) * scaler
            if trade.entry_price < trade.close_price:
                trade.pip_value_difference = -abs(trade.entry_price - trade.close_price) * scaler
                
        trade.trade_earnings = trade.pip_value_difference * get_pip_value(trade.lotsize, currency_pair)
        equity.equity += trade.trade_earnings
        equity.total_margin = equity.total_margin - trade.margin
        trade.order_status = "closed"
        closed_trades.append(trade)
        times.append(date)
        earnings.append(equity.equity)
        
def get_market_data(currency_pair, per, candle_number, con):
    data = con.get_candles(currency_pair, period=per, number=candle_number)
    ohlc = data.iloc[:, [0, 1, 2, 3, 5, 8]]
    ohlc.columns = ["Open", "Adj Close", "High", "Low", "Ask Close", "Volume"]
    ohlc['weekday'] = ohlc.index.dayofweek
    return ohlc

def close_at_sl_or_tp(data, equity):
    for trade in trades:
        if trade.order_type == "Buy":
            if data["Low"] <= trade.stop_loss or data["Open"] <= trade.stop_loss or data["Adj Close"] <= trade.stop_loss or data["High"] <= trade.stop_loss:
                close_trade(trade, trade.stop_loss, "Buy", equity, data.name)
                
            if trade.take_profit != 0:
                if data["High"] >= trade.take_profit or data["Open"] >= trade.take_profit or data["Adj Close"] >= trade.take_profit or data["Low"] >= trade.take_profit:
                    close_trade(trade, trade.take_profit, "Buy", equity, data.name)

                
        elif trade.order_type == "Sell":
            if data["High"] >= trade.stop_loss or data["Low"] >= trade.stop_loss or data["Adj Close"] >= trade.stop_loss or data["Open"] >= trade.stop_loss:
                close_trade(trade, trade.stop_loss, "Sell", equity, data.name)

              
            if trade.take_profit != 0:
                if data["Low"] <= trade.take_profit or data["High"] <= trade.take_profit or data["Adj Close"] <= trade.take_profit or data["Open"] <= trade.take_profit:
                    close_trade(trade, trade.take_profit, "Sell", equity, data.name)


# In[4]:


field_names_2 = ['Date', 'Currency', 'Day', 'Hour', 'Minute', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Stop_loss', 'Take_Profit', 'Stochastic_K', 'Stochastic_D', 'SMA_slow',
                'SMA_fast', 'MACD', 'Signal', 'OBV', 'OBV Angle', 'ATR', 'ADX',"Dochian Upper",
                 "Dochian Lower", 'Chaikin', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'ADL', 'ADL ANGLE',
                 'TENKAN', 'KIJUN', 'senkou_span_a', 'SENKOU', 'CHIKOU', 'Candle', 'Candle Value', 'Candle Sentiment', 'Buy/Sell',  'Win/Loss']

def write_trade_to_file(trade, file_name):
    if trade.trade_earnings > 0:
        win_loss = 1
    else:
        win_loss = 0
    with open(file_name, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names_2)
            writer.writerow({"Date": trade.market_conditions.date,
                             "Currency": trade.currency_pair,
                             "Day": trade.market_conditions.day,
                             "Hour": trade.market_conditions.hour,
                             "Minute": trade.market_conditions.minute,
                             "TradeId": trade.trade_id,
                             "Open": trade.market_conditions._open,
                             "High": trade.market_conditions.high,
                             "Low": trade.market_conditions.high,
                             "Close": trade.market_conditions.close,
                             "Volume": trade.market_conditions.vol,
                             "Stop_loss": trade.stop_loss,
                             "Take_Profit": trade.take_profit,
                             'Stochastic_K': trade.market_conditions.stoch_k,
                             'Stochastic_D': trade.market_conditions.stoch_d,
                             'SMA_slow': trade.market_conditions.sma_slow,
                             'SMA_fast': trade.market_conditions.sma_fast,
                             'MACD': trade.market_conditions.macd,
                             'Signal':trade.market_conditions.signal,
                             'OBV': trade.market_conditions.obv,
                             'OBV Angle': trade.market_conditions.obv_angle,
                             'ATR': trade.market_conditions.atr,
                             'ADX': trade.market_conditions.adx,
                             'Dochian Upper':trade.market_conditions.dochian_up,
                             'Dochian Lower':trade.market_conditions.dochian_low,
                             'Chaikin':trade.market_conditions.chaikin,
                             'BB_UPPER':trade.market_conditions.bb_upper,
                             'BB_MIDDLE':trade.market_conditions.bb_middle,
                             'BB_LOWER': trade.market_conditions.bb_lower,
                             'ADL': trade.market_conditions.adl,
                             'ADL ANGLE': trade.market_conditions.adl_angle,
                             'TENKAN': trade.market_conditions.tenkan,
                             'KIJUN': trade.market_conditions.kijun,
                             'senkou_span_a': trade.market_conditions.senkou_a,
                             'SENKOU': trade.market_conditions.senkou,
                             'CHIKOU': trade.market_conditions.chikou,
                             'Candle': trade.market_conditions.candle,
                             'Candle Value': trade.market_conditions.candle_value,
                             'Candle Sentiment': trade.market_conditions.candle_sentiment,
                             "Buy/Sell": trade.order_type,
                             "Win/Loss": win_loss
                             })


# In[5]:


currency_pairs_forex = ["USD/CHF", "EUR/USD", "USD/JPY", "USD/CAD", "AUD/JPY", "AUD/CHF", "CHF/JPY", "EUR/AUD",
                        "EUR/JPY", "EUR/NZD",
                        "GBP/CHF", "EUR/GBP", "GBP/CAD", "GBP/JPY", "NZD/USD", "NZD/JPY"]


# In[6]:




leverage = {'EUR/USD': 30, 'USD/JPY': 30, 'AUD/USD': 30, 'GBP/USD': 30, 'USD/CAD': 30,
            'NZD/USD': 30, 'AUD/JPY': 20, 'EUR/CHF': 20, 'EUR/GBP': 20, 'EUR/JPY': 20,
            'GBP/JPY': 20, 'USD/CHF': 30, 'BTC/USD': 2, "ETH/USD": 2, "LTC/USD": 2, 'GER30': 20, 'FRA40': 20,
                   "UK100": 20, "US30": 20, "SPX500":20, "USOil": 20}

# lot_size_scaler = {'EUR/USD': 10000, 'USD/JPY': 100, 'AUD/USD': 10000, 'GBP/USD': 10000, 'USD/CAD': 10000,
#                    'NZD/USD': 10000, 'AUD/JPY': 100, 'EUR/CHF': 10000, 'EUR/GBP': 10000, 'EUR/JPY': 100,
#                    'GBP/JPY': 100, 'USD/CHF': 10000, 'BTC/USD': 1, "ETH/USD": 1, "LTC/USD": 100, 'GER30': 1, 'FRA40': 1,
#                    "UK100": 1, "US30": 1, "SPX500":1, "USOil": 100}


# In[7]:


def get_candle(ohlc, i, doji, dragon_fly_doji, engulfing, hammer, hanging_man, harami, harami_cross,
              inverted_hammer, long_legged_doji, marubozu, shooting_star, spinning_top):
               
    spinning_top_row = spinning_top[spinning_top.index == ohlc.iloc[i].name][-1]
    
    #both
    engulfing_row = engulfing[engulfing.index == ohlc.iloc[i].name][-1] 
    if engulfing_row > 0:
        return ["bullish", engulfing_row, "engulfing"]
    
    if engulfing_row < 0:
        return ["bearish", engulfing_row, "engulfing"]
    
    harami_row = harami[harami.index == ohlc.iloc[i].name][-1]
    if harami_row  > 0:
        return ["bullish", harami_row, "harami"]
    
    if harami_row < 0:
        return ["bearish", harami_row, "harami"]
    
    harami_cross_row = harami_cross[harami_cross.index == ohlc.iloc[i].name][-1]   
    if harami_cross_row  > 0:
        return ["bullish", harami_cross_row, "harami_cross"]
    
    if harami_cross_row < 0:
        return ["bearish", harami_cross_row, "harami_cross"]
    
    marubozu_row  = marubozu[marubozu.index == ohlc.iloc[i].name][-1]
    if marubozu_row  > 0:
        return ["bullish", marubozu_row, "marubozu"]
    
    if marubozu_row < 0:
        return ["bearish", marubozu_row, "marubozu"]
    
    #bullish
    dragon_fly_doji_row = dragon_fly_doji[dragon_fly_doji.index == ohlc.iloc[i].name][-1]
    if dragon_fly_doji_row != 0:
        return ["bullish", dragon_fly_doji_row, "dragon_fly_doji"]
    
    hammer_row = hammer[hammer.index == ohlc.iloc[i].name][-1]
    if hammer_row != 0:
        return ["bullish", hammer_row, "hammer"]
    
    inverted_hammer_row = inverted_hammer[inverted_hammer.index == ohlc.iloc[i].name][-1]
    if inverted_hammer_row != 0:
        return ["bullish", inverted_hammer_row, "inverted_hammer"]
    
    #bearish
    shooting_star_row = shooting_star[shooting_star.index == ohlc.iloc[i].name][-1] 
    if shooting_star_row != 0:
        return ["bearish", shooting_star_row, "shooting_star"]
    
    hanging_man_row = hanging_man[hanging_man.index == ohlc.iloc[i].name][-1] 
    if hanging_man_row != 0:
        return ["bearish", hanging_man_row, "hanging_man"]
    
    #if previous bearish then bullish
    doji_row = doji[doji.index == ohlc.iloc[i].name][-1]
    if doji_row != 0:
        return ["bullish", doji_row, "doji"]
    
    #if previous bullish then beasrish
    long_legged_doji_row = long_legged_doji[long_legged_doji.index == ohlc.iloc[i].name][-1]
    if long_legged_doji_row != 0:
        return ["bearish", long_legged_doji_row, "long_legged_doji"]
    
    return []


# In[17]:


equity = Equity()

trades = []
closed_trades = []
times = []
earnings = []
leverage = 1 / 30

# for currency_pair in currency_pairs_forex:
currency_pair = "GBP/USD"

ohlc = pd.DataFrame()
ohlc_h = pd.DataFrame()
ohlc_d = pd.DataFrame()

# if datetime.now().weekday() == 5 or datetime.now().weekday() == 6:
# ohlc = pd.read_csv("EURUSD.txt", parse_dates=['date'], index_col="date") 
# ohlc_h = pd.read_csv("EURUSD_hour.txt", parse_dates=['date'], index_col="date") 
# ohlc_d = pd.read_csv("EURUSD_day.txt", parse_dates=['date'],  index_col="date") 
    
# else:    
ohlc = get_market_data(currency_pair, "m15", 10000, con)
ohlc_h = get_market_data(currency_pair, "H1", 10000, con)
ohlc_d = get_market_data(currency_pair, "D1", 10000, con)
    
d1 = dochian_channel_formal(ohlc_d)
d2 = dochian_channel_formal(ohlc)
d1_h =  dochian_channel_formal(ohlc_h)

t_open = ohlc["Open"]
t_high = ohlc["High"]
t_low = ohlc["Low"]
t_close = ohlc["Adj Close"]

doji = talib.CDLDOJI(t_open, t_high, t_low, t_close)
dragon_fly_doji = talib.CDLDRAGONFLYDOJI(t_open, t_high, t_low, t_close)
engulfing = talib.CDLENGULFING(t_open, t_high, t_low, t_close)
hammer = talib.CDLHAMMER(t_open, t_high, t_low, t_close)
hanging_man = talib.CDLHANGINGMAN(t_open, t_high, t_low, t_close)
harami = talib.CDLHARAMI(t_open, t_high, t_low, t_close)
harami_cross = talib.CDLHARAMICROSS(t_open, t_high, t_low, t_close)
inverted_hammer = talib.CDLINVERTEDHAMMER(t_open, t_high, t_low, t_close)
long_legged_doji = talib.CDLLONGLEGGEDDOJI(t_open, t_high, t_low, t_close)
marubozu = talib.CDLMARUBOZU(t_open, t_high, t_low, t_close)
shooting_star = talib.CDLSHOOTINGSTAR(t_open, t_high, t_low, t_close)
spinning_top = talib.CDLSPINNINGTOP(t_open, t_high, t_low, t_close)

pivots = TA.SMA(d1, 50)
ma_df = pd.DataFrame({'index':pivots.index, 'MA':pivots.values})
ma_df = ma_df.set_index('index')
m_x = slope(ma_df["MA"], 5)
ma_df['angle'] = m_x
#     pivots.dropna(inplace=True)

hourly_ma = TA.SMA(d1_h, 50)
mah_df = pd.DataFrame({'index':hourly_ma.index, 'MA':hourly_ma.values})
mah_df = mah_df.set_index('index')
mh_x = slope(mah_df["MA"], 5)
mah_df['angle'] = mh_x
#     pivots.dropna(inplace=True)

ohlc_stoch = TA.STOCH(d2)
ohlc_stochd = TA.STOCHD(d2)
ohlc_ma_slow = TA.SMA(d2,100)
ohlc_ma_fast = TA.SMA(d2,50)
ohlc_mac = TA.MACD(d2)
ohlc_obv = TA.OBV(d2)
ohlc_obv_df = pd.DataFrame({'index':ohlc_obv.index, 'OBV':ohlc_obv.values})
ohlc_obv_df['rolling'] = ohlc_obv_df['OBV'].rolling(20).mean()
ohlc_obv_df = ohlc_obv_df.set_index('index')
x = slope(ohlc_obv_df["OBV"], 5)
ohlc_obv_df['angle'] = x
ohlc_atr = TA.ATR(d2)
ohlc_adx = TA.ADX(d2)
chaikin = TA.CHAIKIN(d2)
bbBands = TA.BBANDS(d2)

adl_df=TA.ADL(d2)           
adl = pd.DataFrame({'index':adl_df.index, 'ADL':adl_df.values})
adl = adl.set_index('index')
x = slope(adl["ADL"], 5)
adl['angle'] = x

ichi = TA.ICHIMOKU(d2)

period = 200
dochian_channel = TA.DO(d2, period)
dochian_channel.dropna(inplace=True)
rsi = TA.RSI(d2)
rsi.dropna(inplace=True)
dochian_channel['LOWER MIN'] = dochian_channel['LOWER'].rolling(48).min()
dochian_channel['UPPER MAX'] = dochian_channel['UPPER'].rolling(48).max()

dochian_channel_exit = TA.DO(d2, 20)
dochian_channel_exit.dropna(inplace=True)
i = 0
previous_fast_ma = np.nan
previous_slow_ma = np.nan

for index, data in ohlc.iterrows():
    
    if i == 0:
        i+=1
        continue
        
    candle_signal = get_candle(ohlc, i, doji, dragon_fly_doji, engulfing, hammer, hanging_man, harami, harami_cross,
              inverted_hammer, long_legged_doji, marubozu, shooting_star, spinning_top)
   
    ma_df_hour_row = ma_df[(ma_df.index.day == ohlc.iloc[i].name.day) & (ma_df.index.month == ohlc.iloc[i].name.month)& (ma_df.index.year == ohlc.iloc[i].name.year)]

    ma_df_row = mah_df[(mah_df.index.day == ohlc.iloc[i].name.day) & (mah_df.index.month == ohlc.iloc[i].name.month)& (mah_df.index.year == ohlc.iloc[i].name.year) & (mah_df.index.hour == ohlc.iloc[i].name.hour)]
    
    fast_ma_row_p = ohlc_ma_fast[ohlc_ma_fast.index == ohlc.iloc[i-1].name]
    
    slow_ma_row_p = ohlc_ma_slow[ohlc_ma_slow.index == ohlc.iloc[i-1].name]
    
    fast_ma_row = ohlc_ma_fast[ohlc_ma_fast.index == ohlc.iloc[i].name]
    
    slow_ma_row = ohlc_ma_slow[ohlc_ma_slow.index == ohlc.iloc[i].name]

    dochian_row = dochian_channel[dochian_channel.index == ohlc.iloc[i].name]
    
    previous_data = ohlc[ohlc.index == ohlc.iloc[i-1].name]

#     pivot_row = ma_df[ma_df.index.date == ohlc.iloc[i].name.date()]

    dochian_row_exit = dochian_channel_exit[dochian_channel_exit.index == ohlc.iloc[i].name]

    close_at_sl_or_tp(data, equity)   

    hour = data.name.hour

#     if len(dochian_row_exit)>0:
#         dochian_row_lower_exit = dochian_row_exit["LOWER"][-1]
#         dochian_row_upper_exit = dochian_row_exit["UPPER"][-1]

#         if data["Low"] <= dochian_row_lower_exit:
#             ### close buy
#             close_trade_loop(data["Low"], "Buy", equity, data.name)                    

#         elif data["High"] >= dochian_row_upper_exit:
#             ### close sell
#             close_trade_loop(data["High"], "Sell", equity, data.name)

#     trades = [x for x in trades if x not in closed_trades]
    
    
    if len(candle_signal) ==0:
        i+=1
        continue
        
    if len(ma_df_hour_row)>0 and len(ma_df_row)>0:
        if ma_df_hour_row['angle'][-1] != np.nan and ma_df_row['angle'][-1] != np.nan:    
            if ma_df_hour_row['angle'][-1] < -10 and ma_df_row['angle'][-1] < -10 and candle_signal[0] == "bearish":
                if candle_signal[2] == "long_legged_doji":
                    if previous_data["Open"][-1] > previous_data["Adj Close"][-1] :
                        i+=1
                        continue

                order = create_market_order("Sell", currency_pair, data, ohlc, ohlc_stoch, ohlc_stochd, ohlc_ma_slow, ohlc_ma_fast, 
                            ohlc_mac, ohlc_obv_df, ohlc_atr, ohlc_adx, chaikin, bbBands, adl, ichi, pd.DataFrame({'A' : []}), 
                            0, 0, i, equity, 0, 0, candle_signal[2], candle_signal[1], candle_signal[0])
                if equity.equity - (equity.total_margin + order.margin) > 0 and order.market_conditions.obv_angle>20:
                    trades.append(order)
                    equity.total_margin += order.margin
            elif ma_df_hour_row['angle'][-1] > 10 and ma_df_row['angle'][-1] > 10 and candle_signal[0] == "bullish":
                if candle_signal[2] == "doji":
                    if previous_data["Open"][-1]  < previous_data["Adj Close"][-1] :
                        i+=1
                        continue

                order = create_market_order("Buy", currency_pair, data, ohlc, ohlc_stoch, ohlc_stochd, ohlc_ma_slow, ohlc_ma_fast, 
                            ohlc_mac, ohlc_obv_df, ohlc_atr, ohlc_adx, chaikin, bbBands, adl, ichi, pd.DataFrame({'A' : []}), 
                            0, 0, i, equity, 0, 0, candle_signal[2], candle_signal[1], candle_signal[0])
                if equity.equity - (equity.total_margin + order.margin) > 0 and order.market_conditions.obv_angle<-20:
                    trades.append(order)
                    equity.total_margin += order.margin

    i+=1
trades = []       

# write_header("mcs.csv",field_names_2)
# for trade in closed_trades:
#     write_trade_to_file(trade, "mcs.csv")

# plt.plot(times, earnings)        
fig = go.Figure(data=go.Scatter(x=times, y=earnings))
fig.show()


# In[ ]:


# plt.plot(times, earnings)        
fig = go.Figure(data=go.Scatter(x=times, y=earnings))
fig.show()


# In[ ]:


equity.equity


# In[ ]:


write_header("mcs.csv",field_names_2)
for trade in closed_trades:
    write_trade_to_file(trade, "mcs.csv")


# In[ ]:


equity.total_margin


# In[ ]:


# USD/CHF g G - 2620 2404 1069 y 1094 y
#EUR/USD N N 736 y 
# USD/JPY n N 767 y
# USD/CAD g G - 722 773 y
# AUD/JPY n N - 594 779 y
# AUD/CHF n N - 692 946 y
# "CAD/CHF" n N - 664 631 n
# CAD/JPY n N - 657 614 n
# CHF/JPY y
# EUR/AUD g G - 650 731 y
# EUR/JPY n G 650 451 y
# EUR/NZD - g G y
# GBP/CHF n N - 3575 889 y
#EUR/GBP N y
# GBP/CAD n y
# GBP/USD N N 1642 1041 n
# GBP/JPY N N 1323 846 y
# NZD/USD n N 1786 1329 y
# NZD/JPY N N 1791 963 


# In[ ]:


currency_pairs_forex = ["USD/CHF", "EUR/USD", "USD/JPY", "USD/CAD", "AUD/JPY", "AUD/CHF", "CHF/JPY", "EUR/AUD", "EUR/JPY", "EUR/NZD",
                       "GBP/CHF", "EUR/GBP", "GBP/CAD", "GBP/JPY", "NZD/USD", "NZD/JPY"]


# In[ ]:


pd.set_option('display.max_rows', None)


# In[ ]:


ohlc = pd.read_csv("EURUSD.txt", index_col="date")


# In[ ]:


datetime.now().weekday()


# In[ ]:


ma_df_hour_row = ma_df[(ma_df.index.day == ohlc.iloc[i].name.day)]


# In[ ]:


ma_df.iloc[-1].name.day


# In[ ]:


shooting_star_row = shooting_star[shooting_star.index == ohlc.iloc[-1].name][-1] 


# In[ ]:


X = ["HELLO", 1]


# In[ ]:


print(X)


# In[ ]:


engulfing[engulfing.index == ohlc.iloc[-41].name][-1] > 0


# In[ ]:


closed_trades[-2].take_profit


# In[ ]:


# EUR/USD
# "EUR/GBP"
#GBP/USD
#"EUR/AUD"
#"GBP/JPY"


# In[ ]:


"EUR/USD", "EUR/GBP", "GBP/USD", "EUR/AUD", "GBP/JPY"

