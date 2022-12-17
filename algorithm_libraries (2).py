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
         'GBP/JPY', 'USD/CHF', 'BTC/USD', "ETH/USD", "LTC/USD", 'GER30', "FRA40", "UK100", "US30", "SPX500",
         "USOil"]

pairs_refined = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY',
                 'GBP/JPY', 'USD/CHF', "ETH/USD", 'GER30', "FRA40", "UK100", "US30", "SPX500",
                 "USOil"]

pairs_no_indices = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'NZD/USD', 'AUD/JPY', 'EUR/CHF', 'EUR/GBP',
                    'EUR/JPY',
                    'GBP/JPY', 'USD/CHF', 'BTC/USD', "ETH/USD", "LTC/USD"]

pairs_only_forex = ['EUR/USD', 'USD/JPY', 'AUD/USD', 'GBP/USD', 'USD/CAD', 'NZD/USD', 'AUD/JPY', 'EUR/CHF', 'EUR/GBP',
                    'EUR/JPY',
                    'GBP/JPY', 'USD/CHF', 'BTC/USD', "ETH/USD", "LTC/USD"]

alternative_dictionary = {'XAU/USD': 'gold-price', 'XAG/USD': 'silver-prices', 'BTC/USD': 'bitcoin', 'GER30': 'dax-30',
                          'FRA40': 'cac-40', 'ETH/USD': 'ether-eth', 'LTC/USD': 'litecoin-ltc', 'NAS100': 'nas-100',
                          'UK100': 'ftse-100', 'US30': 'dow-jones', 'SPX500': 'sp-500', "USOil": "crude-oil"}

alternative_sentiment_dictionary = {'XAU/USD': 'Gold', 'XAG/USD': 'Silver', 'BTC/USD': 'Bitcoin', 'GER30': 'Germany 30',
                                    'FRA40': 'France 40', 'NAS100': 'nas-100',
                                    'UK100': 'FTSE 100', 'US30': 'Wall Street', 'SPX500': 'US 500',
                                    "USOil": "Oil - US Crude"}

full_sentiment_dictionary = {"BTC/USD": "Bitcoin", "GER30": "Germany 30",
                             "ETH/USD": "Ethereum", "FRA40": "France 40", "XAU/USD": "Gold",
                             "LTC/USD": "Litecoin", "XAG/USD": "Silver", 'NAS100': 'nas-100',
                             'UK100': 'FTSE 100', 'US30': 'Wall Street', 'SPX500': 'US 500', "USOil": "Oil - US Crude"
                             }
full_pivot_table_dictionary = {"BTC/USD": "Bitcoin", "ETH/USD": "Ethereum",
                               "LTC/USD": "Litecoin", "XAU/USD": "Gold", "XAG/USD": "Silver",
                               "GER30": "Germany30", "FRA40": "France40", 'NAS100': 'nas-100',
                               'UK100': 'FTSE 100', 'US30': 'Wall Street', 'SPX500': 'US 500',
                               "USOil": "Oil - US Crude"}

field_names = ['Date', 'Currency', 'Day', 'Hour', 'Minute', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Sentiment',
               'Net Long', 'Net Short', 'S3', 'S2',
               'S1', 'P', 'R1', 'R2', 'R3', 'Stop_loss', 'Take_Profit', 'Stochastic_K', 'Stochastic_D', 'SMA_slow',
               'SMA_fast', 'Renko_Uptrend', 'Renko_bar_num', 'MACD', 'OBV', 'ATR', 'ADX', 'Resistance1', 'Resistance2',
               'Resistance3',
               'Resistance1Strength', 'Resistance2Strength', 'Resistance3Strength', 'Support1', 'Support2', 'Support3',
               'Support1Strength', 'Support2Strength', 'Support3Strength', 'Trend', 'Buy/Sell', 'Win/Loss']

field_names2 = ['Date', 'Currency', 'Day', 'Hour', 'Minute', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Sentiment',
                'Net Long', 'Net Short', 'S3', 'S2',
                'S1', 'P', 'R1', 'R2', 'R3', 'Stop_loss', 'Take_Profit', 'Stochastic_K', 'Stochastic_D', 'SMA_slow',
                'SMA_fast', 'Renko_Uptrend', 'Renko_bar_num', 'MACD', 'OBV', 'ATR', 'ADX', 'Buy/Sell', 'Win/Loss']

field_names3 = ['Date', 'Currency', 'Day', 'Hour', 'Minute', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Sentiment',
                'Net Long', 'Net Short', 'S3', 'S2',
                'S1', 'P', 'R1', 'R2', 'R3', 'Stop_loss', 'Take_Profit', 'Stochastic_K', 'Stochastic_D', 'SMA_slow',
                'SMA_fast', 'Renko_Uptrend', 'Renko_bar_num', 'MACD', 'OBV', 'ATR', 'ADX', 'RSI', 'Buy/Sell',
                'Win/Loss']

field_names4 = ['Date', 'Currency', 'Day', 'Hour', 'Minute', 'TradeId', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Sentiment',
                'Net Long', 'Net Short', 'S3', 'S2',
                'S1', 'P', 'R1', 'R2', 'R3', 'Stop_loss', 'Take_Profit', 'Stochastic_K', 'Stochastic_D', 'SMA_slow',
                'SMA_fast', 'Renko_Uptrend', 'Renko_bar_num', 'MACD', 'OBV', 'ATR', 'ADX', 'Resistance1', 'Resistance2',
                'Resistance3',
                'Resistance1Strength', 'Resistance2Strength', 'Resistance3Strength', 'Support1', 'Support2', 'Support3',
                'Support1Strength', 'Support2Strength', 'Support3Strength', 'Price Min', 'Price Max',
                'Level1', 'Level2', 'Level3', 'Level4', 'Level5', 'Trend', 'Buy/Sell', 'Win/Loss']


def save_support_and_resistance_to_file(pairs_to_iterate):
    f_names = ['Currency', 'R3', 'R2', 'R1', 'S1', 'S2', 'S3', 'R3 strength', 'R2 strength', 'R1 strength',
               'S1 strength', 'S2 strength', 'S3 strength']
    with open('support_and_resistance.txt', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=f_names)
        writer.writeheader()

    for currency_pair in pairs_to_iterate:
        alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD", "UK100", "US30",
                            "SPX500",
                            "USOil"]
        if currency_pair not in alternative_list:
            pivot_name = get_currency_name_for_pivot(currency_pair)
        else:
            pivot_name = alternative_dictionary[currency_pair]

        supports_and_resistances = get_supports_and_resistances(pivot_name)
        if supports_and_resistances != {}:
            with open('support_and_resistance.txt', 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=f_names)
                writer.writerow({"Currency": currency_pair,
                                 "R3": supports_and_resistances["R3"][0],
                                 "R2": supports_and_resistances["R2"][0],
                                 "R1": supports_and_resistances["R1"][0],
                                 "S1": supports_and_resistances["S1"][0],
                                 "S2": supports_and_resistances["S2"][0],
                                 "S3": supports_and_resistances["S3"][0],
                                 "R3 strength": supports_and_resistances["R3"][1],
                                 "R2 strength": supports_and_resistances["R2"][1],
                                 "R1 strength": supports_and_resistances["R1"][1],
                                 "S1 strength": supports_and_resistances["S1"][1],
                                 "S2 strength": supports_and_resistances["S2"][1],
                                 "S3 strength": supports_and_resistances["S3"][1]
                                 })


def save_net_buy_sell_to_file(dictionary):
    f_names = ['Currency', 'Net Long', 'Net Short', 'Sentiment']
    with open('net_buy.txt', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=f_names)
        writer.writeheader()
    for key in dictionary:
        with open('net_buy.txt', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=f_names)
            writer.writerow({"Currency": key,
                             "Net Long": dictionary[key]['Net Long'],
                             "Net Short": dictionary[key]['Net Short'],
                             "Sentiment": dictionary[key]['Sentiment']
                             })


def save_trend_to_file(dictionary):
    f_names = ['Currency', 'Trend']
    with open('trend.txt', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=f_names)
        writer.writeheader()
    for key in dictionary:
        with open('trend.txt', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=f_names)
            writer.writerow({"Currency": key,
                             "Trend": dictionary[key]
                             })


def save_pivot_table_to_file(pairs_to_iterate):
    f_names = ['Currency', 'R3', 'R2', 'R1', 'P', 'S1', 'S2', 'S3']
    with open('pivot_tables.txt', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=f_names)
        writer.writeheader()

    for currency_pair in pairs_to_iterate:
        alternative_list = ['XAU/USD', 'XAG/USD', 'BTC/USD', 'GER30', 'FRA40', "ETH/USD", "LTC/USD", "UK100", "US30",
                            "SPX500",
                            "USOil"]
        if currency_pair not in alternative_list:
            pivot_name = get_currency_name_for_pivot(currency_pair)
        else:
            pivot_name = alternative_dictionary[currency_pair]

        pivot_table = get_daily_pivot_points(pivot_name)
        if pivot_table != {}:
            with open('pivot_tables.txt', 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=f_names)
                writer.writerow({"Currency": currency_pair,
                                 "R3": pivot_table["R3"],
                                 "R2": pivot_table["R2"],
                                 "R1": pivot_table["R1"],
                                 "P": pivot_table["P"],
                                 "S1": pivot_table["S1"],
                                 "S2": pivot_table["S2"],
                                 "S3": pivot_table["S3"],
                                 })


def get_data(currency_pair, per, candle_number, con):
    data = con.get_candles(currency_pair, period=per, number=candle_number)
    ohlc = data.iloc[:, [0, 1, 2, 3, 8]]
    ohlc.columns = ["Open", "Adj Close", "High", "Low", "Volume"]
    ohlc['weekday'] = ohlc.index.dayofweek
    return ohlc


def write_header(file_name, field_type):
    with open(file_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=field_type)
        writer.writeheader()


def write_to_csv(field_name_num, file_name, currency, _open, high, low, close, sentiment, net_long,
                 net_short, s3, s2, s1, p, r1, r2, r3, stop, limit, order_type,
                 win_loss, volume, date, day, hour, minute, trade_id, stoch_k, stoch_d, sma_slow, sma_fast,
                 renko_uptrend,
                 renko_bar_num, macd, obv, atr, adx, resistance_1=None, resistance_1_strength=None, resistance_2=None,
                 resistance_2_strength=None,
                 resistance_3=None, resistance_3_strength=None, support_1=None, support_1_strength=None, support_2=None,
                 support_2_strength=None,
                 support_3=None, support_3_strength=None, trend=None, price_min=None, price_max=None, level1=None,
                 level2=None, level3=None, level4=None, level5=None):
    if field_name_num == 2:
        with open(file_name, 'a', newline='') as f:
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
    elif field_name_num == 4:
        with open(file_name, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names4)
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
                             'Resistance1': resistance_1,
                             'Resistance2': resistance_2,
                             'Resistance3': resistance_3,
                             'Resistance1Strength': resistance_1_strength,
                             'Resistance2Strength': resistance_2_strength,
                             'Resistance3Strength': resistance_3_strength,
                             'Support1': support_1,
                             'Support2': support_2,
                             'Support3': support_3,
                             'Support1Strength': support_1_strength,
                             'Support2Strength': support_2_strength,
                             'Support3Strength': support_3_strength,
                             'Price Min': price_min,
                             'Price Max': price_max,
                             'Level1': level1,
                             'Level2': level2,
                             'Level3': level3,
                             'Level4': level4,
                             'Level5': level5,
                             'Trend': trend,
                             "Buy/Sell": order_type,
                             "Win/Loss": win_loss,
                             })

    else:
        with open(file_name, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
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
                             'Resistance1': resistance_1,
                             'Resistance2': resistance_2,
                             'Resistance3': resistance_3,
                             'Resistance1Strength': resistance_1_strength,
                             'Resistance2Strength': resistance_2_strength,
                             'Resistance3Strength': resistance_3_strength,
                             'Support1': support_1,
                             'Support2': support_2,
                             'Support3': support_3,
                             'Support1Strength': support_1_strength,
                             'Support2Strength': support_2_strength,
                             'Support3Strength': support_3_strength,
                             'Trend': trend,
                             "Buy/Sell": order_type,
                             "Win/Loss": win_loss,
                             })


def write_logger_header(file_name, file_header):
    with open(file_name, 'w', newline='') as f:
        f.write(file_header)


def write_open_positions_to_output_file(output_file, trainer_trades_file, log_file, field_name_num,
                                        open_positions):
    df = pd.read_csv(trainer_trades_file)
    if len(open_positions) > 0:
        try:
            for index, data in df.iterrows():
                t_id = data['TradeId']
                trade_id = open_positions[open_positions['tradeId'] == str(t_id)]
                if len(trade_id) > 0:
                    if trade_id['grossPL'].iloc[-1] > 0:
                        data['Win/Loss'] = 1
                    else:
                        data['Win/Loss'] = 0

                    if field_name_num == 2:
                        write_to_csv(field_name_num, output_file, data['Currency'], data['Open'], data['High'],
                                     data['Low'], data['Close'], data['Sentiment'], data['Net Long'],
                                     data['Net Short'],
                                     data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'],
                                     data['R3'],
                                     data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                     data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                     data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                     data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                     data["OBV"], data["ATR"], data["ADX"])

                    elif field_name_num == 4:
                        write_to_csv(field_name_num, output_file, data['Currency'], data['Open'], data['High'],
                                     data['Low'], data['Close'], data['Sentiment'], data['Net Long'],
                                     data['Net Short'],
                                     data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'],
                                     data['R3'],
                                     data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                     data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                     data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                     data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                     data["OBV"], data["ATR"], data["ADX"], data['Resistance1'],
                                     data['Resistance1Strength'],
                                     data['Resistance2'], data['Resistance2Strength'], data['Resistance3'],
                                     data['Resistance3Strength'], data['Support1'], data['Support1Strength'],
                                     data['Support2'],
                                     data['Support2Strength'], data['Support3'], data['Support3Strength'],
                                     data['Trend'], data['Price Min'], data['Price Max'], data['Level1'],
                                     data['Level2'], data['Level3'],
                                     data['Level4'], data['Level5'])

                    else:
                        write_to_csv(field_name_num, output_file, data['Currency'], data['Open'], data['High'],
                                     data['Low'], data['Close'], data['Sentiment'], data['Net Long'],
                                     data['Net Short'],
                                     data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'],
                                     data['R3'],
                                     data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                     data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                     data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                     data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                     data["OBV"], data["ATR"], data["ADX"], data['Resistance1'],
                                     data['Resistance1Strength'],
                                     data['Resistance2'], data['Resistance2Strength'], data['Resistance3'],
                                     data['Resistance3Strength'], data['Support1'], data['Support1Strength'],
                                     data['Support2'],
                                     data['Support2Strength'], data['Support3'], data['Support3Strength'],
                                     data['Trend'])

            if field_name_num == 2:
                write_header(trainer_trades_file, field_names2)
            elif field_name_num == 4:
                write_header(trainer_trades_file, field_names4)
            else:
                write_header(trainer_trades_file, field_names)

        except:
            message = "From open positions -Expected error" + str(sys.exc_info()[0]) + "value:" + str(
                sys.exc_info()[1]) + str(
                sys.exc_info()[2])
            logger(message, log_file=log_file)


def iterate_through_closed_positions(output_file, trainer_trades_file, log_file, field_name_num, closed_positions):
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

                    if field_name_num == 2:
                        write_to_csv(field_name_num, output_file, data['Currency'], data['Open'], data['High'],
                                     data['Low'], data['Close'], data['Sentiment'], data['Net Long'],
                                     data['Net Short'],
                                     data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'],
                                     data['R3'],
                                     data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                     data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                     data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                     data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                     data["OBV"], data["ATR"], data["ADX"])

                    elif field_name_num == 4:
                        write_to_csv(field_name_num, output_file, data['Currency'], data['Open'], data['High'],
                                     data['Low'], data['Close'], data['Sentiment'], data['Net Long'],
                                     data['Net Short'],
                                     data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'],
                                     data['R3'],
                                     data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                     data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                     data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                     data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                     data["OBV"], data["ATR"], data["ADX"], data['Resistance1'],
                                     data['Resistance1Strength'],
                                     data['Resistance2'], data['Resistance2Strength'], data['Resistance3'],
                                     data['Resistance3Strength'], data['Support1'], data['Support1Strength'],
                                     data['Support2'],
                                     data['Support2Strength'], data['Support3'], data['Support3Strength'],
                                     data['Trend'], data['Price Min'], data['Price Max'], data['Level1'],
                                     data['Level2'], data['Level3'],
                                     data['Level4'], data['Level5'])
                    else:
                        write_to_csv(field_name_num, output_file, data['Currency'], data['Open'], data['High'],
                                     data['Low'], data['Close'], data['Sentiment'], data['Net Long'],
                                     data['Net Short'],
                                     data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'],
                                     data['R3'],
                                     data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                     data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                     data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                     data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                     data["OBV"], data["ATR"], data["ADX"], data['Resistance1'],
                                     data['Resistance1Strength'],
                                     data['Resistance2'], data['Resistance2Strength'], data['Resistance3'],
                                     data['Resistance3Strength'], data['Support1'], data['Support1Strength'],
                                     data['Support2'],
                                     data['Support2Strength'], data['Support3'], data['Support3Strength'],
                                     data['Trend'])

                    index_to_delete = df[df['TradeId'] == t_id].index[-1]
                    df.drop(index_to_delete, inplace=True)

            if field_name_num == 2:
                write_header(trainer_trades_file, field_names2)
            elif field_name_num == 4:
                write_header(trainer_trades_file, field_names4)
            else:
                write_header(trainer_trades_file, field_names)

            for index, data in df.iterrows():
                if field_name_num == 2:
                    write_to_csv(field_name_num, trainer_trades_file, data['Currency'], data['Open'], data['High'],
                                 data['Low'], data['Close'], data['Sentiment'], data['Net Long'], data['Net Short'],
                                 data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'], data['R3'],
                                 data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                 data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                 data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                 data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                 data["OBV"], data["ATR"], data["ADX"])

                elif field_name_num == 4:
                    write_to_csv(field_name_num, trainer_trades_file, data['Currency'], data['Open'], data['High'],
                                 data['Low'], data['Close'], data['Sentiment'], data['Net Long'], data['Net Short'],
                                 data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'], data['R3'],
                                 data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                 data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                 data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                 data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                 data["OBV"], data["ATR"], data["ADX"], data['Resistance1'],
                                 data['Resistance1Strength'],
                                 data['Resistance2'], data['Resistance2Strength'], data['Resistance3'],
                                 data['Resistance3Strength'], data['Support1'], data['Support1Strength'],
                                 data['Support2'],
                                 data['Support2Strength'], data['Support3'], data['Support3Strength'],
                                 data['Trend'], data['Price Min'], data['Price Max'], data['Level1'],
                                 data['Level2'],
                                 data['Level3'],
                                 data['Level4'], data['Level5'])

                else:
                    write_to_csv(field_name_num, trainer_trades_file, data['Currency'], data['Open'], data['High'],
                                 data['Low'], data['Close'], data['Sentiment'], data['Net Long'], data['Net Short'],
                                 data['S3'], data['S2'], data['S1'], data['P'], data['R1'], data['R2'], data['R3'],
                                 data['Stop_loss'], data['Take_Profit'], data['Buy/Sell'], data['Win/Loss'],
                                 data['Volume'], data['Date'], data['Day'], data['Hour'], data['Minute'],
                                 data['TradeId'], data["Stochastic_K"], data["Stochastic_D"], data["SMA_slow"],
                                 data["SMA_fast"], data["Renko_Uptrend"], data["Renko_bar_num"], data["MACD"],
                                 data["OBV"], data["ATR"], data["ADX"], data['Resistance1'],
                                 data['Resistance1Strength'],
                                 data['Resistance2'], data['Resistance2Strength'], data['Resistance3'],
                                 data['Resistance3Strength'], data['Support1'], data['Support1Strength'],
                                 data['Support2'],
                                 data['Support2Strength'], data['Support3'], data['Support3Strength'],
                                 data['Trend'])

        except:
            message = "From closed positions -Expected error" + str(sys.exc_info()[0]) + "value:" + str(
                sys.exc_info()[1]) + str(
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
    try:
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
    except:
        currency_dictionary_full = {}

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
    except:
        return {}


def get_trend():
    currency_sentiment = {}
    try:
        url = "https://www.dailyfx.com/support-resistance"
        page = requests.get(url)
        page_content = page.content
        soup = BeautifulSoup(page_content, 'html.parser')
        currency_information = soup.find_all("div", {
            "class": "dfx-supportResistanceBlock text-black dfx-border--a-1 dfx-supportResistanceBlock--tableViewMd"})
        for info in currency_information:
            currency_sentiment[info.a.string] = get_trend_direction(info.find_all("div", {
                "class": "dfx-supportResistanceBlock__trend"})[0].svg['class'])
    except:
        currency_sentiment = {}

    return currency_sentiment


def get_trend_direction(info_string):
    if 'dfx-signalIcon--neutral' in info_string:
        return 'neutral'
    if 'dfx-signalIcon--up' in info_string:
        return 'up'
    if 'dfx-signalIcon--down' in info_string:
        return 'down'


def get_supports_and_resistances(currency_name):
    try:
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
    except:
        return {}


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
    try:
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
    except:
        return {}


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
