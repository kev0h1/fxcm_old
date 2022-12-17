#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fxcmpy
import time
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import copy
import talib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from talib import MA_Type


# In[2]:


import sys
get_ipython().system('{sys.executable} -m pip install fxcmpy')


# In[5]:


get_ipython().system('{sys.executable} -m pip install statsmodels')


# In[9]:


get_ipython().system('{sys.executable} -m pip install stocktrends')


# In[10]:


import fxcmpy
import time
import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import copy


# In[11]:


TOKEN = "912ffdbeaec31419ef155a6cdc666d4dc28fb69c"


# In[12]:


con = fxcmpy.fxcmpy(access_token = TOKEN, log_level = 'error')


# In[13]:


pairs =['EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD']
pos_size = 10


# In[15]:


def main():
    try:
        open_pos = con.get_open_positions()
        for currency in pairs:
            data = con.get_candles(currency, period='m5', number=250)
            ohlc = data.iloc[:,[0,1,2,3,8]]
            ohlc.columns = ["Open","Adj Close","High","Low","Volume"]
            signal = np.random.randint(0,3)
            if(signal == 1):
                con.close_all_for_symbol(currency)
                con.open_trade(symbol=currency, is_buy=True, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                
                print("New long position initiated for ", currency)
            elif (signal ==2):
                con.close_all_for_symbol(currency)
                con.open_trade(symbol=currency, is_buy=False, is_in_pips=True, amount=pos_size, 
                               time_in_force='GTC', stop=-8, trailing_step =True, order_type='AtMarket')
                print("New short position initiated for ", currency)
    except:
        print("error encountered....skipping this iteration")


# In[ ]:


starttime=time.time()
timeout = time.time() + 60*60*24  # 60 seconds times 60 meaning the script will run for 1 hr
while time.time() <= timeout:
    try:
        print("passthrough at ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        main()
        time.sleep((60*2) - ((time.time() - starttime) % (60*2))) # 5 minute interval between each new execution
    except KeyboardInterrupt:
        print('\n\nKeyboard exception received. Exiting.')
        exit()


# In[ ]:




