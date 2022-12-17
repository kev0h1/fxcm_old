
# coding: utf-8

# In[1]:

from algorithm_libraries import *


# In[ ]:

log_file = "log_beautiful.txt"


# In[ ]:

count = 0
start_time = time.time()
first_iteration = True
while True:
    try:
        robot_message = "pass through at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + "\r"
        print(robot_message)
        logger(robot_message, log_file)
        if datetime.today().weekday() != 5 and datetime.today().weekday() != 6 and datetime.now().hour < 22:
            save_net_buy_sell_to_file(get_net_buy_sell())
            save_pivot_table_to_file(pairs_refined)
            save_support_and_resistance_to_file(pairs_refined)
            save_trend_to_file(get_trend())
        
        else:
            if datetime.now().hour == 0:
                first_iteration = True
            if first_iteration:
                write_logger_header(log_file, 'New Beautiful logger')
                first_iteration = False
        time.sleep((60 * 60) - ((time.time() - start_time) % (60 * 60)))
    except:
        print("Expected error", sys.exc_info()[0], "value:", sys.exc_info()[1], sys.exc_info()[2])


# In[ ]:



