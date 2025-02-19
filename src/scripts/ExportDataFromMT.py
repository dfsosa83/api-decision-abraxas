import MetaTrader5 as mt5
import pandas as pd
import pytz
from datetime import datetime, timedelta

# Set the paths
working_dir = "C:/Users/david/OneDrive/Documents/api-decision-abraxas"

#mt5Path = "C:/Program Files/RoboForexMT5/terminal64.exe"
mt5Path = "C:/Program Files/FPMarkets MT5 Terminal/terminal64.exe"
#symbol = "EURUSD.r" #EURUSD.r USDJPY.r GBPUSD.r USDCHF.r AUDUSD.r NZDUSD.r USDCAD.r
symbol =  "EURUSD" #"USDJPY"  #EURUSD USDJPY GBPUSD USDCHF AUDUSD NZDUSD USDCAD #"USDJPY"
csv_file = working_dir + "/data/Export_" + symbol + "_H1.csv"

# Initialize MetaTrader5
if not mt5.initialize(path=mt5Path):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

if not mt5.login(login=7082362, password="u#wWU#64esZjNVn", server="FPMarketsLLC-Demo"): #standard account
#if not mt5.login(login=7044310, password="75@N@3uYzcYmiz", server="FPMarketsLLC-Demo"): #raw account
    print("login failed, error code: {}".format(mt5.last_error()))
    quit()

timezone = pytz.timezone("Etc/UTC")
from_time = datetime(2019, 1, 2, tzinfo=timezone)
#to_time = datetime(2025, 2, 15, tzinfo=timezone)
to_time = datetime.now(timezone) 

data = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, from_time, to_time)
#print('data: ', data)
if data is None:
    print(f"Failed to retrieve data for {symbol}")
else:
    # Convert the latest data to a DataFrame 
    history_df = pd.DataFrame(data)
    history_df['time'] = pd.to_datetime(history_df['time'], unit='s')
    history_df.to_csv(csv_file, index=False)

# Shut down the MetaTrader5 connection
mt5.shutdown()
