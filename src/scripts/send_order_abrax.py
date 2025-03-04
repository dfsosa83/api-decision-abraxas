import MetaTrader5 as mt5

# Initialize the MetaTrader 5 connection
if not mt5.initialize():
    print("Initialization failed")
    mt5.shutdown()

# Define the order parameters
symbol = "EURUSD"
lot = 2.0
order_type = mt5.ORDER_TYPE_BUY
price = mt5.symbol_info_tick(symbol).ask
sl = price 
tp = price 

# prepare the order request
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot,
    "type": order_type,
    "price": price,
    "sl": sl,
    "tp": tp,
    "deviation": 10,
    "magic": 234000,
    "comment": "Python order",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

# send a trading request
result = mt5.order_send(request)

# check the execution result
if result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"Order failed. Error code: {result.retcode}")
else:
    print(f"Order placed successfully. Order ticket: {result.order}")

#close
mt5.shutdown()
