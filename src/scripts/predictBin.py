import os
from datetime import datetime, timedelta
import MetaTrader5 as mt5
import pandas as pd
from flask import Flask #Mono
from flask import request #Mono
import ta
import pytz
import time
import pickle
import numpy as np
from sklearn.metrics import make_scorer, precision_score 
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from keras.models import load_model
import lightgbm as lgb
from xgboost import XGBClassifier
import warnings
import tensorflow as tf
import keras.backend as K
import logging
import asyncio

#TO RUN THIS:flask --app C:/Users/david/OneDrive/Documents/api-decision-abraxas/src/scripts/predictBin run --port 5000
#fpmarkets: u#wWU#64esZjNVn - 51395733

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__) #Mono

# Project name
project_name = "DeafBot"

# Set the paths
working_dir = "C:/Users/david/OneDrive/Documents/api-decision-abraxas"
#mt5Path = "C:/Program Files/RoboForexMT5/terminal64.exe"
mt5Path = "C:/Program Files/FPMarkets MT5 Terminal/terminal64.exe"

# Apply Conformal Prediction
def apply_conformal_prediction(probs, threshold):
    conforms = np.max(probs) >= threshold
    return conforms

# Define a function to preprocess the data
def features_engineering(data):
    # Check if the input is a single row or a DataFrame
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data).T

    # Preserve the original index
    original_index = data.index

    # Reset index to avoid any potential issues with operations
    data = data.reset_index(drop=True)

    #define day,month,year
    data['day'] = data['time'].dt.day
    data['month'] = data['time'].dt.month
    data['year'] = data['time'].dt.year

   # Moving Averages
    data.loc[:, 'MA_10'] = data['close'].rolling(window=10).mean()
    data.loc[:, 'MA_20'] = data['close'].rolling(window=20).mean()
    data.loc[:, 'MA_50'] = data['close'].rolling(window=50).mean()

   # Exponential Moving Averages
    data.loc[:, 'EMA_10'] = data['close'].ewm(span=10, adjust=False).mean()
    data.loc[:, 'EMA_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data.loc[:, 'EMA_50'] = data['close'].ewm(span=50, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data['BB_middle'] = data['close'].rolling(window=20).mean()
    data['BB_std'] = data['close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (2 * data['BB_std'])
    data['BB_lower'] = data['BB_middle'] - (2 * data['BB_std'])

    # Momentum
    data['Momentum'] = data['close'] - data['close'].shift(10)

    # Stochastic Oscillator
    data['Stochastic_K'] = ((data['close'] - data['low'].rolling(window=14).min()) /
                            (data['high'].rolling(window=14).max() - data['low'].rolling(window=14).min())) * 100
    data['Stochastic_D'] = data['Stochastic_K'].rolling(window=3).mean()

    # Lagged Features
    lags = [1, 2, 3, 5, 10]
    for lag in lags:
        data[f'close_lag_{lag}'] = data['close'].shift(lag)
        data[f'volume_lag_{lag}'] = data['tick_volume'].shift(lag)
        data[f'MA_10_lag_{lag}'] = data['MA_10'].shift(lag)
        data[f'RSI_lag_{lag}'] = data['RSI'].shift(lag)
        data[f'BB_middle_lag_{lag}'] = data['BB_middle'].shift(lag)

    # Price Change Features
    data['price_change_1'] = data['close'].pct_change(periods=1)
    data['price_change_5'] = data['close'].pct_change(periods=5)
    data['price_change_10'] = data['close'].pct_change(periods=10)

    # Volatility Features
    data['volatility_10'] = data['close'].rolling(window=10).std()
    data['volatility_20'] = data['close'].rolling(window=20).std()
    data['volatility_50'] = data['close'].rolling(window=50).std()

    # Average True Range (ATR)
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    data['ATR'] = ranges.max(axis=1).rolling(window=14).mean()

    # Williams %R
    highest_high = data['high'].rolling(window=14).max()
    lowest_low = data['low'].rolling(window=14).min()
    data['Williams_%R'] = -100 * ((highest_high - data['close']) / (highest_high - lowest_low))

    # Fibonacci Retracement Levels
    latest_high = data['high'].rolling(window=100).max().shift(1)
    latest_low = data['low'].rolling(window=100).min().shift(1)
    data['Fib_0.236'] = latest_high - (latest_high - latest_low) * 0.236
    data['Fib_0.382'] = latest_high - (latest_high - latest_low) * 0.382
    data['Fib_0.618'] = latest_high - (latest_high - latest_low) * 0.618

    # MACD (Moving Average Convergence Divergence)
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema_12 - ema_26
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_histogram'] = data['MACD'] - data['MACD_signal']

    # On-Balance Volume (OBV)
    data['OBV'] = (np.sign(data['close'].diff()) * data['tick_volume']).fillna(0).cumsum()

    # Commodity Channel Index (CCI)
    tp = (data['high'] + data['low'] + data['close']) / 3
    data['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

    # Rate of Change (ROC)
    data['ROC_10'] = data['close'].pct_change(periods=10) * 100

    # Average Directional Index (ADX)
    def calculate_adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.rolling(period).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (period - 1)) + dx) / period
        adx_smooth = adx.ewm(alpha=1/period).mean()
        return adx_smooth

    data['ADX'] = calculate_adx(data['high'], data['low'], data['close'])

    # Ichimoku Cloud
    def ichimoku_cloud(df, conversion_line_period=9, base_line_period=26, leading_span_b_period=52, lagging_span_period=26):
        high_prices = df['high']
        low_prices = df['low']
        closing_prices = df['close']

        conversion_line = (high_prices.rolling(window=conversion_line_period).max() + low_prices.rolling(window=conversion_line_period).min()) / 2
        base_line = (high_prices.rolling(window=base_line_period).max() + low_prices.rolling(window=base_line_period).min()) / 2
        leading_span_a = ((conversion_line + base_line) / 2).shift(lagging_span_period)
        leading_span_b = ((high_prices.rolling(window=leading_span_b_period).max() + low_prices.rolling(window=leading_span_b_period).min()) / 2).shift(lagging_span_period)
        lagging_span = closing_prices.shift(-lagging_span_period)

        return pd.DataFrame(index=df.index, data={
            'Conversion Line': conversion_line,
            'Base Line': base_line,
            'Leading Span A': leading_span_a,
            'Leading Span B': leading_span_b,
            'Lagging Span': lagging_span
        })

    ichimoku = ichimoku_cloud(data)
    data = pd.concat([data, ichimoku], axis=1)

    # Pivot Points
    data['Pivot'] = (data['high'].shift(1) + data['low'].shift(1) + data['close'].shift(1)) / 3
    data['R1'] = 2 * data['Pivot'] - data['low'].shift(1)
    data['S1'] = 2 * data['Pivot'] - data['high'].shift(1)

    # Volume-based indicators
    data['Volume_MA'] = data['tick_volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['tick_volume'] / data['Volume_MA']

    # Price and volume trend
    data['Price_Up'] = ((data['close'] > data['close'].shift(1)) * 1).rolling(window=14).sum()
    data['Volume_Up'] = ((data['tick_volume'] > data['tick_volume'].shift(1)) * 1).rolling(window=14).sum()


    # Heikin-Ashi Candles
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2

    # Create a DataFrame for max and min calculations
    ha_high_df = pd.concat([data['high'], ha_open, ha_close], axis=1)
    ha_low_df = pd.concat([data['low'], ha_open, ha_close], axis=1)

    ha_high = ha_high_df.max(axis=1)
    ha_low = ha_low_df.min(axis=1)

    data['HA_Close'] = ha_close
    data['HA_Open'] = ha_open
    data['HA_High'] = ha_high
    data['HA_Low'] = ha_low

    # Volume Oscillator
    short_vol_ma = data['tick_volume'].rolling(window=5).mean()
    long_vol_ma = data['tick_volume'].rolling(window=20).mean()
    data['Volume_Oscillator'] = ((short_vol_ma - long_vol_ma) / long_vol_ma) * 100

    # Chaikin Money Flow (CMF)
    mfv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low']) * data['tick_volume']
    data['CMF'] = mfv.rolling(window=20).sum() / data['tick_volume'].rolling(window=20).sum()

    # Keltner Channels
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    kelner_middle = typical_price.rolling(window=20).mean()
    kelner_range = 2 * typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    data['Keltner_Upper'] = kelner_middle + kelner_range
    data['Keltner_Lower'] = kelner_middle - kelner_range

    # Additional features for class 0 prediction
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
    data['Ichimoku_Distance'] = data['close'] - (data['Leading Span A'] + data['Leading Span B']) / 2

    # Additional features for class -1 (Sell) prediction
    data['MACD_histogram_slope'] = data['MACD_histogram'].diff()
    data['RSI_overbought'] = (data['RSI'] > 70).astype(int)
    #data['BB_squeeze_sell'] = ((data['close'] - data['BB_upper']) / data['BB_std']).clip(lower=0)
    data['Negative_momentum'] = data['Momentum'].clip(upper=0)
    data['Ichimoku_bearish'] = ((data['close'] < data['Leading Span A']) & (data['close'] < data['Leading Span B'])).astype(int)

    # Additional features for class 1 (Buy) prediction
    data['MACD_bullish_cross'] = ((data['MACD'] > data['MACD_signal']) & (data['MACD'].shift(1) <= data['MACD_signal'].shift(1))).astype(int)
    data['RSI_oversold'] = (data['RSI'] < 30).astype(int)
    data['BB_bounce_buy'] = ((data['BB_lower'] - data['close']) / data['BB_std']).clip(lower=0)
    data['Positive_momentum'] = data['Momentum'].clip(lower=0)
    data['Ichimoku_bullish'] = ((data['close'] > data['Leading Span A']) & (data['close'] > data['Leading Span B'])).astype(int)

    # OHLC Analysis
    data['Price_Range'] = data['high'] - data['low']
    data['Body_Size'] = abs(data['close'] - data['open'])
    data['Direction'] = np.where(data['close'] > data['open'], 1, -1)
    data['Volatility'] = data['Price_Range'] / data['open']

    # Spread Analysis
    data['Spread_Ratio'] = data['spread'] / data['close']
    data['Spread_MA'] = data['spread'].rolling(window=20).mean()
    data['Spread_Volatility'] = data['spread'].rolling(window=20).std() / data['Spread_MA']

    # Tick Data Analysis
    data['Tick_Volume_MA'] = data['tick_volume'].rolling(window=20).mean()
    data['Tick_Volume_Ratio'] = data['tick_volume'] / data['Tick_Volume_MA']
    data['Tick_Price_Impact'] = (data['close'] - data['open']) / data['tick_volume']

    data['Support_Level'] = data['low'].rolling(window=20).min()
    data['Resistance_Level'] = data['high'].rolling(window=20).max()

    # Trend Analysis
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['Trend'] = np.where(data['close'] > data['SMA_20'], 1, -1)

    # Technical Indicators
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    data['BB_Middle'] = data['close'].rolling(window=20).mean()
    data['BB_Std'] = data['close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']

    # MACD
    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

    # Statistical Analysis
    data['Returns'] = data['close'].pct_change()
    data['Volatility_20'] = data['Returns'].rolling(window=20).std()
    data['Z_Score'] = (data['close'] - data['SMA_20']) / data['BB_Std']

    # Risk Management Metrics
    data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
    data['Stop_Loss'] = data['close'] - (2 * data['ATR'])
    data['Risk_Reward_Ratio'] = (data['Resistance_Level'] - data['close']) / (data['close'] - data['Stop_Loss'])

    # Liquidity Zone Detection
    data['High_Volume_Zone'] = (data['Volume_Ratio'] > 1.5).astype(int)
    data['Price_Cluster'] = data['close'].rolling(window=50).apply(lambda x: pd.cut(x, bins=10).value_counts().max() / len(x))

    # Order Flow Imbalance
    data['Buy_Volume'] = np.where(data['close'] > data['open'], data['tick_volume'], 0)
    data['Sell_Volume'] = np.where(data['close'] < data['open'], data['tick_volume'], 0)
    data['Volume_Imbalance'] = (data['Buy_Volume'] - data['Sell_Volume']) / data['tick_volume']

    # Historical Support/Resistance Strength
    data['Support_Strength'] = data['low'].rolling(window=100).apply(lambda x: (x == x.min()).sum() / len(x))
    data['Resistance_Strength'] = data['high'].rolling(window=100).apply(lambda x: (x == x.max()).sum() / len(x))

    # Liquidity Index
    data['Liquidity_Index'] = (data['Volume_Ratio'] + data['Price_Cluster'] +
                           abs(data['Volume_Imbalance']) +
                           (data['Support_Strength'] + data['Resistance_Strength']) / 2) / 4

    # After all feature engineering is done, reapply the original index
    data.index = original_index

    # Handle NaN values
    #data.fillna(method='ffill', inplace=True)
    #data.fillna(method='bfill', inplace=True)
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    return data

def confirm_prediction_sell_abrax(row):
    strong_signals = 0

    # OBV
    if row['OBV'].iloc[0] >= -8902.0:
        strong_signals += 1

    # CMF
    if row['CMF'].iloc[0] >= -0.004935601531600464:
        strong_signals += 1

    # ADX
    if row['ADX'].iloc[0] >= 22.523426795906914:
        strong_signals += 1

    # RSI
    if 39.64334705075517 <= row['RSI_lag_10'].iloc[0] <= 60.59907834101579:
        strong_signals += 1

    # Volatility
    if row['Volatility_20'].iloc[0] >= 0.0008333715753742228:
        strong_signals += 1

    # Volume_Ratio
    if row['Volume_Ratio'].iloc[0] >= 0.8832762146127926:
        strong_signals += 1

    # ATR
    if row['ATR'].iloc[0] >= 0.002215251727417138:
        strong_signals += 1

    # Liquidity_Index
    if row['Liquidity_Index'].iloc[0] >= 0.5278289060516576:
        strong_signals += 1

    # Confirm if majority of signals are strong
    return 'SELL' if strong_signals >= 4 else 'not_strong'

def confirm_prediction_buy_abrax(row):
    strong_signals = 0

    # Volume_Oscillator
    if -28.5396861756958 <= row['Volume_Oscillator'].iloc[0] <=71.37023040351092:
        strong_signals += 1

    # RSI_lag_5
    if 49.288617886178685 <= row['RSI_lag_5'].iloc[0] <= 98.50560398505462:
        strong_signals += 1

    # ATR
    if 0.001396535106622054 <= row['ATR'].iloc[0] <= 0.004038045176624583:
        strong_signals += 1

    # volume_lag_1
    if row['volume_lag_1'].iloc[0] >= 6808.0:
        strong_signals += 1

    # OBV
    if row['OBV'].iloc[0] >= -190251.0:
        strong_signals += 1

    # Liquidity_Index
    if 0.5281126701738172 <= row['Liquidity_Index'].iloc[0] <= 1.0330107540995082:
        strong_signals += 1

    # MACD_histogram_slope
    if row['MACD_histogram_slope'].iloc[0] >= -1.2553173171557606e-06:
        strong_signals += 1

    # Volatility_20
    if 0.0009053769452826013 <= row['Volatility_20'].iloc[0] <= 0.0043951260677922216:
        strong_signals += 1

    # CMF
    if row['CMF'].iloc[0] >= -0.0052163919280026096:
        strong_signals += 1

    # Price_Range
    if 0.0024100000000000232 <= row['Price_Range'].iloc[0] <= 0.02176:
        strong_signals += 1

    # Confirm if majority of signals are strong
    return 'BUY' if strong_signals >= 6 else 'not_strong'

def confirm_prediction_sell_dokko(row):
    strong_signals = 0

    # ATR - Volatility confirmation
    if 0.219857330189408 <= row['ATR'] <= 1.1685834831218533:
        strong_signals += 1

    # volume_lag_3 - Volume trend confirmation
    if 7235.0 <= row['volume_lag_3'] <= 65350.0:
        strong_signals += 1

    # volume_lag_5 - Volume trend confirmation
    if 7274.0 <= row['volume_lag_5'] <= 65350.0:
        strong_signals += 1

    # ADX - Strong trend confirmation
    if 28.146064682648415 <= row['ADX'] <= 72.15325451924065:
        strong_signals += 1

    # volume_lag_1 - Short-term volume confirmation
    if 7251.0 <= row['volume_lag_1'] <= 65350.0:
        strong_signals += 1

    # RSI_lag_5 - Using optimal range for trend confirmation
    if 51.709027169149365 <= row['RSI_lag_5'] <= 99.54545454545422:
        strong_signals += 1

    # Volume_MA - Volume moving average confirmation
    if 3683.5 <= row['Volume_MA'] <= 22913.1:
        strong_signals += 1

    # Spread_Volatility - Spread volatility confirmation
    if 1.6589167144263486 <= row['Spread_Volatility'] <= 4.4721359550220585:
        strong_signals += 1

    # Volume_Oscillator - Volume trend confirmation
    if -14.1629144494474 <= row['Volume_Oscillator'] <= 32.547319751481005:
        strong_signals += 1

    # Liquidity_Index - Market liquidity confirmation
    if 0.5532688119030449 <= row['Liquidity_Index'] <= 0.8029032631471555:
        strong_signals += 1

    # Return confirmation if majority of signals are strong
    return 'SELL' if strong_signals >= 7 else 'not_strong'

def confirm_prediction_buy_dokko(row):
    strong_signals = 0

    # RSI_lag_10 - Middle range for stable trend
    if row['RSI_lag_10'] <= 51.47:
        strong_signals += 1

    # ATR - Volatility confirmation
    if 0.18 <= row['ATR'] <= 0.34:
        strong_signals += 1

    # OBV - Volume trend confirmation
    if row['OBV'] >= 595509.0:
        strong_signals += 1

    # ADX - Strong trend confirmation
    if row['ADX'] >= 28.34:
        strong_signals += 1

    # MACD_histogram_slope - Momentum confirmation
    if row['MACD_histogram_slope'] >= -0.0017:
        strong_signals += 1

    # Volume_Ratio - Volume strength
    if row['Volume_Ratio'] >= 1.32:
        strong_signals += 1

    # Price_Range - Volatility confirmation
    if row['Price_Range'] <= 0.33:
        strong_signals += 1

    # Body_Size - Candle strength
    if row['Body_Size'] >= 0.209:
        strong_signals += 1

    # Volatility_20 - Market stability
    if row['Volatility_20'] <= 0.0016:
        strong_signals += 1

    # Liquidity_Index - Market liquidity
    if row['Liquidity_Index'] >= 0.64:
        strong_signals += 1

    # Confirm if majority of signals are strong
    return 'BUY' if strong_signals >= 7 else 'not_strong'

####### Beging Strategies #######

class Strategy:
    name = None
    action = None
    symbol = None
    timeframe = None
    threshold = None
    volumeConfirmation = None
    csv_file = None
    historical_df = None
    features_xg = None
    features_lg = None
    features_meta_model = None
    xg_model = None
    lg_model = None
    meta_model = None
    confirm_prediction_sell = None
    confirm_prediction_buy = None

    def read_csv(self):
        self.historical_df = pd.read_csv(self.csv_file)
        self.historical_df['time'] = pd.to_datetime(self.historical_df['time'])

    # Add threshold validation and confirmation
    def validate_and_confirm_sell(self, row):
        if row['Class 0'].iloc[0] >= self.threshold or row['Class 1'].iloc[0] >= self.threshold:
            return self.confirm_prediction_sell(row)
        else:
            return 'below_threshold'

    # Add threshold validation and confirmation
    def validate_and_confirm_buy(self, row):
        if row['Class 0'].iloc[0] >= self.threshold or row['Class 1'].iloc[0] >= self.threshold:
            return self.confirm_prediction_buy(row)
        else:
            return 'below_threshold'


abrax_sell = Strategy() #SELL:
abrax_sell.name = "Abraxas"
abrax_sell.action = "SELL"
abrax_sell.symbol = "EURUSD"
abrax_sell.timeframe = "H1"
#abrax_sell.threshold = 0.49017851029379
#abrax_sell.threshold = 0.47942295673184954
#abrax_sell.threshold = 0.7513185456178011
abrax_sell.threshold = 0.6350406453614887
abrax_sell.confirm_prediction_sell = confirm_prediction_sell_abrax
#abrax_sell.volumeConfirmation = 8870
abrax_sell.volumeConfirmation = 3642
abrax_sell.csv_file = working_dir + "/data/" + abrax_sell.symbol + "_" + abrax_sell.timeframe + ".csv"
# Load the historical data from the CSV file
abrax_sell.read_csv()

# Sell models features
abrax_sell.features_xg = [
#'tick_volume', 'BB_std', 'Momentum', 'price_change_1', 'price_change_5', 'volatility_10', 'volatility_20', 'volatility_50', 'ATR', 'ADX', 'BB_width', 'Ichimoku_Distance', 'MACD_histogram_slope', 'Price_Range', 'BB_Std', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
#'tick_volume', 'BB_lower', 'volume_lag_1', 'price_change_1', 'price_ch#ange_5', 'volatility_10', 'volatility_50', 'ATR', 'ADX', 'Volume_Oscillator', 'BB_width', 'Ichimoku_Distance', 'MACD_histogram_slope', 'Price_Range', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
'tick_volume', 'BB_std', 'volume_lag_1', 'volume_lag_2', 'volume_lag_5', 'volume_lag_10', 'price_change_1', 'price_change_5', 'volatility_10', 'volatility_20', 'volatility_50', 'ATR', 'ADX', 'Volume_Oscillator', 'BB_width', 'MACD_histogram_slope', 'Price_Range', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
]
abrax_sell.features_lg = [
#'tick_volume', 'BB_std', 'Momentum', 'price_change_1', 'volatility_10', 'volatility_20', 'volatility_50', 'ATR', 'ADX', 'BB_width', 'Ichimoku_Distance', 'MACD_histogram_slope', 'Price_Range', 'BB_Std', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
#'tick_volume', 'BB_lower', 'volume_lag_1', 'price_change_1', 'price_change_5', 'volatility_10', 'volatility_50', 'ATR', 'ADX', 'Volume_Oscillator', 'BB_width', 'Ichimoku_Distance', 'MACD_histogram_slope', 'Price_Range', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
'tick_volume', 'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_10', 'price_change_1', 'price_change_5', 'volatility_10', 'volatility_20', 'volatility_50', 'ATR', 'ADX', 'Volume_Oscillator', 'BB_width', 'MACD_histogram_slope', 'Price_Range', 'BB_Std', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
]
abrax_sell.features_meta_model = [
#'tick_volume', 'volume_lag_1', 'volume_lag_2', 'RSI_lag_2',
#'volume_lag_3', 'volume_lag_5', 'RSI_lag_5', 'volume_lag_10',
#'RSI_lag_10', 'volatility_10', 'volatility_50', 'ATR', 'Lagging Span',
#'Volume_MA', 'Volume_Oscillator', 'Price_Range', 'Volatility',
#'Volatility_20', 'Liquidity_Index', 'zero_pred_0', 'zero_pred_1',
#'one_pred_0', 'one_pred_1'

#'tick_volume', 'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
#'volume_lag_5', 'volume_lag_10', 'RSI_lag_10', 'volatility_10',
#'volatility_50', 'ATR', 'Fib_0.236', 'OBV', 'ADX', 'Lagging Span',
#'Volume_MA', 'Volume_Ratio', 'Volume_Oscillator', 'Volatility',
#'Spread_Ratio', 'Spread_MA', 'Spread_Volatility', 'Liquidity_Index',
#'zero_pred_0', 'zero_pred_1', 'one_pred_0', 'one_pred_1'

'tick_volume', 'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
'volume_lag_5', 'volume_lag_10', 'RSI_lag_10', 'volatility_10',
'volatility_50', 'ATR', 'OBV', 'ADX', 'Lagging Span', 'Volume_MA',
'Volume_Ratio', 'Volume_Oscillator', 'CMF', 'Body_Size', 'Volatility',
'Spread_Ratio', 'Tick_Price_Impact', 'Volatility_20', 'Liquidity_Index',
'zero_pred_0', 'zero_pred_1', 'one_pred_0', 'one_pred_1'
]

# Load sell models
with open(os.path.join(working_dir, "modelsBin/" + abrax_sell.action.lower() + "/" + abrax_sell.name + "_" + abrax_sell.timeframe + "_" + abrax_sell.action.lower() + "_xg_model_v3_std.sav"), "rb") as file:
    abrax_sell.xg_model = pickle.load(file)
with open(os.path.join(working_dir, "modelsBin/" + abrax_sell.action.lower() + "/" + abrax_sell.name + "_" + abrax_sell.timeframe + "_" + abrax_sell.action.lower() + "_lg_model_v3_std.sav"), "rb") as file:
    abrax_sell.lg_model = pickle.load(file)
with open(os.path.join(working_dir, "modelsBin/" + abrax_sell.action.lower() + "/" + abrax_sell.name + "_" + abrax_sell.timeframe + "_" + abrax_sell.action.lower() + "_meta_model_v3_std.sav"), "rb") as file:
    abrax_sell.meta_model = pickle.load(file)

abrax_buy = Strategy() #BUY:
abrax_buy.name = "Abraxas"
abrax_buy.action = "BUY"
abrax_buy.symbol = "EURUSD"
abrax_buy.timeframe = "H1"
#abrax_buy.threshold = 0.5733193463759658
#abrax_buy.threshold = 0.503296562216009
abrax_buy.threshold = 0.6098459517214893
abrax_buy.confirm_prediction_buy = confirm_prediction_buy_abrax
#abrax_buy.volumeConfirmation = 8108
abrax_buy.volumeConfirmation = 3328
abrax_buy.csv_file = working_dir + "/data/" + abrax_buy.symbol + "_" + abrax_buy.timeframe + ".csv"
# Load the historical data from the CSV file
abrax_buy.read_csv()

# Buy models features
abrax_buy.features_xg = [
#'tick_volume', 'day', 'BB_std', 'Momentum', 'volatility_10', 'volatility_20', 'volatility_50', 'ATR', 'ADX', 'Volume_Ratio', 'Volume_Oscillator', 'BB_width', 'Ichimoku_Distance', 'Price_Range', 'Volatility', 'BB_Std', 'BB_Width', 'Volatility_20'
#'tick_volume', 'day', 'volatility_10', 'volatility_50', 'ATR', 'Volume_Ratio', 'BB_width', 'Price_Range', 'Volatility', 'BB_Width', 'Volatility_20', 'Support_Strength', 'Liquidity_Index'
'tick_volume', 'day', 'BB_std', 'volume_lag_1', 'volume_lag_2', 'volume_lag_10', 'price_change_1', 'volatility_10', 'volatility_50', 'ATR', 'Volume_Oscillator', 'BB_width', 'Ichimoku_Distance', 'Price_Range', 'Volatility', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
]
abrax_buy.features_lg = [
#'tick_volume', 'day', 'BB_std', 'BB_lower', 'Momentum', 'volume_lag_1', 'price_change_1', 'volatility_10', 'volatility_20', 'volatility_50', 'ATR', 'ADX', 'Volume_Ratio', 'Volume_Oscillator', 'BB_width', 'Ichimoku_Distance', 'Price_Range', 'Volatility', 'BB_Std', 'BB_Width', 'Volatility_20'
#'tick_volume', 'day', 'BB_lower', 'volume_lag_1', 'volume_lag_10', 'price_change_1', 'price_change_5', 'volatility_10', 'volatility_50', 'ATR', 'Volume_Oscillator', 'BB_width', 'MACD_histogram_slope', 'Price_Range', 'Body_Size', 'Volatility', 'BB_Lower', 'BB_Width', 'Volatility_20', 'Support_Strength', 'Liquidity_Index'
'tick_volume', 'day', 'BB_std', 'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10', 'price_change_1', 'volatility_10', 'volatility_20', 'volatility_50', 'ATR', 'ADX', 'Volume_MA', 'Volume_Ratio', 'Volume_Oscillator', 'BB_width', 'Ichimoku_Distance', 'Price_Range', 'Volatility', 'BB_Std', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
]
abrax_buy.features_meta_model = [
#'tick_volume', 'volume_lag_1', 'volume_lag_2', 'RSI_lag_2',
#'volume_lag_3', 'volume_lag_5', 'RSI_lag_5', 'volume_lag_10',
#'RSI_lag_10', 'price_change_1', 'price_change_5', 'volatility_10',
#'volatility_50', 'ATR', 'ADX', 'Volume_MA', 'Volume_Ratio',
#'Volume_Oscillator', 'CMF', 'MACD_histogram_slope', 'Price_Range',
#'Body_Size', 'Volatility', 'Returns', 'Volatility_20',
#'Liquidity_Index', 'zero_pred_0', 'zero_pred_1', 'one_pred_0',
#'one_pred_1'

#'tick_volume', 'volume_lag_1', 'volume_lag_2', 'RSI_lag_2',
#'volume_lag_3', 'RSI_lag_3', 'volume_lag_5', 'RSI_lag_5',
#'volume_lag_10', 'RSI_lag_10', 'price_change_5', 'volatility_10',
#'volatility_50', 'ATR', 'ADX', 'Lagging Span', 'Volume_MA',
#'Volume_Oscillator', 'CMF', 'Ichimoku_Distance', 'MACD_histogram_slope',
#'Price_Range', 'Body_Size', 'Volatility', 'Tick_Price_Impact',
#'Volatility_20', 'Price_Cluster', 'Liquidity_Index', 'zero_pred_0',
#'zero_pred_1', 'one_pred_0', 'one_pred_1'

'tick_volume', 'month', 'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
'volume_lag_5', 'RSI_lag_5', 'volume_lag_10', 'RSI_lag_10',
'price_change_1', 'price_change_5', 'volatility_50', 'ATR', 'OBV',
'Volume_MA', 'Volume_Ratio', 'Volume_Oscillator', 'CMF',
'MACD_histogram_slope', 'Price_Range', 'Body_Size', 'Volatility',
'Volatility_20', 'Liquidity_Index', 'zero_pred_0', 'zero_pred_1',
'one_pred_0', 'one_pred_1'
]

# Load buy models
with open(os.path.join(working_dir, "modelsBin/" + abrax_buy.action.lower() + "/" + abrax_buy.name + "_" + abrax_buy.timeframe + "_" + abrax_buy.action.lower() + "_xg_model_v3_std.sav"), "rb") as file:
    abrax_buy.xg_model = pickle.load(file)
with open(os.path.join(working_dir, "modelsBin/" + abrax_buy.action.lower() + "/" + abrax_buy.name + "_" + abrax_buy.timeframe + "_" + abrax_buy.action.lower() + "_lg_model_v3_std.sav"), "rb") as file:
    abrax_buy.lg_model = pickle.load(file)
with open(os.path.join(working_dir, "modelsBin/" + abrax_buy.action.lower() + "/" + abrax_buy.name + "_" + abrax_buy.timeframe + "_" + abrax_buy.action.lower() + "_meta_model_v3_std.sav"), "rb") as file:
    abrax_buy.meta_model = pickle.load(file)

#-----

dokko_sell = Strategy() #SELL: 
dokko_sell.name = "Dokkodo"
dokko_sell.action = "SELL"
dokko_sell.symbol = "USDJPY"
dokko_sell.timeframe = "H1"
dokko_sell.threshold = 0.6916390507233896
dokko_sell.confirm_prediction_sell = confirm_prediction_sell_dokko
dokko_sell.csv_file = working_dir + "/data/" + dokko_sell.symbol + "_" + dokko_sell.timeframe + ".csv"
# Load the historical data from the CSV file
dokko_sell.read_csv()

# Sell models features
dokko_sell.features_xg = [
'tick_volume', 'volume_lag_2', 'volume_lag_5', 'price_change_5', 'volatility_10', 'ATR', 'ADX', 'Volume_Ratio', 'CMF', 'BB_width', 'Price_Range', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
]
dokko_sell.features_lg = [
'tick_volume', 'volume_lag_2', 'volume_lag_5', 'price_change_5', 'volatility_10', 'ATR', 'ADX', 'CMF', 'BB_width', 'Price_Range', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
]
dokko_sell.features_meta_model = [
'tick_volume', 'EMA_50', 'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
'volume_lag_5', 'RSI_lag_5', 'volume_lag_10', 'RSI_lag_10',
'volatility_10', 'volatility_50', 'ATR', 'MACD_histogram', 'OBV', 'ADX',
'Leading Span A', 'Volume_MA', 'Volume_Oscillator', 'CMF',
'Ichimoku_Distance', 'MACD_histogram_slope', 'Price_Range', 'Body_Size',
'Volatility', 'Spread_Ratio', 'Spread_MA', 'Spread_Volatility',
'Tick_Price_Impact', 'Volatility_20', 'Liquidity_Index', 'zero_pred_0',
'zero_pred_1', 'one_pred_0', 'one_pred_1', 'Year'
]

# Load sell models
with open(os.path.join(working_dir, "modelsBin/" + dokko_sell.action.lower() + "/" + dokko_sell.name + "_" + dokko_sell.timeframe + "_" + dokko_sell.action.lower() + "_xg_model_v3_std.sav"), "rb") as file:
    dokko_sell.xg_model = pickle.load(file)
with open(os.path.join(working_dir, "modelsBin/" + dokko_sell.action.lower() + "/" + dokko_sell.name + "_" + dokko_sell.timeframe + "_" + dokko_sell.action.lower() + "_lg_model_v3_std.sav"), "rb") as file:
    dokko_sell.lg_model = pickle.load(file)
with open(os.path.join(working_dir, "modelsBin/" + dokko_sell.action.lower() + "/" + dokko_sell.name + "_" + dokko_sell.timeframe + "_"  + dokko_sell.action.lower() + "_meta_model_v3_std.sav"), "rb") as file:
    dokko_sell.meta_model = pickle.load(file)

dokko_buy = Strategy() #BUY: 
dokko_buy.name = "Dokkodo"
dokko_buy.action = "BUY"
dokko_buy.symbol = "USDJPY"
dokko_buy.timeframe = "H1"
dokko_buy.threshold = 0.6728549941923228
dokko_buy.confirm_prediction_buy = confirm_prediction_buy_dokko
dokko_buy.csv_file = working_dir + "/data/" + dokko_buy.symbol + "_" + dokko_buy.timeframe + ".csv"
# Load the historical data from the CSV file
dokko_buy.read_csv()

# Sell models features
dokko_buy.features_xg = [
'tick_volume', 'day', 'volume_lag_2', 'volume_lag_5', 'volume_lag_10', 'price_change_5', 'volatility_10', 'volatility_50', 'ATR', 'ADX', 'Volume_MA', 'BB_width', 'Ichimoku_Distance', 'MACD_histogram_slope', 'Price_Range', 'Spread_MA', 'BB_Width', 'Volatility_20'
]
dokko_buy.features_lg = [
'tick_volume', 'day', 'EMA_20', 'volume_lag_2', 'volume_lag_5', 'volume_lag_10', 'price_change_1', 'price_change_5', 'volatility_10', 'volatility_50', 'ATR', 'ADX', 'Volume_MA', 'Volume_Ratio', 'BB_width', 'Ichimoku_Distance', 'MACD_histogram_slope', 'Price_Range', 'Body_Size', 'Spread_MA', 'BB_Width', 'Volatility_20', 'Liquidity_Index'
]
dokko_buy.features_meta_model = [
'month', 'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
'volume_lag_10', 'RSI_lag_10', 'volatility_10', 'volatility_50', 'ATR',
'MACD_histogram', 'OBV', 'ADX', 'Lagging Span', 'Volume_MA',
'Volume_Ratio', 'Volume_Oscillator', 'MACD_histogram_slope',
'Price_Range', 'Body_Size', 'Volatility', 'Spread_Ratio', 'Spread_MA',
'Tick_Price_Impact', 'Volatility_20', 'Liquidity_Index', 'zero_pred_0',
'zero_pred_1', 'one_pred_0', 'one_pred_1'
]

# Load sell models
with open(os.path.join(working_dir, "modelsBin/" + dokko_buy.action.lower() + "/" + dokko_buy.name + "_" + dokko_buy.timeframe + "_"  + dokko_buy.action.lower() + "_xg_model_v3_std.sav"), "rb") as file:
    dokko_buy.xg_model = pickle.load(file)
with open(os.path.join(working_dir, "modelsBin/" + dokko_buy.action.lower() + "/" + dokko_buy.name + "_" + dokko_buy.timeframe + "_" + dokko_buy.action.lower() + "_lg_model_v3_std.sav"), "rb") as file:
    dokko_buy.lg_model = pickle.load(file)
with open(os.path.join(working_dir, "modelsBin/" + dokko_buy.action.lower() + "/" + dokko_buy.name + "_" + dokko_buy.timeframe + "_" + dokko_buy.action.lower() + "_meta_model_v3_std.sav"), "rb") as file:
    dokko_buy.meta_model = pickle.load(file)

#-----

####### End Strategies #######

def updateHistoryFile(strategy, new_data):
    # Sort the data by time in ascending order
    #strategy.historical_df = strategy.historical_df.sort_values('time')
    # Remove duplicate rows based on the 'time' column
    #strategy.historical_df = strategy.historical_df.drop_duplicates(subset='time', keep='last')
    # Save the updated data to the CSV file
    #strategy.historical_df.to_csv(strategy.csv_file, index=False)
    #print('new_data: ', new_data)
    new_data.to_csv(strategy.csv_file, mode='a', header=False, index=False, date_format='%Y-%m-%d %H:%M:%S')

async def updateHistoryFileAsync(strategy, new_data):
    updateHistoryFile(strategy, new_data)

def calcLastProba(strategy, historical_df_engineered):
    # predictions with models
    xg_preds = strategy.xg_model.predict_proba(historical_df_engineered[strategy.features_xg])
    lg_preds = strategy.lg_model.predict_proba(historical_df_engineered[strategy.features_lg])
    #print('xg_preds', xg_preds)
    #print('lg_preds', lg_preds)

    # Create column names for probability predictions
    xg_pred_names = ['zero_pred_0', 'zero_pred_1']
    lg_pred_names = ['one_pred_0','one_pred_1']

    # Assign the probability predictions to the respective column names
    zero_preds_df = pd.DataFrame(xg_preds, columns=xg_pred_names)
    one_preds_df = pd.DataFrame(lg_preds, columns=lg_pred_names)
    #print('zero_preds_df.tail(1)', zero_preds_df.tail(1))
    #print('one_preds_df.tail(1)', one_preds_df.tail(1))

    data_metamodel = pd.concat([historical_df_engineered, zero_preds_df, one_preds_df], axis=1)
    #print(data_metamodel.shape)
    #print('data_metamodel.tail(3)', data_metamodel.tail(3))

    # Step 7: Combine base model predictions with meta-model features
    X_test_meta = np.hstack([
        data_metamodel[strategy.features_meta_model],
        zero_preds_df,
        one_preds_df
    ])

    final_proba = strategy.meta_model.predict_proba(X_test_meta)
    #print('X_test_meta', X_test_meta)
    #print('final_proba', final_proba)

    # Make predictions
    class_names = ['Class 0', 'Class 1']
    meta_model_predictions = pd.DataFrame(final_proba, columns=class_names)
    #print('meta_model_predictions', meta_model_predictions)
    joined_data_test = pd.concat([data_metamodel.reset_index(), meta_model_predictions], axis=1)
    df_to_check_test = joined_data_test[strategy.features_meta_model + ['time', 'Class 0', 'Class 1']]
    #print('df_to_check_test', df_to_check_test)

    # Get the last row of predicted probabilities
    #last_proba = final_proba[-1]
    last_proba = df_to_check_test.tail(1)
    #last_proba['time'].iloc[0] = pd.to_datetime(last_proba['time'].iloc[0])
    #print('last_proba', last_proba.columns.to_list())
    #print('last_proba', last_proba)
    #print('last_proba["OBV"]', last_proba['OBV'])
    
    return last_proba

def calcAction(last_proba, conformal_intervals, retAction):
    # Check if conformal_intervals is True
    if conformal_intervals:
        #current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if last_proba[1] > last_proba[0]:
            return retAction
        else:
            return "HOLD"
    return ""

def calcFinalAction(action_sell, action_buy):
    if action_sell == "SELL" and action_buy == "BUY":
        return "INDIRECTION"
    elif action_sell == "HOLD" and action_buy == "HOLD":
        return "HOLD"
    elif action_sell == "" and action_buy == "":
        return "" # No conformal in both
    elif action_sell == "SELL" and action_buy == "":
        return "SELL"
    elif action_sell == "" and action_buy == "BUY":
        return "BUY"
    else:
        return ""

def calcFinalAction2(action_sell, action_buy):
    if action_sell == "SELL" and action_buy == "BUY":
        return "INDIRECTION"
    elif action_sell == "HOLD" and action_buy == "HOLD":
        return "HOLD"
    elif action_sell == "" and action_buy == "":
        return "" # No conformal in both
    elif action_sell == "SELL":
        return "SELL"
    elif action_buy == "BUY":
        return "BUY"
    else:
        return ""

# Initialize MetaTrader5
if not mt5.initialize(path=mt5Path):
    print("initialize() failed, error code =", mt5.last_error())
    quit()

if not mt5.login(login=7082362, password="u#wWU#64esZjNVn", server="FPMarketsLLC-Demo"): #standard account
#if not mt5.login(login=7044310, password="75@N@3uYzcYmiz", server="FPMarketsLLC-Demo"): #raw account demo
    print("login failed, error code: {}".format(mt5.last_error()))
    quit()

@app.route('/predict', methods=['GET','POST'])
def predict():
    strategy = None
    strategy_sell = None
    strategy_buy = None
    strategy_name = None
    is_testing = None
    history = None
    if request.method == 'POST':
        to_time = pd.to_datetime(request.form['datetime'], format="%Y.%m.%d %H:%M:%S")
        strategy_name = request.form['strgName']
        is_testing = request.form['isTesting']
    else:
        to_time = pd.to_datetime(request.args.get('datetime'), format="%Y.%m.%d %H:%M:%S")
        strategy_name = request.args.get('strgName')
        is_testing = request.args.get('isTesting')

    if strategy_name and strategy_name.startswith("Abraxas"):
        strategy_sell = abrax_sell
        strategy_buy = abrax_buy
        #strategy_sell = abrax_sell_m30
        #strategy_buy = abrax_buy_m30
    elif strategy_name and strategy_name.startswith("Dokkodo"):
        strategy_sell = dokko_sell
        strategy_buy = dokko_buy
    #elif strategy_name and strategy_name.startswith("Helvetia"):
    #    strategy_sell = helve_sell
    #    strategy_buy = helve_buy
    #elif strategy_name and strategy_name.startswith("Hamilton"):
    #    strategy_sell = hamil_sell
    #    strategy_buy = hamil_buy
    #elif strategy_name and strategy_name.startswith("Mana"):
    #    strategy_sell = mana_sell
    #    strategy_buy = mana_buy
    #elif strategy_name and strategy_name.startswith("Diggers"):
    #    strategy_sell = digg_sell
    #    strategy_buy = digg_buy
    #elif strategy_name and strategy_name.startswith("Loonie"):
    #    strategy_sell = loon_sell
    #    strategy_buy = loon_buy

    # Pick one of the strategies to access the history
    strategy = strategy_sell
    print('strategy:', strategy_name)
    
    if strategy:
        end_time = strategy.historical_df.tail(1)['time'].values[0]
        print('to_time:', to_time, 'end_time:', end_time, 'is_testing:', is_testing)
        
        if to_time > end_time:
            # Get the latest data from MetaTrader5
            latest_data = mt5.copy_rates_from(strategy.symbol, mt5.TIMEFRAME_H1, to_time, 1)
            #print('latest_data: ', latest_data)
            if latest_data is None:
                print(f"Failed to retrieve data for {strategy.symbol}")
                return "MT_ERROR"
            # Convert the latest data to a DataFrame 
            latest_df = pd.DataFrame(latest_data)
            latest_df['time'] = pd.to_datetime(latest_df['time'], unit='s')
            # Append the latest data to the historical data
            strategy.historical_df = pd.concat([strategy.historical_df, latest_df], ignore_index=True)
            history = strategy.historical_df.copy()
            if is_testing == "true":
                updateHistoryFile(strategy, latest_df)
            else:
                asyncio.run(updateHistoryFileAsync(strategy, latest_df))
        else:
            # Filter historical data up to the given date 
            history = strategy.historical_df[strategy.historical_df['time'] <= to_time].copy()
        
        #history = pd.concat([history, received_df], ignore_index=True)
        print('history.tail(1)', history.tail(1))

        # Perform features engineering on the historical data
        historical_df_engineered = features_engineering(history)
        print('historical_df_engineered.tail(1)', historical_df_engineered.tail(1))

        # Calculate last proba
        last_proba_sell = calcLastProba(strategy_sell, historical_df_engineered)
        last_proba_buy = calcLastProba(strategy_buy, historical_df_engineered)

        # Calculate confirmations
        action_sell = strategy_sell.validate_and_confirm_sell(last_proba_sell)
        action_buy = strategy_buy.validate_and_confirm_buy(last_proba_buy)
        print('action_sell:', action_sell)
        print('action_buy:', action_buy)

        # Apply conformal prediction
        #conformal_intervals_sell = apply_conformal_prediction(last_proba_sell, strategy_sell.threshold)
        #conformal_intervals_buy = apply_conformal_prediction(last_proba_buy, strategy_buy.threshold)
        #print('conformal_intervals_sell:', conformal_intervals_sell)
        #print('conformal_intervals_buy:', conformal_intervals_buy)

        # Calculating actions
        #action_sell = calcAction(last_proba_sell, conformal_intervals_sell, strategy_sell.action)
        #action_buy = calcAction(last_proba_buy, conformal_intervals_buy, strategy_buy.action)
        
        # Calculating final action
        final_action = calcFinalAction2(action_sell, action_buy)
        
#        tick_volume = history.tail(1)['tick_volume'].values[0]
#        if final_action == "SELL":
#            if tick_volume < strategy_sell.volumeConfirmation:
#                final_action = "SELL_NO_VOL"
#        elif final_action == "BUY":
#            if tick_volume < strategy_buy.volumeConfirmation:
#                final_action = "BUY_NO_VOL"

        print(f"{strategy_name} >> {action_sell} : {action_buy} -> {final_action}")
        return final_action
    else:
        return "NO_STRG"
