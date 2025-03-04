import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import ta
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
import pickle
import joblib

#define data path
path = 'C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/'

#read data from path
mtd5_df = pd.read_csv(path+'Export_USDJPY_H1.csv')
#pint shape
#print(mtd5_df.shape)

#time to datitime
mtd5_df['time'] = pd.to_datetime(mtd5_df['time'])
#filter data by time
mtd5_df = mtd5_df[mtd5_df['time'] >= '2024-01-01']
#reset index
mtd5_df = mtd5_df.reset_index(drop=True)


#define models path
path_models = 'C:/Users/david/OneDrive/Documents/api-decision-abraxas/modelsBin/sell/'

# Load the base models
xg_model_0 = joblib.load(path_models + 'Dokkodo_H1_sell_xg_model_v3_std.sav')
lg_model_1 = joblib.load(path_models + 'Dokkodo_H1_sell_lg_model_v3_std.sav')
#load metamodel
best_meta_model = joblib.load(path_models + 'Dokkodo_H1_sell_meta_model_v3_std.sav')  # Assuming you saved it with this name

#feature engineering function
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
    data['Year'] = data['time'].dt.year

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

#apply feature_function to dataframe
mtd5_df = features_engineering(mtd5_df)
#print(mtd5_df.shape)

#xgboos model
selected_features_names_0 = [
'tick_volume', 
'volume_lag_2', 
'volume_lag_5', 
'price_change_5', 
'volatility_10', 
'ATR', 
'ADX', 
'Volume_Ratio', 
'CMF', 
'BB_width', 
'Price_Range', 
'BB_Width', 
'Volatility_20', 
'Liquidity_Index'
]

#lightgbm model
selected_features_names_1 = [
'tick_volume', 
'volume_lag_2', 
'volume_lag_5', 
'price_change_5', 
'volatility_10', 
'ATR', 
'ADX', 
'CMF', 
'BB_width', 
'Price_Range', 
'BB_Width', 
'Volatility_20', 
'Liquidity_Index'
]

#metamodel
selected_features_names_meta_model = [
'tick_volume',
 'EMA_50',
 'volume_lag_1',
 'volume_lag_2',
 'volume_lag_3',
 'volume_lag_5',
 'RSI_lag_5',
 'volume_lag_10',
 'RSI_lag_10',
 'volatility_10',
 'volatility_50',
 'ATR',
 'MACD_histogram',
 'OBV',
 'ADX',
 'Leading Span A',
 'Volume_MA',
 'Volume_Oscillator',
 'CMF',
 'Ichimoku_Distance',
 'MACD_histogram_slope',
 'Price_Range',
 'Body_Size',
 'Volatility',
 'Spread_Ratio',
 'Spread_MA',
 'Spread_Volatility',
 'Tick_Price_Impact',
 'Volatility_20',
 'Liquidity_Index',
 'zero_pred_0',
 'zero_pred_1',
 'one_pred_0',
 'one_pred_1',
 'Year'
]

#made predictions
#define final data set:
full_test_data_final = mtd5_df.copy()
#print(full_test_data_final.shape)

#Step 1:predictions with models
zero_preds = xg_model_0.predict_proba(full_test_data_final[selected_features_names_0])
one_preds = lg_model_1.predict_proba(full_test_data_final[selected_features_names_1])

# Step 2: Create column names for probability predictions
zero_pred_names = ['zero_pred_0', 'zero_pred_1']
one_pred_names = ['one_pred_0','one_pred_1']

#Step 3: Assign the probability predictions to the respective column names
zero_preds_df = pd.DataFrame(zero_preds, columns=zero_pred_names)
one_preds_df = pd.DataFrame(one_preds, columns=one_pred_names)

#Step 4: add new columns to full_data_to_predict
data_metamodel = pd.concat([full_test_data_final, zero_preds_df, one_preds_df], axis=1)
#print(data_metamodel.shape)

# Step 5: Extract features for base models
X_test_0_final = data_metamodel[selected_features_names_0]
X_test_1_final = data_metamodel[selected_features_names_1]

# Step 6: Get predictions from base models
final_zero_test_preds = xg_model_0.predict_proba(X_test_0_final)
final_one_test_preds = lg_model_1.predict_proba(X_test_1_final)

# Step 7: Combine base model predictions with meta-model features
X_test_meta_final = np.hstack([
    data_metamodel[selected_features_names_meta_model],
    final_zero_test_preds,
    final_one_test_preds
])

# Step 8: Make predictions
y_pred = best_meta_model.predict(X_test_meta_final)
y_pred_proba = best_meta_model.predict_proba(X_test_meta_final)

# Step 9: probabilities are needed, use predict_proba
test_meta_prob_preds = y_pred_proba
#print(test_meta_prob_preds)

#define class names
class_names = ['Class0_sell', 'Class1_sell']

#add names
meta_model_predictions = pd.DataFrame(test_meta_prob_preds, columns=class_names)

#join meta_model_predictions to df_test data,
joined_data_test = pd.concat([data_metamodel.reset_index(), meta_model_predictions], axis=1)
print(joined_data_test.shape)

#only with metamodel features
df_to_apply_rules_sell = joined_data_test[selected_features_names_meta_model + ['time', 'Class0_sell', 'Class1_sell']]
#print(df_to_apply_rules_sell.shape)

#rename time by datetime
df_to_apply_rules_sell.rename(columns={'time': 'datetime'}, inplace=True)

#conformal threshold
threshold = 0.6916390507233896

#rules function
def confirm_prediction(row):
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
    return 'confirm' if strong_signals >= 6 else 'not_strong'

def validate_and_confirm(row):
    if row['Class0_sell'] >= threshold or row['Class1_sell'] >= threshold:
        return confirm_prediction(row)
    else:
        return 'below_threshold'

# Apply the validation and confirmation function
df_to_apply_rules_sell['confir_sell'] = df_to_apply_rules_sell.apply(validate_and_confirm, axis=1)

# define decision dataset buy
decision_dataset_sell = df_to_apply_rules_sell[['datetime', 'Class0_sell', 'Class1_sell', 'confir_sell']].copy()

#define path for processed data
path_decision = 'C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/processed/'

#save decision dataset
decision_dataset_sell.to_csv(path_decision + 'decision_dataset_sell_dokkodo.csv', index=False)

#print head
print('Sell Predictions',decision_dataset_sell.tail())