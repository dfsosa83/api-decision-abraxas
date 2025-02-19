import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from colorama import Fore, Back, Style, init
import ta
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
import pickle
import joblib

#define data path
path = 'C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/processed/'

#load data
df_buy = pd.read_csv(path + 'decision_dataset_buy_dokkodo.csv')
df_sell = pd.read_csv(path + 'decision_dataset_sell_dokkodo.csv')

#merge data by datetime
df_0 = pd.merge(df_buy, df_sell, on='datetime', how='inner')

# Define the desired column order
ordered_columns = ['datetime', 'Class0_buy', 'Class1_buy', 'Class0_sell', 'Class1_sell', 'confir_buy', 'confir_sell']

# Reorder the columns of the merged dataframe
merged_df = df_0[ordered_columns].copy()

#head
#print(df.head(10))

from collections import Counter

# Count combinations of 'confir_buy' and 'confir_sell'
combinations = Counter(zip(merged_df['confir_buy'], merged_df['confir_sell']))

# Print the combinations and their counts
#for (buy, sell), count in combinations.items():
 #   print(f"confir_buy: {buy}, confir_sell: {sell}, Count: {count}")

# Define the decision column
def add_final_decision_column(df):
    conditions = [
        (df['confir_buy'] == 'below_threshold') & (df['confir_sell'] == 'below_threshold'),
        (df['confir_buy'] == 'not_strong') & (df['confir_sell'] == 'below_threshold'),
        (df['confir_buy'] == 'not_strong') & (df['confir_sell'] == 'confirm'),
        (df['confir_buy'] == 'confirm') & (df['confir_sell'] == 'confirm'),
        (df['confir_buy'] == 'confirm') & (df['confir_sell'] == 'below_threshold'),
        (df['confir_buy'] == 'confirm') & (df['confir_sell'] == 'not_strong'),
        (df['confir_buy'] == 'not_strong') & (df['confir_sell'] == 'not_strong'),
        (df['confir_buy'] == 'below_threshold') & (df['confir_sell'] == 'not_strong'),
        (df['confir_buy'] == 'below_threshold') & (df['confir_sell'] == 'confirm')
    ]

    decisions = [
        'do_nothing',
        'do_nothing',
        'sell',
        'indecision',
        'buy',
        'buy',
        'do_nothing',
        'do_nothing',
        'sell'
    ]

    df['final_decision'] = pd.Series(np.select(conditions, decisions, default='unknown'))
    return df

# Add the final decision column to the dataframe
updated_df = add_final_decision_column(merged_df)
#print(updated_df.head(10))

#print updated_df values
#print(updated_df['final_decision'].value_counts())

#add a column with the currency pair
updated_df['currency_pair'] = 'USD/JPY'

#save in a df last 5 rows
df_last_row = updated_df.tail(10)

# print a message only if last row is sell or buy
if df_last_row['final_decision'].iloc[0] in ['sell', 'buy']:
    print(Fore.GREEN + Style.BRIGHT + '\n' + '=' * 50)
    print('Last row is sell or buy for: DOKKODO'.center(50))
    print('=' * 50 + '\n' + Style.RESET_ALL)
else:
    print(Fore.RED + Style.BRIGHT + '\n' + '=' * 50)
    print('Last row is not sell or buy for: DOKKODO'.center(50))
    print('=' * 50 + '\n' + Style.RESET_ALL)

#save in a csv file
df_last_row.to_csv(path + 'last_final_decision_dokkodo.csv', index=False)





