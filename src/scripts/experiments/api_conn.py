import requests
import pandas as pd
import os

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Path to last final decision
path = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data//processed/last_final_decision.csv"

# Load data
df = pd.read_csv(path)
print(df)

# Filter for 'sell' or 'buy' values in the 'final_decision' column
filtered_df = df[df['final_decision'].isin(['sell', 'buy'])]

def query_perplexity(action, currency_pair):
    data = {
        "model": "sonar-reasoning-pro",
        "messages": [
            {"role": "system", "content": "You are a quantum forex trading expert."},
            {"role": "user", "content": f"Validate this forex prediction for the next hour: {action} {currency_pair}. Consider recent news and market analysis from tradingview.com/news, investing.com/analysis/market-overview, dailyforex.com/currencies/eur/usd, forexfactory.com/market/eurusd, fxstreet.es and market sentiment from https://www.dailyfx.com/sentiment. After your analysis, provide a clear CONFIRMATION or REJECTION of the {action} action, starting with 'FINAL DECISION:'."}
        ]
    }
    
    response = requests.post(API_ENDPOINT, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}, {response.text}"

# Iterate through the filtered DataFrame and query the API
for index, row in filtered_df.iterrows():
    action = row['final_decision']
    currency_pair = row['currency_pair']
    
    result = query_perplexity(action, currency_pair)
    print(f"Query: {action} {currency_pair}")
    print(f"Response: {result}\n")
