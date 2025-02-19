import requests
import pandas as pd
import os
import re

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = os.getenv("PPLX_API_KEY")  # Recommended: Use environment variable

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

path = "C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/processed/last_final_decision.csv"

# Load and validate data structure
try:
    df = pd.read_csv(path)
    required_columns = ['timestamp', 'currency_pair', 'final_decision', 'confidence_score']
    assert all(col in df.columns for col in required_columns), "Missing required columns"
except Exception as e:
    print(f"Data validation error: {str(e)}")
    exit()

# Time validation (1hr prediction window)
current_time = pd.Timestamp.now(tz='UTC')
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')
time_valid_df = df[(current_time - df['timestamp']) < pd.Timedelta(hours=1)]

# Action filtering
filtered_df = time_valid_df[time_valid_df['final_decision'].isin(['sell', 'buy'])].copy()

def generate_validation_prompt(action: str, pair: str) -> dict:
    """Structured validation prompt with technical/fundamental requirements"""
    return {
        "model": "sonar-reasoning-pro",
        "messages": [
            {
                "role": "system",
                "content": f"""Act as Senior FX Market Analyst. Conduct rigorous 1hr timeframe validation for {pair}:

**Fundamental Validation (News/Sentiment):**
1. Verify breaking news from tradingview.com/news, forexfactory.com/calendar (last 2hrs)
2. Analyze market sentiment from dailyfx.com/sentiment (current readings)
3. Check central bank interventions (ECB/FED/BOE announcements)
4. Validate risk appetite (DXY, bond yields correlation)

**Technical Validation (1hr Chart):**
1. Price action: Support/resistance confluence within ±0.5%
2. Momentum: RSI(14) divergence from 1hr/4hr trends
3. Order flow: Recent liquidity pools & volume spikes
4. Confluence with 50/100 EMA cross on 1hr

**Decision Framework:**
- STRONG CONFIRMATION: ≥3 technical + ≥2 fundamental factors aligned
- WEAK CONFIRMATION: 2 technical + 1 fundamental factors aligned
- REJECTION: Any conflicting signals or insufficient confluence

Return ONLY this format:
FINAL DECISION: [CONFIRM/REJECT] {action.upper()} {pair}
RATIONALE: <2-sentence summary>"""
            }
        ]
    }

def execute_validation(action: str, pair: str) -> str:
    """Execute API query with error handling"""
    try:
        response = requests.post(
            API_ENDPOINT,
            headers=headers,
            json=generate_validation_prompt(action, pair),
            timeout=10
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"

# Process decisions with logging
for index, row in filtered_df.iterrows():
    action = row['final_decision'].lower()
    pair = row['currency_pair'].replace(" ", "").upper()  # Normalize pair format
    
    if not re.match(r"^[A-Z]{3}/[A-Z]{3}$", pair):
        print(f"Invalid pair format: {pair}")
        continue
        
    result = execute_validation(action, pair)
    
    # Parse structured response
    if "FINAL DECISION:" in result:
        decision = result.split("FINAL DECISION:")[1].strip()
        print(f"{pair}@{row['timestamp']} | Action: {action.upper()} | Result: {decision}")
    else:
        print(f"Invalid response format: {result}")
