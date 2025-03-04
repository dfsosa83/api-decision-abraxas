import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

def find_latest_timestamp(file_path: str) -> str:
    """Find the most recent timestamp in the log file"""
    latest = None
    
    with open(file_path, 'r') as f:
        log_entries = f.read().split('================================================================================')
        
        for entry in log_entries:
            if match := re.search(r'Timestamp:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', entry):
                current = datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
                if not latest or current > latest:
                    latest = current
                    
    return latest.strftime('%Y-%m-%d %H:%M:%S') if latest else None

def parse_trading_log(file_path: str) -> pd.DataFrame:
    """Parse log file with enhanced regex patterns"""
    data = []
    pattern = re.compile(
        r'Timestamp:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
        r'.*?FINAL DECISION:\s*(.*?)\*\*'
        r'.*?Trade Execution.*?'
        r'- Entry:\s*\[([\d\.-]+)\].*?'
        r'- Stop Loss:\s*\[([\d\.]+)\].*?'
        r'- Take Profit:\s*\[([\d\.]+)\].*?',
        re.DOTALL
    )
    
    with open(file_path, 'r') as f:
        for entry in f.read().split('================================================================================'):
            if match := pattern.search(entry):
                data.append({
                    'timestamp': datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S'),
                    'decision': match.group(2).strip(),
                    'entry': match.group(3),
                    'stop_loss': match.group(4),
                    'take_profit': match.group(5)
                })
    
    return pd.DataFrame(data)

# Find most recent timestamp
input_path = 'C:/Users/david/OneDrive/Documents/api-decision-abraxas/data/logs/agent_responses.log'
latest_ts = find_latest_timestamp(input_path)
print(f"Most recent timestamp in log: {latest_ts}")

# Process and save to Parquet
df = parse_trading_log(input_path)
if not df.empty:
    output_path = f'{input_path.rsplit(".", 1)[0]}.parquet'
    
    schema = pa.schema([
        ('timestamp', pa.timestamp('ns')),
        ('decision', pa.string()),
        ('entry', pa.string()),
        ('stop_loss', pa.string()),
        ('take_profit', pa.string())
    ])
    
    pq.write_table(pa.Table.from_pandas(df, schema=schema), output_path)
    print(f"\nParquet file contents for {latest_ts}:")
    print(pq.read_table(output_path).to_pandas())
else:
    print("No valid entries found")
