"""
inspect_csv.py — Read all three CSV files and print their structure.
No assumptions about column names. Outputs:
  - filename, shape, columns, dtypes, head(), missing counts
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path("D:/pythonWork/formal_test/figure2")
FILES = [
    "profile_paper.csv",
    "ppo_training_paper.csv",
    "ppo_summary_paper.csv",
]

for fname in FILES:
    fp = DATA_DIR / fname
    print("=" * 70)
    print(f"File: {fname}")
    print(f"  Exists: {fp.exists()}, Size: {fp.stat().st_size} bytes")

    df = pd.read_csv(fp)

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")
    print(f"  Missing per column:\n{df.isnull().sum().to_string()}")
    print(f"  Duplicated rows: {df.duplicated().sum()}")
    print(f"  Head (5 rows):")
    print(df.head().to_string(index=True))
    print(f"  Tail (5 rows):")
    print(df.tail().to_string(index=True))

    # Numeric summary
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        print(f"  Numeric summary:")
        print(df[num_cols].describe().to_string())

    print()
