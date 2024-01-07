#!/usr/bin/env python3

import numpy as np
import yfinance as yf
import pandas as pd
import requests
start_date = "2017-01-01"
end_date = "2024-01-01"

sample_data = yf.download("AAPL SPY", start=start_date, end=end_date, progress=False)["Adj Close"]
sample_returns = sample_data.pct_change().dropna()
sample_returns.index = pd.DatetimeIndex(sample_returns.index)

print(sample_returns.index)

