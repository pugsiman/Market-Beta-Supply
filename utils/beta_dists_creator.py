#!/usr/bin/env python3

import numpy as np
from yahooquery import Ticker
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import os

def create_beta_distribution(sample_data, date):
    filepath = f'data/beta_dist-{date}.json'

    if not os.path.exists(filepath):
        with open(filepath, 'w+') as f:
            trailing_window = sample_data.loc[date-pd.DateOffset(days=252):date]
    
            welch_betas = []
            for ticker in tickers.split(' '):
                try:
                    beta = Beta(trailing_window[BENCHMARK_INDEX], trailing_window[ticker]).welch()
                    welch_betas.append(beta)
                except KeyError:
                    print(f'{ticker} was truncated out of dataframe and could not be calculated')
                    continue

            welch_series = pd.Series(welch_betas)
            f.write(welch_series.to_json())

    return filepath

tickers = open('data/tickers.txt', 'r').read()
sample_data = Ticker(f'{tickers} {BENCHMARK_INDEX}', asynchronous=True).history(period='2y')['adjclose']
sample_returns = sample_data.unstack().T.pct_change(fill_method=None).dropna() 
sample_returns.index = pd.DatetimeIndex(sample_returns.index)

dates = pd.bdate_range(start='05/1/2023', end='18/1/2024')
for date in dates:
    create_beta_distribution(sample_returns, date)

