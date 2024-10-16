#!/usr/bin/env python3

from yahooquery import Ticker
import pandas as pd
from datetime import datetime, date, timedelta
import os
from beta import Beta

BENCHMARK_INDEX = 'SPY'


def create_beta_distribution(sample_returns, date_str, tickers):
    filepath = f'data/beta_dist-{date_str}.json'

    if not os.path.exists(filepath):
        with open(filepath, 'w+') as f:
            trailing_window = sample_returns.loc[
                date_str - pd.DateOffset(days=252) : date_str
            ]

            welch_betas = {}
            for ticker in tickers.split(' '):
                try:
                    beta = Beta(
                        trailing_window[BENCHMARK_INDEX], trailing_window[ticker]
                    ).welch()
                    welch_betas[ticker] = beta
                except KeyError:
                    print(
                        f'{ticker} ({date_str}) was truncated out of dataframe and could not be calculated'
                    )
                    continue

            welch_series = pd.Series(welch_betas)
            f.write(welch_series.to_json())

    return filepath


def persist_tickers(tickers):
    filepath = 'data/tickers.txt'
    if not os.path.exists(filepath):
        with open(filepath, 'a+') as f:
            f.write(' '.join(map(str, tickers)))

    return filepath


def main():
    stocks_filepath = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt'
    df = pd.read_csv(stocks_filepath, sep='|')
    sample_stocks = df[
        (df['Test Issue'] == 'N')
        & (df['ETF'] == 'N')
        & (df['Symbol'].str.match(r'^[A-Za-z]{2,4}$', na=False))
        & ~(df['Security Name'].str.contains('ETN', na=False))
    ]
    tickers = sample_stocks['Symbol'].values
    tickers_filepath = persist_tickers(tickers)
    tickers = open(tickers_filepath, 'r').read()

    sample_data = Ticker(f'{tickers} {BENCHMARK_INDEX}', asynchronous=True).history(
        period='3y', interval='1d'
    )['adjclose']

    sample_returns = (
        sample_data.unstack()
        .T.pct_change(fill_method=None)
        .dropna(axis='index', how='all')
    )
    sample_returns.index = pd.DatetimeIndex(sample_returns.index).tz_localize(None)
    dates = pd.bdate_range(
        start='1/1/2021', end=pd.to_datetime('now').tz_localize('EST').date()
    )

    for date_str in dates:
        create_beta_distribution(sample_returns, date_str, tickers)


if __name__ == '__main__':
    main()
