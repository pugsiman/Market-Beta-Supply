#!/usr/bin/env python3

from yahooquery import Ticker
import pandas as pd
import os
from beta import Beta

BENCHMARK_INDEX = 'SPY'
INITIAL_DATE = '1/1/2021'
NASDAQ_FTP_PATH = 'ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt'


def create_beta_distribution(sample_returns, date_str: str, tickers: str) -> str:
    """Creates a data distribution for all current beta estimator values and residuals for the date, then saves it
    Parameters
    ----------
    sample_returns: dataframe
    date_str: string formatted as '%Y-%m-%d'
    tickers: string containing comma-seperated tickers

    Returns
    -------
    filepath: string
    """
    filepath = f'data/beta_dist-{date_str}.json'

    if not os.path.exists(filepath):
        with open(filepath, 'w+') as f:
            trailing_window = sample_returns.loc[
                date_str - pd.DateOffset(days=252) : date_str
            ]

            welch_betas = {'values': {}, 'residuals': {}}
            for ticker in tickers.split(' '):
                try:
                    beta = Beta(
                        trailing_window[BENCHMARK_INDEX], trailing_window[ticker]
                    ).welch()

                    welch_betas['values'][ticker] = beta[1]
                    welch_betas['residuals'][ticker] = beta[0]
                except KeyError:
                    print(
                        f'{ticker} ({date_str}) was truncated out of dataframe and could not be calculated'
                    )
                    continue

            welch_series = pd.Series(welch_betas)
            f.write(welch_series.to_json())

    return filepath


def persist_tickers(tickers) -> str:
    filepath = 'data/tickers.txt'
    if not os.path.exists(filepath):
        with open(filepath, 'a+') as f:
            f.write(' '.join(map(str, tickers)))

    return filepath


def main():
    stocks_filepath = NASDAQ_FTP_PATH
    df = pd.read_csv(stocks_filepath, sep='|')
    # attempt to clean off some of the obviously bad tickers (ETFs, ETNs, tests, broken symbols etc')
    sample_stocks = df[
        (df['Test Issue'] == 'N')
        & (df['ETF'] == 'N')
        & (df['Symbol'].str.match(r'^[A-Za-z]{1,4}$', na=False))
        & ~(df['Security Name'].str.contains('ETN', na=False))
        & ~(df['Security Name'].str.contains('Acquisition', na=False))
        & ~(df['Security Name'].str.contains('ADR', na=False))
        & ~(df['Security Name'].str.contains('Depositary', na=False))
        & ~(df['Security Name'].str.contains('Trust', na=False))
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
        start=INITIAL_DATE, end=pd.to_datetime('now').tz_localize('EST').date()
    )

    for date_str in dates:
        create_beta_distribution(sample_returns, date_str, tickers)


if __name__ == '__main__':
    main()
