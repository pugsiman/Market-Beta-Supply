#!/usr/bin/env python3

from rich import print
import numpy as np
import yfinance as yf
import pandas as pd
import investpy
import requests
import plotly.graph_objects as go
import tabula
from datetime import datetime
import os

pd.options.plotting.backend = 'plotly'

class Beta:
    def __init__(self, x: np.array, y: np.array):
        self.x = np.atleast_2d(x).T
        self.y = np.atleast_2d(y).T
        self.n_obs = x.shape[0]
        self.x_mat = np.hstack([np.ones((self.x.shape[0], 1)), self.x])

    def ols(self):
       return np.ravel(self._ols(self.x, self.y))[-1]

    def welch(self, delta: float = 3, rho = 2/256) -> float:
        bm_min, bm_max = (1 - delta) * self.x, (1 + delta) * self.x
        lower, upper = np.minimum(bm_min, bm_max), np.maximum(bm_min, bm_max)
        y_winsorized = np.atleast_2d(np.clip(self.y, lower, upper))
        weights = np.exp(-rho * np.arange(self.n_obs)[::-1]) if rho else None
        return np.ravel(self._ols(self.x_mat, y_winsorized, weights=weights))[1]

    def _ols(self, x, y, weights = None):
        if weights is None:
            weights = np.ones(x.shape[0])
        weights = np.diag(weights)

        return np.linalg.inv(x.T @ weights @ x) @ x.T @ weights @ y

def rolling_beta_fig(sample_returns):
    cov = sample_returns.rolling(21).cov().unstack()[BENCHMARK_INDEX][STOCK]
    var = sample_returns[BENCHMARK_INDEX].to_frame().rolling(21).var()
    rolling_ols_beta = (cov/var.iloc[:,0])

    sample_returns[STOCK] *= 100
    sample_returns[BENCHMARK_INDEX] *= 100

    returns_fig = sample_returns.plot().update_traces(opacity=0.3)
    beta_fig = go.Figure(data=go.Scatter(
        x = rolling_ols_beta.index,
        y = rolling_ols_beta.values,
        name='beta',
    )).update_traces(marker=dict(color='blue'))

    fig = go.Figure(data = returns_fig.data + beta_fig.data).update_layout(title=f'{STOCK} rolling beta', title_x = 0.5)
    fig.show()

def period_betas_window(chunk):
    date, df = chunk
    beta = Beta(df[BENCHMARK_INDEX], df[STOCK])
    trailing_window = sample_returns.loc[date-pd.DateOffset(days=252):date]

    trailing_beta = Beta(trailing_window[BENCHMARK_INDEX], trailing_window[STOCK])
    inception_beta = Beta(df[BENCHMARK_INDEX], df[STOCK])

    return [
        date, {
            'ols_beta': beta.ols(),
            'bsw_beta': beta.welch(rho=False),
            'bswa_beta': beta.welch(),
            'true_bswa_beta(inception)': inception_beta.welch(),
            'true_bswa_beta(trailing)': trailing_beta.welch()
        }
    ]

def isins_to_tickers(isin):
    try: 
        print(f'attempting to tanslate ISIN #{isin}...')
        return investpy.stocks.search_stocks(by='isin', value=f'US{isin}')['symbol'][0]
    except RuntimeError as e:
        print('failed')
        if 'ERR#0043' in e.args[0]: pass # match investpy network exception specifically
        raise e

def persist_isins_as_tickers(isins):
    filepath = f'data/tickers-{today_date}.txt'
    if not os.path.exists(filepath):
        with open(filepath, 'a+') as f:
            tickers = list(filter(None, map(isins_to_tickers, isins)))
            f.write(' '.join(map(str, tickers)))

    return open(filepath, 'r').read()

def create_beta_distribution(ticker):
    filepath = f'data/beta_dist-{today_date}.json'

    if not os.path.exists(filepath):
        with open(filepath, 'w+') as f:
            sample_data = yf.download(f'{tickers} {BENCHMARK_INDEX}', start=start_date, progress=False)['Adj Close']
            sample_returns = sample_data.pct_change().dropna(axis='columns', how='all').dropna(axis='index', how='all')
            sample_returns.index = pd.DatetimeIndex(sample_returns.index)
            trailing_window = sample_returns.iloc[-252:]
    
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


BENCHMARK_INDEX = 'SPY'
STOCK = 'NVDA'
start_date = '2018-01-01'
today_date = datetime.today().strftime('%Y-%m-%d')

# sample_data = yf.download(f'{STOCK} {BENCHMARK_INDEX}', start=start_date, progress=False)['Adj Close']
# sample_returns = sample_data.pct_change().dropna()
# sample_returns.index = pd.DatetimeIndex(sample_returns.index)
# rolling_beta_fig(sample_returns)

# results = dict(map(period_betas_window, sample_returns.groupby(pd.Grouper(freq='M'))))
# df = pd.DataFrame(results).T

# fig = df.plot(title=f'{STOCK} rolling betas comparison').update_layout(title_x=0.5, xaxis_title='Date', legend_title='estimators').update_xaxes(dtick='M1').show()
pdf = tabula.read_pdf('https://assets-global.website-files.com/60f8038183eb84c40e8c14e9/6584d5f51c7b43b924c8e414_Wilshire-5000-Index-Fund-Holdings.pdf', stream=True, pages='all', guess=False)
isins = np.concatenate(list(map(lambda df: df.iloc[2::,0].values, pdf)))
tickers = persist_isins_as_tickers(isins) # 'AAPL CVX NRG ...'
beta_series_json_path = create_beta_distribution(tickers)
beta_series = pd.read_json(beta_series_json_path, typ='series')
beta_series.hist().update_layout(title_text='Market beta distribution', title_x=0.5, bargap=0.2).show()

