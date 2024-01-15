#!/usr/bin/env python3

from os import name
from rich import print
import numpy as np
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.options.plotting.backend = 'plotly'

class Beta:
    def __init__(self, x: np.array, y: np.array):
        self.x = np.atleast_2d(x).T
        self.y = np.atleast_2d(y).T
        self.n_obs = x.shape[0]
        self.x_mat = np.hstack([np.ones((self.x.shape[0], 1)), self.x])

    def ols(self, x, y, weights = None):
        if weights is None:
            weights = np.ones(x.shape[0])
        weights = np.diag(weights)

        return np.linalg.inv(x.T @ weights @ x) @ x.T @ weights @ y

    def welch(self, delta: float = 3, rho = 2/256) -> float:
        bm_min, bm_max = (1 - delta) * self.x, (1 + delta) * self.x
        lower, upper = np.minimum(bm_min, bm_max), np.maximum(bm_min, bm_max)
        y_winsorized = np.atleast_2d(np.clip(self.y, lower, upper))
        weights = np.exp(-rho * np.arange(self.n_obs)[::-1])
        beta = self.ols(self.x_mat, y_winsorized, weights=weights)
        return np.ravel(beta)[1]

BENCHMARK_INDEX = 'SPY'
STOCK = 'AAPL'
start_date = '2018-01-01'

sample_data = yf.download(f'{STOCK} {BENCHMARK_INDEX}', start=start_date, progress=False)['Adj Close']
sample_returns = sample_data.pct_change().dropna()
sample_returns.index = pd.DatetimeIndex(sample_returns.index)

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

beta = Beta(sample_returns[BENCHMARK_INDEX], sample_returns[STOCK])
ols_beta = beta.ols(beta.x, beta.y)[0]
welch_beta = beta.welch()
breakpoint()

fig = go.Figure(data = returns_fig.data + beta_fig.data).update_layout(title=f'{STOCK} rolling beta', title_x = 0.5)
# fig.show()

