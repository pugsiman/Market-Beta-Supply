#!/usr/bin/env python3

from rich import print
import numpy as np
import yahooquery as yq
import pandas as pd
import requests
import plotly.graph_objects as go
import tabula
from datetime import datetime, date, timedelta
import os
from utils.beta import Beta

pd.options.plotting.backend = 'plotly'

BENCHMARK_INDEX = 'SPY'
STOCK = 'NVDA'
start_date = '2018-01-01'
today_date = datetime.today().strftime('%Y-%m-%d')


def rolling_beta_fig(sample_returns):
    cov = sample_returns.rolling(21).cov().unstack()[BENCHMARK_INDEX][STOCK]
    var = sample_returns[BENCHMARK_INDEX].to_frame().rolling(21).var()
    rolling_ols_beta = cov / var.iloc[:, 0]

    sample_returns[STOCK] *= 100
    sample_returns[BENCHMARK_INDEX] *= 100

    returns_fig = sample_returns.plot().update_traces(opacity=0.3)
    beta_fig = go.Figure(
        data=go.Scatter(
            x=rolling_ols_beta.index,
            y=rolling_ols_beta.values,
            name='beta',
        )
    ).update_traces(marker=dict(color='blue'))

    fig = go.Figure(data=returns_fig.data + beta_fig.data).update_layout(
        title=f'{STOCK} rolling beta', title_x=0.5
    )
    fig.show()


def period_betas_window(chunk):
    date_idx, df = chunk
    beta = Beta(df[BENCHMARK_INDEX], df[STOCK])
    trailing_window = sample_returns.loc[date_idx - pd.DateOffset(days=252) : date_idx]
    inception_window = sample_returns.loc[:date_idx]

    trailing_beta = Beta(trailing_window[BENCHMARK_INDEX], trailing_window[STOCK])
    inception_beta = Beta(inception_window[BENCHMARK_INDEX], inception_window[STOCK])

    return [
        date_idx,
        {
            'ols_beta': beta.ols(),
            'bsw_beta': beta.welch(rho=False),
            'bswa_beta': beta.welch(),
            'true_bswa_beta(inception)': inception_beta.welch(),
            'true_bswa_beta(trailing)': trailing_beta.welch(),
        },
    ]

def persist_tickers(tickers):
    filepath = 'data/tickers.txt'
    if not os.path.exists(filepath):
        with open(filepath, 'a+') as f:
            f.write(' '.join(map(str, tickers)))

    return open(filepath, 'r').read()

def main():
    # sample_data = yq.Ticker(f'{STOCK} {BENCHMARK_INDEX}', asynchronous=True).history(period='1y')['adjclose']
    # sample_returns = sample_data.unstack().T.pct_change(fill_method=None).dropna()
    # sample_returns.index = pd.DatetimeIndex(sample_returns.index)
    # rolling_beta_fig(sample_returns)

    # results = dict(map(period_betas_window, sample_returns.groupby(pd.Grouper(freq='M'))))
    # df = pd.DataFrame(results).T
    # fig = df.plot(title=f'{STOCK} rolling betas comparison').update_layout(title_x=0.5, xaxis_title='Date', legend_title='estimators').update_xaxes(dtick='M1').show()


    # beta_series_json_path = 'data/beta_dist-2024-01-18 00:00:00.json'
    # beta_series = pd.read_json(beta_series_json_path, typ='series').rename('beta')
    # beta_series.hist(x='beta', marginal='box', nbins=100).update_traces(
    #     opacity=0.7, selector=dict(type='histogram')
    # ).update_layout(
    #     title_text='Market beta distribution',
    #     title_font=dict(size=24),
    #     title_x=0.5,
    #     bargap=0.1,
    #     xaxis=dict(showgrid=False, showline=True, linecolor='black'),
    #     yaxis=dict(showgrid=False, showline=True, linecolor='black'),
    #     showlegend=False,
    # ).add_vline(x=np.median(beta_series), line_dash='dash', line_width=2).show()
    filepath = 'data/nasdaq_screener_1706639204979.csv'
    df = pd.read_csv(filepath)
    # filter out small, biotech, and warrants or other special instruments
    sample_stocks = df[(df['Country'] == 'United States') & (df['Market Cap'] > 1000000) & (df['Sector'] != 'Health Care') & (df['Symbol'].astype(str).map(len) <= 4)]
    tickers = sample_stocks['Symbol'].values
    persist_tickers(tickers)

    directory = os.fsencode('data')
    series = [] 
    for file in os.scandir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.json'): 
            try:
                with open(f'data/{filename}') as f: 
                    json_series = pd.read_json(f.read(), typ='series')
                    series.append(json_series)
            except ValueError as e:
                continue
    df = pd.DataFrame(series)

if __name__ == '__main__':
    main()

