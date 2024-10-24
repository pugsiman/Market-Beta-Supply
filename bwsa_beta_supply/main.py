#!/usr/bin/env python3

import numpy as np
import json
from scipy.stats import linregress
from scipy.stats import zscore

import yahooquery as yq
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


from datetime import datetime
import os
import re
from utils.beta import Beta

pd.options.plotting.backend = 'plotly'

BENCHMARK_INDEX = 'SPY'
STOCK = ''
JUNK_STOCKS = 'IONQ BBAI BTBT OPEN JOBY AFRM BYND SHOT NEWT BEEM APLD AAOI'
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


def main():
    directory = os.fsencode('data')
    series = []
    for file in os.scandir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.json'):
            try:
                with open(filename) as f:
                    index = re.search('(?<=-).+(?= )', f.name).group()  # extract date
                    data = json.load(f)
                    # backwards compatiability for data sets that didn't have residuals calculated
                    if 'values' in data:
                        json_series = pd.Series(data['values'])
                    else:
                        json_series = pd.Series(data)

                    series.append(json_series.rename(index))
            except ValueError:
                continue

    df = pd.DataFrame(series).sort_index().T
    beta_supply = ((df[df > 1.9].count() / df.count()) * 100).to_frame(
        name='supply_count'
    )
    beta_dispersion = (
        df.where(df.gt(df.quantile(0.9))).stack().groupby(level=1).agg('mean')
    ) - df.where(df.lt(df.quantile(0.1))).stack().groupby(level=1).agg('mean')
    beta_dispersion_trace = beta_dispersion.to_frame(name='beta_dispersion').plot()

    beta_supply['short_slope'] = beta_supply.rolling(7).apply(
        lambda s: linregress(range(len(s)), s)[0]
    )
    slope_deriv = np.gradient(beta_supply['short_slope'])
    infls = np.where(np.diff(np.sign(slope_deriv)))[0]

    short_trace = go.Figure(
        data=go.Scatter(
            x=beta_supply.index,
            y=beta_supply['short_slope'].values,
            name='7-day change',
        )
    )

    figs = (
        go.Figure(
            data=short_trace.data
            + beta_supply['supply_count'].plot().data
            + beta_dispersion_trace.data
        )
        .update_layout(
            title=dict(text='Beta supply', font_size=24, x=0.5),
            xaxis=dict(
                ticklabelmode='period',
                dtick='M1',
                showline=True,
                showgrid=False,
                type='date',
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all'),
                        ]
                    ),
                    font_color='black',
                    activecolor='gray',
                ),
                rangeslider=dict(visible=True),
            ),
            yaxis=dict(showline=True, showgrid=False),
            template='plotly_dark',
            legend=dict(x=0.1, y=1.1, orientation='h', font=dict(color='#FFFFFF')),
            yaxis_title=dict(
                text='high beta supply rate', font=dict(size=16, color='#FFFFFF')
            ),
        )
        .add_hline(
            y=-1 * beta_supply['short_slope'].std(), line_dash='dash', line_width=1
        )
        .add_hline(
            y=-2 * beta_supply['short_slope'].std(), line_dash='dash', line_width=1
        )
    )

    for _, infl in enumerate(infls, 1):
        if (
            beta_supply['short_slope'].iloc[infl]
            < -1 * beta_supply['short_slope'].std()
        ):
            figs.add_vline(
                x=beta_supply['short_slope'].index[infl + 1],
                opacity=0.4,
                line_color='green',
            )

    figs.update_traces(
        texttemplate='%{y:.2f}',
        textposition='top center',
        hovertemplate='Date: %{x}<br>Value: %{y:.2f}<br>',
        marker_line_color='#800020',
        marker_line_width=1.5,
    )

    figs.show()


if __name__ == '__main__':
    main()
