import os
import pandas as pd
from datetime import datetime, timezone
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

symbol = 'BTCUSDT'


def get_klines(start_t=[2019, 10, 15, 0, 0, 0], end_t=[],
               symbol='BTCUSDT', interval='30m', verbose=True):
    """

    start_t and end_t : [year, month, day, hour, minute, second]

    response:
    [
       [
         1499040000000,      // Open time
         "0.01634790",       // Open
        "0.80000000",        // High
         "0.01575800",       // Low
         "0.01577100",       // Close
         "148976.11427815",  // Volume
         1499644799999,      // Close time
         "2434.19055334",    // Quote asset volume
         308,                // Number of trades
         "1756.87402397",    // Taker buy base asset volume
         "28.46694368",      // Taker buy quote asset volume
         "17928899.62484339" // Ignore.
       ]
     ]

    See the Binance API for details:
    https://github.com/binance-exchange/binance-official-api-docs/blob/master/rest-api.md#enum-definitions
    """

    # A hack to convert 30min (30 minutes) to milliseconds
    interval_ms = 30 * 60 * 1e3

    dt0 = datetime(*start_t).replace(tzinfo=timezone.utc)
    ts0 = int(dt0.timestamp() * 1000.)

    if not len(end_t):
        dt1 = datetime.now().replace(tzinfo=timezone.utc)
    else:
        dt1 = datetime(*end_t).replace(tzinfo=timezone.utc)

    ts1 = int(dt1.timestamp() * 1000.)

    url = 'https://api.binance.com/api/v3/klines'

    columns = ['open_ts', 'open', 'high', 'low', 'close',
               'volume', 'close_ts', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume',
               'taker_buy_quote_asset_volume', 'ignore']

    responses = []
    ts = ts0
    # Done when the last time stamp is less than INTERVAL from end_t
    while (ts1 - ts) > interval_ms:
        response = requests.get(url, params={'symbol': symbol,
                                             'interval': interval,
                                             'startTime': ts,
                                             'endTime': ts1,
                                             'limit': 1000})
        if response.status_code == 200:
            resp = response.json()
            ts = int(resp[-1][6])
            responses.extend(resp)
            if verbose:
                print(datetime.utcfromtimestamp(
                    ts / 1e3).strftime('%Y-%m-%d %H:%M'))

    klines = pd.DataFrame(responses, columns=columns, dtype=np.float64)
    klines['date'] = pd.to_datetime(klines['close_ts'], utc=True, unit='ms')
    klines['returns'] = klines['high'].pct_change(periods=1)
    klines.set_index('date', inplace=True)

    klines = klines.drop(['quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume',
                          'taker_buy_quote_asset_volume', 'ignore'], axis=1)

    return klines


def load_binance_klines(fname, date_as_index=False):
    """
    """
    columns = ['date', 'open_ts', 'open', 'high', 'low', 'close',
               'volume', 'close_ts', 'returns']

    klines = pd.read_csv(fname, skiprows=1, names=columns)
    if date_as_index:
        klines.set_index('date', inplace=True)

    return klines


def windowed_dataset_numpy(x, y, win_sz, shuffle=True, kind='regress'):
    """
    Returns data in numpy arrays that can be used for non tensorflow models.
    """
    windowed_x = np.empty((y.shape[0], win_sz, x.shape[1]))
    windowed_y = np.empty((y.shape))
    for i in range(y.shape[0]-win_sz):
        windowed_x[i] = x[i:i+win_sz]
        windowed_y[i] = y[i+win_sz-1]

    windowed_x = windowed_x[:i]
    windowed_y = windowed_y[:i]

    if shuffle:
        shuffle_ids = np.random.permutation(np.arange(i))
        windowed_x = windowed_x[shuffle_ids]
        windowed_y = windowed_y[shuffle_ids]

    return windowed_x, windowed_y


def windowed_dataset(x, y, win_sz, batch_sz, kind='regress'):
    """
    Helper to prepare a windowed data set from a series

    kind : "regress" or "class"
    """

    if kind == 'class':
        # to class labels
        y = y > 0

    dataset = TimeseriesGenerator(x, y, win_sz,
                                  sampling_rate=1,
                                  shuffle=True,
                                  batch_size=batch_sz)
    return dataset
