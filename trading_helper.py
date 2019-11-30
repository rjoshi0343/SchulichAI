import requests
import os
from datetime import datetime, timezone, timedelta
from time import sleep
from binance.client import Client
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


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


def plot_loss_lr(history, fig_name, show=False):

    if show:
        plt.ion()
    else:
        plt.ioff()

    # get index of lowest loss
    minloss_i = np.argmin(history['loss'])
    # get the learning rate at that index
    best_lr = history['lr'][minloss_i]

    fig = plt.figure(figsize=[16, 6])
    ax = fig.add_subplot(121)
    ax.semilogx(history["lr"], history["loss"])
    # get index of lowest loss
    minloss_i = np.argmin(history['loss'])
    # get the learning rate at that index
    min_lr = history['lr'][minloss_i]
    # plot best learing rate, and add value to label
    ax.plot([min_lr, min_lr], ax.get_ylim(), 'r',
            label='best lr: {:.5E}'.format(min_lr))
    ax.set_ylabel('loss')
    ax.set_title('Loss as a function of learning rate')

    ax = fig.add_subplot(122)
    ax.semilogx(history["lr"], history["loss"])
    # ax.set_ylim([0, 1e-4])
    ax.set_xlim([history["lr"][0], 1e-5])
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0] * 1.000025, ylim[-1] * .99933])
    # plot best learing rate, and add value to label
    ax.plot([best_lr, best_lr], ax.get_ylim(), 'r',
            label='best lr: {:.5E}'.format(min_lr))
    # Print the best learning rate
    print('best learning rate:', min_lr)
    ax.set_xlabel('learning rate')
    ax.set_title('Loss as a function of learning rate')
    ax.legend()

    fig.savefig(fig_name)

    if not show:
        plt.close(fig)

    # Print the best learning rate
    print('best learning rate:', min_lr)    


def optimize_lr(model, train_data):
    """
    """
    tf.keras.backend.clear_session()
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-8 * 10**(epoch / 20))

    history = model.fit(train_data, epochs=100, callbacks=[lr_schedule])
    history = history.history
    # get index of lowest loss
    minloss_i = np.argmin(history['loss'])
    # get the learning rate at that index
    best_lr = history['lr'][minloss_i]

    return history, best_lr    


def load_data(batch_sz=64, win_sz=48, frac_train=.66):

    fname = './data/BTCUSD_2h_2011-09-13_to_2019-10-23_bitstamp.csv'
    data = pd.read_csv(fname)
    data.set_index('date', inplace=True)
    # linear interpolation to fill missing values
    data = data.interpolate()
    data['returns'] = data['close'].pct_change(periods=1)

    data['returns'] = 100. * data['returns']

    N = data.shape[0]
    split_t = int(N * frac_train)  # 2/3 to training & 1/3 to validation

    time_train = data.index[1:split_t]  # kip to first value since it is NaN
    x_train = data['returns'][1:split_t].to_numpy()  # first 2/3
    y_train = data['returns'][2:split_t + 1].to_numpy()  # first 2/3
    time_valid = data.index[split_t:]
    x_valid = data['returns'][split_t:-1].to_numpy()  # last 1/3
    y_valid = data['returns'][split_t + 1:].to_numpy()  # last 1/3

    x_train = x_train.reshape((-1, 1))
    x_valid = x_valid.reshape((-1, 1))

    train_data = windowed_dataset(x_train, y_train, win_sz,
                                    batch_sz, kind='regress')
    valid_data = windowed_dataset(x_valid, y_valid, win_sz,
                                    batch_sz, kind='regress')

    return train_data, valid_data


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
    interval = interval.replace('min', 'm')

    interval_s = interval_string_to_seconds(interval)

    if type(start_t) is list:
        dt0 = datetime(*start_t).replace(tzinfo=timezone.utc)
    elif type(start_t) is datetime:
        dt0 = start_t.replace(tzinfo=timezone.utc)
    else:
        raise ValueError('Unsuported type of start_t:', type(start_t))

    ts0 = int(dt0.timestamp() * 1000.)

    if not len(end_t):
        dt1 = datetime.utcnow()
    else:
        dt1 = datetime(*end_t).replace(tzinfo=timezone.utc)

    ts1 = int(dt1.timestamp() * 1000.)

    url = 'https://api.binance.com/api/v3/klines'

    columns = ['open_ts', 'open', 'high', 'low', 'close',
               'volume', 'close_ts', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume',
               'taker_buy_quote_asset_volume', 'ignore']

    responses = []
    dt = dt0
    ts = ts0
    # Done when the last time stamp is less than INTERVAL from end_t
    while (dt1.timestamp() - dt.timestamp()) > interval_s:
    # while (ts1 - ts) > interval_ms:
        response = requests.get(url, params={'symbol': symbol,
                                             'interval': interval,
                                             'startTime': ts,
                                             'endTime': ts1,
                                             'limit': 1000})
        if response.status_code == 200:
            resp = response.json()
            ts = int(resp[-1][6])
            dt = datetime.utcfromtimestamp(ts / 1e3)
            responses.extend(resp)
            if verbose:
                print('\t', dt.strftime('%Y-%m-%d %H:%M'))
        else:
            if verbose:
                print('Binance error:', response.status_code)
                import ipdb
                ipdb.set_trace()

    klines = pd.DataFrame(responses, columns=columns, dtype=np.float64)
    klines['date'] = pd.to_datetime(klines['close_ts'], utc=True, unit='ms')
    klines['returns'] = klines['high'].pct_change(periods=1)
    klines.set_index('date', inplace=True)
    klines.index = klines.index.round(freq='1s')

    klines = klines.drop(['quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume',
                          'taker_buy_quote_asset_volume', 'ignore'], axis=1)

    return klines



def limit_sell(client, pair, price, quantity='all'):
    """
    """
    if quantity == 'all':
        balance = client.get_asset_balance('ENG')
        quantity = np.float64(balance['free'])

    if type(quantity) is str:
        quantity = np.float64(quantity)
    elif type(quantity) not in [np.float, np.float64, int]:
        raise ValueError('Arguemnt "quantity" has wrong type\n'
                         '\ttype(quantity): %s' % type(quantity))

    order = client.order_limit_sell(
        symbol=pair,
        quantity=int(quantity),
        price='%1.8f' % price)

    return order


def limit_buy(client, pair, price, quantity='all'):
    """
    """
    if quantity == 'all':
        balance = client.get_asset_balance('BTC')
        quantity = np.float64(balance['free']) / price

    if type(quantity) is str:
        quantity = np.float64(quantity)
    elif type(quantity) not in [np.float, np.float64, int]:
        raise ValueError('Arguemnt "quantity" has wrong type\n'
                         '\ttype(quantity): %s' % type(quantity))

    order = client.order_limit_buy(
        symbol=pair,
        quantity=int(quantity),
        price='%1.8f' % price)

    return order


def stop_limit_buy(client, pair, limit_price, stop_price, quantity='all'):
    """
    """
    if quantity == 'all':
        balance = client.get_asset_balance('BTC')
        quantity = np.float64(balance['free']) / price

    if type(quantity) is str:
        quantity = np.float64(quantity)
    elif type(quantity) not in [np.float, np.float64, int]:
        raise ValueError('Arguemnt "quantity" has wrong type\n'
                         '\ttype(quantity): %s' % type(quantity))

    order = client.order_limit_buy(
        symbol=pair,
        quantity=int(quantity),
        stopPrice='%1.8f' % stop_price,
        price='%1.8f' % limit_price)

    return order


def check_order(client, pair, orderId):
    """
    """
    order = client.get_order(
        symbol=pair,
        orderId=orderId)

    return order


def cancel_order(client, pair, orderId):
    """
    """
    res = client.cancel_order(
        symbol=pair,
        orderId=orderId)

    return res["status"] == "CANCELED"


def interval_string_to_seconds(trading_interval):

    if 'min' in trading_interval:
        interval_s = int(trading_interval[:-3]) * 60
    elif 'm' in trading_interval:
        interval_s = int(trading_interval[:-1]) * 60        
    elif 'h' in trading_interval:
        interval_s = int(trading_interval[:-1]) * 3600
    elif 'd' in trading_interval:
        interval_s = int(trading_interval[:-1]) * 3600 * 24
    elif 'w' in trading_interval:
        interval_s = int(trading_interval[:-1]) * 3600 * 24 * 7

    return interval_s


def init_client(keys_fname):
    """
    """
    f = open(keys_fname)

    keys = eval(f.read())
    f.close()

    client = Client(keys['api_key'], keys['secret_key'])

    return client


def UTC(kind='time'):
    if kind == 'time':
        return datetime.utcnow()
    elif kind == 'str':
        return datetime.utcnow().strftime('%-d %b, %Y %H:%M')
    elif kind == 'iso':
        return datetime.utcnow().isoformat()
    else:
        raise ValueError('Unknown kind: %s' % kind)


def log(fname, order_type=None, quantity=None, price=np.nan,
        time=None, new_log=False):
    """
    order_type  : str
    quantity    : float
    price       : float
    time        : str
    """

    if new_log:
        f = open(fname, 'w')
        f.write('order_type,quantity,price,time\n')
    else:
        f = open(fname, 'a')
        f.write('%s,%1.8f,%1.8f,%s\n' %
                (order_type, quantity, price, time))

    f.close()
