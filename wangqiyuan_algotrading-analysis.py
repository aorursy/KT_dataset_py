import random

import os

from datetime import datetime

from collections import OrderedDict

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from tqdm.notebook import tqdm, trange
data_path = '../input/price-volume-data-for-all-us-stocks-etfs/Data/Stocks/'

file_names = [name for name in os.listdir(data_path) if os.stat(data_path + name).st_size > 0]

print(random.sample(file_names, 4))
file_suffix = '.us.txt'

tickers = [file_name.replace(file_suffix, '') for file_name in file_names]

print(random.sample(tickers, 4))
example_ticker = 'wafd'

example_stock = pd.read_csv(data_path + example_ticker + file_suffix)

example_stock.head()
class Stock:

    def __init__(self, dates, prices):

        self.dates = dates

        self.prices = prices



    def __getitem__(self, idx):

        return type(self)(self.dates[idx], self.prices[idx])

    

    def __len__(self):

        return len(self.dates)



    @classmethod

    def read(cls, ticker):

        original = pd.read_csv(data_path + ticker + file_suffix)

        dates = original['Date'].map(lambda t: datetime.strptime(t, '%Y-%m-%d')).to_numpy()

        prices = original['Open'].to_numpy()

        return cls(dates, prices)
stocks = OrderedDict((ticker, Stock.read(ticker)) for ticker in tqdm(tickers))
def plot(contents, ticker, *args, **kwargs):

    plt.figure(figsize=(12, 8))

    plt.title(f'{ticker.upper()} price')

    contents(ticker, *args, **kwargs)

    plt.legend()

    plt.xlabel('date')

    plt.ylabel('price [$]')

    plt.show()
def plot_price(ticker, **kwargs):

    plt.plot(stocks[ticker].dates, stocks[ticker].prices,

             label='price', c='dodgerblue', zorder=0)
plot(plot_price, example_ticker)
def moving_average(signal, period):

    cumsum = np.cumsum(signal)

    result = (cumsum[period:] - cumsum[:-period]) / period

    filler = np.full(period, np.nan)

    return np.concatenate((filler, result))
def plot_moving_average(ticker, period):

    plot_price(ticker)

    plt.plot(stocks[ticker].dates, moving_average(stocks[ticker].prices, period),

             label=f'{period}-day moving average', c='indigo', zorder=1)
plot(plot_moving_average, example_ticker, 30)
def bands(signal, period, power=2, interval=2, log=False):

    if log:

        signal = np.log(signal)

    average = moving_average(signal, period)

    deviation = np.abs(signal - average) ** power

    band_width = moving_average(deviation[period:], period) ** (1/power) * interval

    filler = np.full(period, np.nan)

    filled_width = np.concatenate((filler, band_width))

    low, high = average - filled_width, average + filled_width

    return (np.exp(low), np.exp(high)) if log else (low, high)
def plot_bands(ticker, period, **bands_kwargs):

    plot_moving_average(ticker, period)

    low_band, high_band = bands(stocks[ticker].prices, period, **bands_kwargs)

    plt.fill_between(stocks[ticker].dates, low_band, high_band,

                     label=f'{period}-day band', color='pink', zorder=-1)
plot(plot_bands, example_ticker, period=30)
def keep_strategy(prices):

    '''A baseline trading strategy: buy at the beginning and sell at the end.'''

    return np.array([0]), np.array([len(prices) - 1]) # buys, sells
def band_strategy(prices, **bands_kwargs):

    '''A trading strategy based on bands: when price falls below bands, buy, when price rises above bands, sell.'''

    low_band, high_band = bands(prices, **bands_kwargs)

    buys = []

    sells = []

    for idx, (price, low, high) in enumerate(zip(prices, low_band, high_band)):

        if price < low and len(buys) <= len(sells):

            buys.append(idx)

        elif price > high and len(buys) > len(sells):

            sells.append(idx)

    return np.array(buys, dtype=int), np.array(sells, dtype=int)
def gain_strategy(prices, buy_ratio=0.9, sell_ratio=1.1):

    '''

    Start by buying stock.

    Sell when it is worth more than sell_ratio times the buy price.

    Buy again when it is worth less than buy_ratio times the sell price.

    '''

    buys = [0]

    sells = []

    action_price = prices[0] * sell_ratio

    for idx, price in enumerate(prices):

        if len(sells) < len(buys):

            if price >= action_price:

                sells.append(idx)

                action_price = price * buy_ratio

        else:

            if price <= action_price:

                buys.append(idx)

                action_price = price * sell_ratio

    return np.array(buys, dtype=int), np.array(sells, dtype=int)
def roi(prices, buys, sells):

    if len(sells) < len(buys):

        sells = np.append(sells, -1) # simulate selling at the last price

    return np.prod(prices[sells] / prices[buys]) - 1
def plot_trades(ticker, background, strategy, **kwargs):

    background(ticker, **kwargs)

    buys, sells = strategy(stocks[ticker].prices, **kwargs)

    roi_value = roi(stocks[ticker].prices, buys, sells)

    plt.title(f'{ticker.upper()} trades using {strategy.__name__}, ROI: {roi_value}')

    plt.scatter(stocks[ticker].dates[buys], stocks[ticker].prices[buys],

                label='buys', marker='x', c='red', s=256, zorder=2)

    plt.scatter(stocks[ticker].dates[sells], stocks[ticker].prices[sells],

                label='sells', marker='x', c='lime', s=256, zorder=2)
plot(plot_trades, example_ticker, background=plot_price, strategy=keep_strategy)
plot(plot_trades, example_ticker, background=plot_bands, strategy=band_strategy, period=30)
plot(plot_trades, example_ticker, background=plot_price, strategy=gain_strategy)
def jumps_ok(prices, max_jump):

    jumps = prices[1:] / prices[:-1]

    return np.max(jumps) <= max_jump and np.max(1/jumps) <= max_jump



def filter_stocks(stocks, start_date=None, end_date=None, min_length=256, min_price=0.01, max_jump=8):

    masks = {ticker: np.full(len(stocks[ticker]), True) for ticker in stocks}

    if start_date is not None:

        masks = {ticker: np.logical_and(masks[ticker], start_date <= stocks[ticker].dates) for ticker in stocks}

    if end_date is not None:

        masks = {ticker: np.logical_and(masks[ticker], stocks[ticker].dates < end_date) for ticker in stocks}

    filtered = {ticker: stocks[ticker][masks[ticker]] for ticker in stocks}

    filtered = {ticker: filtered[ticker] for ticker in filtered if len(filtered[ticker]) >= min_length}

    filtered = {ticker: filtered[ticker] for ticker in filtered if min(filtered[ticker].prices) >= min_price}

    filtered = {ticker: filtered[ticker] for ticker in filtered if jumps_ok(filtered[ticker].prices, max_jump)}

    return OrderedDict(filtered)
training = filter_stocks(stocks, end_date=np.datetime64('2015-01-01'))

validation = filter_stocks(stocks, start_date=np.datetime64('2016-01-01'))

print(f'{len(training)} companies in training, {len(validation)} in validation')
def strategy_rois(stocks, strategy, **kwargs):

    result = []

    for ticker in stocks:

        buys, sells = strategy(stocks[ticker].prices, **kwargs)

        result.append(roi(stocks[ticker].prices, buys, sells))

    return np.array(result)



def performance(rois, bias=2):

    '''De facto weighted average of ROIs, which gives less weight to very high ones.'''

    return np.exp(np.mean(np.log(rois + bias))) - bias
def search_params(strategy, num_epochs, param_generator):

    results = []

    for epoch in trange(num_epochs):

        params = param_generator()

        rois = strategy_rois(training, strategy, **params)

        perf = performance(rois)

        results.append({'params': params, 'rois': rois, 'perf': perf})

    return results
def band_params():

    return {

        'period': int(2 ** np.random.uniform(2, 6)),

        'power': 2 ** np.random.uniform(-2, 2),

        'interval': 2 ** np.random.uniform(0, 1),

        'log': random.choice([True, False])

    }



band_results = search_params(band_strategy, 256, band_params)
def gain_params():

    return {

        'buy_ratio': 1 - 2 ** np.random.uniform(-8, -1),

        'sell_ratio': 1 + 2 ** np.random.uniform(-8, -1)

    }



gain_results = search_params(gain_strategy, 256, gain_params)
def span_years(dates):

    timespan = dates[-1] - dates[0]

    return timespan / np.timedelta64(1, 'Y').astype(timespan.dtype)
def roi_hist(strategy, stocks, **kwargs):

    rois = strategy_rois(stocks, strategy, **kwargs)

    spans = [span_years(stocks[ticker].dates) for ticker in stocks]

    yearly = [(roi + 1) ** (1/span) - 1 for roi, span in zip(rois, spans)]

    plt.figure(figsize=(12, 8))

    plt.title(f'yearly ROI histogram for {strategy.__name__}')

    plt.hist(yearly, bins=np.linspace(np.quantile(yearly, .01), np.quantile(yearly, .99), 256))

    plt.axvline(np.median(yearly), color='orange', label=f'median: {np.median(yearly)}')

    plt.axvline(np.mean(yearly), color='violet', label=f'mean: {np.mean(yearly)}')

    plt.xlabel('yearly ROI')

    plt.ylabel('number of companies')

    plt.legend()

    plt.show()
roi_hist(keep_strategy, validation)
def evaluate_results(search_results, strategy, background_plot):

    best_result = max(search_results, key=lambda result: result['perf'])

    print(f'best params for {strategy.__name__}: {best_result["params"]}')

    roi_hist(strategy, validation, **best_result['params'])

    plot(plot_trades, example_ticker, background=background_plot, strategy=strategy, **best_result['params'])

    return best_result
best_band = evaluate_results(band_results, band_strategy, plot_bands)
best_gain = evaluate_results(gain_results, gain_strategy, plot_price)
plt.figure(figsize=(12, 8))

plt.title('performance histogram of band strategies')

plt.xlabel('performance')

plt.ylabel('number of strategies')

plt.hist([result['perf'] for result in band_results], bins=64)

plt.legend()

plt.show()
for param_name in ['period', 'power', 'interval']:

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for idx, ax in enumerate(axes):

        log = bool(idx)

        ax.set_title(f'performance vs {param_name}, log={log}')

        ax.set_xlabel(f'{param_name}')

        ax.set_ylabel('performance')

        ax.set_yscale('symlog')

        fitting_log = [result for result in band_results if result['params']['log'] == log]

        ax.scatter(

            [result['params'][param_name] for result in fitting_log],

            [result['perf'] for result in fitting_log]

        )

    plt.show()
def trade_count(strategy, stocks, **kwargs):

    return sum(len(np.concatenate(strategy(stocks[ticker].prices, **kwargs))) for ticker in stocks)



plt.figure(figsize=(12, 8))

plt.title('number of trades vs. performance, by strategy')

plt.xlabel('number of trades')

plt.ylabel('performance')

plt.yscale('symlog')

plt.scatter(

    [trade_count(band_strategy, validation, **result['params']) for result in tqdm(band_results)],

    [result['perf'] for result in band_results]

)

plt.show()
best_band_trades = {ticker: band_strategy(validation[ticker].prices, **best_band['params']) for ticker in validation}



plt.figure(figsize=(12, 8))

plt.title('best strategy ROI, by company')

plt.xlabel('number of trades')

plt.ylabel('ROI')

plt.yscale('symlog')

plt.scatter(

    [len(np.concatenate(best_band_trades[ticker])) for ticker in validation],

    [roi(validation[ticker].prices, *best_band_trades[ticker]) for ticker in validation]

)

plt.show()
validation_2016 = filter_stocks(

    stocks,

    start_date=np.datetime64('2016-01-01'),

    end_date=np.datetime64('2017-01-01'),

    min_length=best_band['params']['period'] * 2

)

validation_2017 = filter_stocks(

    stocks,

    start_date=np.datetime64('2017-01-01'),

    end_date=np.datetime64('2018-01-01'),

    min_length=best_band['params']['period'] * 2

)

# filter so that only companies in both are kept, arrange in the same order

validation_2016 = OrderedDict((ticker, validation_2016[ticker]) for ticker in validation_2016 if ticker in validation_2017)

validation_2017 = OrderedDict((ticker, validation_2017[ticker]) for ticker in validation_2016 if ticker in validation_2016)



rois_2016 = strategy_rois(validation_2016, band_strategy, **best_band['params'])

rois_2017 = strategy_rois(validation_2017, band_strategy, **best_band['params'])



plt.figure(figsize=(12, 8))

plt.title('predictive power of performance in prevous year')

plt.xlabel('ROI in 2016')

plt.xscale('symlog')

plt.ylabel('ROI in 2017')

plt.yscale('symlog')

plt.scatter(rois_2016, rois_2017)

plt.show()
def roi_by_step(prices, strategy, **kwargs):

    buys, sells = strategy(prices, **kwargs)

    buy_mask = np.full(len(prices), False)

    buy_mask[buys] = True

    sell_mask = np.full(len(prices), False)

    sell_mask[sells] = True

    last_sell_ratio = 1.0

    owned = False

    ratios = []

    for price, is_buy, is_sell in zip(prices, buy_mask, sell_mask):

        if is_buy:

            owned = True

            buy_price = price

        elif is_sell:

            owned = False

            last_sell_ratio = price / buy_price * last_sell_ratio

        if owned:

            ratios.append(price / buy_price * last_sell_ratio)

        else:

            ratios.append(last_sell_ratio)

    return np.array(ratios) - 1
plt.figure(figsize=(12, 8))

plt.title(f'sanity check: ROI over time when trading {example_ticker.upper()}')

plt.plot(

    stocks[example_ticker].dates,

    roi_by_step(stocks[example_ticker].prices, band_strategy, **best_band['params'])

)

plt.show()
def mean_roi_by_date(stocks, strategy, **kwargs):

    min_date = min(np.min(stocks[ticker].dates) for ticker in stocks)

    max_date = max(np.max(stocks[ticker].dates) for ticker in stocks)

    span_days = int((max_date - min_date).astype('timedelta64[D]').astype(int))

    all_dates = [min_date + np.timedelta64(i, 'D') for i in range(span_days + 1)]

    

    aggregator = OrderedDict((date, {}) for date in all_dates)

    for ticker in tqdm(stocks):

        rois = roi_by_step(stocks[ticker].prices, strategy, **kwargs)

        for date, roi in zip(stocks[ticker].dates, rois):

            aggregator[date][ticker] = roi

    

    # fill missing values with last available value

    last_value = {ticker: 0.0 for ticker in stocks}

    for date in aggregator:

        for ticker in stocks:

            if ticker not in aggregator[date]:

                aggregator[date][ticker] = last_value[ticker]

            else:

                last_value[ticker] = aggregator[date][ticker]

    

    return all_dates, [np.mean([aggregator[date][ticker] for ticker in stocks]) for date in aggregator]





plt.figure(figsize=(12, 8))

plt.title('mean ROI over time, by strategy')

plt.xlabel('date')

plt.ylabel('mean ROI over all stocks')

plt.yscale('symlog')

plt.plot(*mean_roi_by_date(validation, keep_strategy), label='keep')

plt.plot(*mean_roi_by_date(validation, band_strategy, **best_band['params']), label='band')

plt.plot(*mean_roi_by_date(validation, gain_strategy, **best_gain['params']), label='gain')

plt.legend()

plt.show()