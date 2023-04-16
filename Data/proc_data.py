import pandas as pd
import numpy as np


VOL_LOOKBACK = 60
VOL_TARGET = 0.15
TIME_TRAIN_FRAC = 0.6
STOCKS_TRAIN_FRAC = 0.6


def get_multiindex(name, stocks):
    names = [name] * len(stocks)
    return pd.MultiIndex.from_arrays([names, stocks])


def calc_returns(df, offset):
    return df.pct_change(offset)


def calc_daily_vol(daily_returns):
    return (
        daily_returns.ewm(span=VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
        .std()
        .fillna(method='bfill')
    )


def calc_vol_scaled_returns(daily_returns):
    daily_vol = calc_daily_vol(daily_returns) * np.sqrt(252)
    return daily_returns * VOL_TARGET / daily_vol


def merge_columns(df1, df2, name, stocks):
    df2.columns = get_multiindex(name, stocks)
    return pd.concat([df1, df2], axis=1)


def main(df_raw):
    prices = df_raw.loc[:, 'Adj Close']
    stocks_list = prices.columns.to_list()
    daily_returns = calc_returns(prices, 1)
    daily_vol = calc_daily_vol(daily_returns)
    df_raw = merge_columns(df_raw, calc_vol_scaled_returns(daily_returns).shift(-1), 'target', stocks_list)

    def calc_normalised_returns(day_offset):
        return (
                calc_returns(prices, day_offset)
                / daily_vol
                / np.sqrt(day_offset)
        )

    df_raw = merge_columns(df_raw, calc_normalised_returns(1), 'norm_daily_return', stocks_list)
    df_raw = merge_columns(df_raw, calc_normalised_returns(21), 'norm_monthly_return', stocks_list)
    df_raw = merge_columns(df_raw, calc_normalised_returns(63), 'norm_quarterly_return', stocks_list)
    df_raw = merge_columns(df_raw, calc_normalised_returns(126), 'norm_biannual_return', stocks_list)
    df_raw = merge_columns(df_raw, calc_normalised_returns(252), 'norm_annual_return', stocks_list)

    return df_raw


if __name__ == '__main__':
    df = pd.read_parquet('raw.parquet')
    df = main(df).iloc[252:-1].swaplevel(axis=1).sort_index(axis=1)
    df.to_parquet('data.parquet')
    counts = pd.read_csv('counts.csv', index_col=0)
    time_cutoff = int(len(df) * TIME_TRAIN_FRAC)
    train_ticker = np.random.choice(counts[(counts == counts.max()).values].index, int(len(counts) * STOCKS_TRAIN_FRAC), replace=False)
    test_ticker = counts[~counts.index.isin(train_ticker)].index
    df.iloc[:time_cutoff][train_ticker].to_parquet('train.parquet')
    df.iloc[time_cutoff:][test_ticker].to_parquet('test.parquet')
