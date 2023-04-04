from tqdm import tqdm
from keys import *
import pandas as pd
import numpy as np
import yfinance as yf


def get_data():
    stock_list = sorted(list(pd.read_csv('stock_tickers.csv', header=None)[0]))

    for stock in tqdm(stock_list):
        prices = yf.download(stock, start='2001-01-01', end='2022-11-01', progress=False)['Adj Close']
        if stock[0] is None:
            continue

        if stock == stock_list[0]:
            df = prices

        else:
            df = pd.concat((df, prices), axis=1)

    df.columns = stock_list
    return df.astype(np.float32).pct_change()[1:]


if __name__ == '__main__':
    returns = get_data()
    returns.to_parquet('t2k_returns.parquet')
