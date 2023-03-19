from tqdm import tqdm
from keys import *
import pandas as pd
import numpy as np
import quandl


def get_data():
    quandl.ApiConfig.api_key = QUANDL_KEY
    stock_list = sorted(list(pd.read_csv('stock_tickers.csv', header=None)[0]))

    for stock in tqdm(stock_list):
        prices = quandl.get_table('QUOTEMEDIA/PRICES', ticker=stock, qopts={'columns': ['date', 'adj_close']},
                                  paginate=True).set_index('date').sort_index().loc['2001-01-01':'2022-10-31'][::-1]
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
