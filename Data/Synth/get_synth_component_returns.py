from tqdm import tqdm
from keys import *
import pandas as pd
import quandl


def get_data():
    quandl.ApiConfig.api_key = QUANDL_KEY
    stock_list = sorted(list(pd.read_csv('synth_components.csv', index_col=0)['ticker']))

    for stock in tqdm(stock_list):
        prices = quandl.get_table('QUOTEMEDIA/PRICES', ticker=stock, qopts={'columns': ['adj_close']},
                                  paginate=True)[:5842][::-1]

        if stock[0] is None or len(stock) > 5843:
            continue

        if stock == stock_list[0]:
            df = prices

        else:
            df = pd.concat((df, prices), axis=1)

    df.columns = stock_list
    df.reset_index(drop=True)
    return df.pct_change()[1:]


if __name__ == '__main__':
    returns = get_data()
    returns.to_csv('synth_component_returns.csv')
