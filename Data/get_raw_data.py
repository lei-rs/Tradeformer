import pandas as pd
import yfinance as yf


def get_raw_data():
    idx = pd.IndexSlice
    sl = pd.read_csv('stock_tickers.csv').sort_values('Market Cap', ascending=False).head(1000)['Symbol'].tolist()
    data = yf.download(sl, start='2002-01-01', end='2023-01-01', threads=8)[['Adj Close', 'Volume']]
    counts = data['Adj Close'].count()
    data = data.drop(columns=counts[counts < 652].index.tolist(), level=1)
    data.to_parquet('data.parquet')
    counts = data['Adj Close'].count()
    counts.to_csv('counts.csv')


if __name__ == '__main__':
    get_raw_data()
