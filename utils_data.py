from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import RobertaTokenizer
from smart_open import smart_open
from keys import *
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch


def sharpe_loss(weights, returns, perfect=False):
    if perfect:
        weights = returns.clone().detach()
        weights[weights < 0] = -1
        weights[weights > 0] = 1

    returns = weights * returns * 100
    sharpe = torch.mean(returns) / torch.std(returns)

    return -sharpe


class A2V(Dataset):
    def __init__(self, data, features, back_length, forward_length, embeddings):
        assert data.shape[0] == features.shape[0], 'Data and features must have the same number of rows'

        self.back_length = back_length
        self.forward_length = forward_length
        self.embeddings = embeddings
        self.col_list = list(features.columns)
        self.counts = [data.shape[0] - count for count in data.count()]
        self.returns = data.copy(deep=True)
        data = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)
        self.df = pd.concat([data, features], axis=1)
        self.back_length = back_length
        range_list = [np.arange(start, data.shape[0] - back_length - forward_length - 1)
                      for start in self.counts]

        self.weights = []
        self.range_list = []
        for i, r in enumerate(range_list):
            self.range_list += zip([data.columns[i]] * len(r), r)
            self.weights += [1 / len(r)] * len(r)

    def __len__(self):
        return len(self.range_list)

    def __getitem__(self, idx):
        ticker, i = self.range_list[idx]
        j, k = i + self.back_length, i + self.back_length + self.forward_length
        cols = [ticker] + self.col_list
        x = torch.from_numpy(self.df.iloc[i:j][cols].to_numpy())
        y = torch.from_numpy(self.returns.iloc[j:k][ticker].to_numpy())
        embd = torch.from_numpy(self.embeddings.loc[:, ticker].to_numpy())
        return x, y, embd


class A2VDataModule(pl.LightningDataModule):
    def __init__(self, back_length, forward_length, val_frac, batch_size, total_train, path=S3_PATH):
        super().__init__()
        np.random.seed(42)
        self.back_length = back_length
        self.forward_length = forward_length
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.total_train = total_train
        self.path = path
        self.embeddings = PretrainedEmbeddings().embeddings

        self.econ = None
        self.date_split = None
        self.stock_list = None
        self.stocks_train = None
        self.stocks_val = None

    def prepare_data(self):
        self.econ = pd.read_csv(smart_open(self.path + 'data/economic_factors.csv'), index_col=0).astype(np.float32)
        stocks = pd.read_parquet(smart_open(self.path + 'data/t2k_returns_embd.parquet')).astype(np.float32)
        self.econ = self.econ.loc[stocks.index]
        self.stock_list = list(stocks.columns)
        stocks.index = stocks.index.astype(str)
        num_train = int(stocks.shape[1] * (1 - self.val_frac))
        train_tickers = np.random.choice(list(stocks.loc[:, stocks.count() > 1000].columns), num_train, replace=False)
        val_tickers = np.random.choice(list(stocks.columns.difference(train_tickers)), 50, replace=False)
        self.date_split = int(len(stocks) * self.val_frac)
        self.stocks_train = stocks[train_tickers]
        self.stocks_val = stocks[val_tickers]

    def train_dataloader(self):
        train_set = A2V(self.stocks_train, self.econ, self.back_length, self.forward_length, self.embeddings)
        sampler = WeightedRandomSampler(train_set.weights, self.total_train * self.batch_size, False)
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=4, drop_last=True, sampler=sampler)

    def val_dataloader(self):
        val_set = A2V(self.stocks_val, self.econ, self.back_length, self.forward_length, self.embeddings)
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=4, drop_last=True)


class Descriptions:
    def __init__(self, path=S3_PATH):
        self.tokenizer = RobertaTokenizer.from_pretrained('tokenizer/')
        self.metadata = pd.read_parquet(smart_open(path + 'data/t2k_metadata.parquet')).astype(str)
        self.tokenized = self.metadata.loc['Description'].apply(
            self.tokenizer, max_length=512, return_tensors='pt', padding='max_length',
            return_attention_mask=True, is_split_into_words=False
        )


class PretrainedEmbeddings:
    def __init__(self, version, path=S3_PATH):
        self.embeddings = pd.read_parquet(smart_open(path + f'data/v{version}.parquet')).astype(np.float32)
