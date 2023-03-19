from torch.utils.data import DataLoader, Dataset
from smart_open import smart_open
from keys import *
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch


class A2V(Dataset):
    def __init__(self, data, features, descriptions, back_length, mask_p):
        assert data.shape[0] == features.shape[0], 'Data and features must have the same number of rows'

        self.stocks = list(data.columns)
        self.descriptions = descriptions
        self.col_list = list(features.columns)
        self.counts = [data.shape[0] - count for count in data.count()]
        self.df = pd.concat([data, features], axis=1)
        self.back_length = back_length
        self.mask_p = mask_p
        range_list = [[(i, i + back_length) for i in np.arange(start, data.shape[0], back_length)[:-1]] +
                      [(data.shape[0] - back_length, data.shape[0])] for start in self.counts]

        self.range_list = []
        for i, r in enumerate(range_list):
            self.range_list += zip([data.columns[i]] * len(r), r)

    def __len__(self):
        return len(self.range_list)

    def __getitem__(self, idx):
        ticker, r = self.range_list[idx]
        i, j = r
        idxs = self.df.index[i:j]
        cols = [ticker] + self.col_list
        x = torch.from_numpy(self.df.loc[idxs, cols].to_numpy(copy=True))
        mask = np.hstack((np.random.choice([False, True], size=(x.shape[0], 1), p=[1 - self.mask_p, self.mask_p]),
                          np.random.choice([False, True], size=(x.shape[0], x.shape[1] - 1),
                                           p=[1 - self.mask_p, self.mask_p])))
        mask = torch.from_numpy(mask)
        mask2 = ~mask.clone()
        mask2[:, 1:] = True
        desc = self.descriptions[ticker]
        return x, mask, mask2, desc['input_ids'][0], desc['attention_mask'][0], self.stocks.index(ticker)


class A2VDataModule(pl.LightningDataModule):
    def __init__(self, back_length, val_frac, batch_size, tokenizer, mask_p, path=S3_PATH):
        super().__init__()
        self.back_length = back_length
        self.val_frac = val_frac
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.mask_p = mask_p
        self.path = path

        self.econ = None
        self.metadata = None
        self.stocks_train = None
        self.stocks_val = None

    def prepare_data(self):
        self.econ = pd.read_csv(smart_open(self.path + 'data/economic_factors.csv'), index_col=0).astype(np.float32)
        self.metadata = pd.read_csv(smart_open(self.path + 'data/t2k_metadata.csv'), index_col=0, dtype=str)
        stocks = pd.read_parquet(smart_open(self.path + 'data/t2k_returns.parquet')).astype(np.float32)
        stocks.index = stocks.index.astype(str)
        num_train = int(stocks.shape[1] * (1 - self.val_frac))
        train_tickers = np.random.choice(list(stocks.columns), num_train, replace=False)
        val_tickers = list(stocks.columns.difference(train_tickers))
        self.stocks_train = stocks[train_tickers]
        self.stocks_val = stocks[val_tickers]

    def setup(self, stage):
        self.metadata.loc['Description'] = self.metadata.loc['Description'].apply(
            self.tokenizer, max_length=512, return_tensors='pt', padding='max_length', return_attention_mask=True, is_split_into_words=False
        )

    def train_dataloader(self):
        train_set = A2V(self.stocks_train, self.econ, self.metadata.loc['Description'], self.back_length, self.mask_p)
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=4, drop_last=True)

    def val_dataloader(self):
        val_set = A2V(self.stocks_val, self.econ, self.metadata.loc['Description'], self.back_length, self.mask_p)
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=4, drop_last=True)
