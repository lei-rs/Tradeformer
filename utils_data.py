from torch.utils.data import DataLoader, Dataset, RandomSampler
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
    sharpe = torch.mean(returns) * np.sqrt(252) / torch.std(returns)

    return -sharpe


class DS(Dataset):
    def __init__(self, data, back_length, embeddings):
        idx = pd.IndexSlice
        col_list = ['norm_daily_return',
                    'norm_monthly_return',
                    'norm_quarterly_return',
                    'norm_biannual_return',
                    'norm_annual_return']
        self.data = data.loc[:, idx[:, col_list]]
        self.target = data.loc[:, idx[:, 'target']].droplevel(1, axis=1)
        self.back_length = back_length
        self.embeddings = embeddings
        counts = data.loc[:, idx[:, 'norm_annual_return']].count()
        counts = [data.shape[0] - count for count in counts]
        range_list = [np.arange(start, data.shape[0] - back_length) for start in counts]
        self.range_list = []
        for i, r in enumerate(range_list):
            self.range_list += zip([self.target.columns[i]] * len(r), r)

    def __len__(self):
        return len(self.range_list)

    def __getitem__(self, idx):
        ticker, i = self.range_list[idx]
        j = i + self.back_length
        x = torch.from_numpy(self.data.iloc[i:j][ticker].values)
        y = torch.Tensor([self.target.iloc[j-1][ticker]])
        #embd = torch.from_numpy(self.embeddings.loc[ticker].values)
        return x, y, ticker


class DataModule(pl.LightningDataModule):
    def __init__(self, back_length, num_val, batch_size, total_train, path=S3_PATH):
        super().__init__()
        np.random.seed(42)
        self.back_length = back_length
        self.num_val = num_val
        self.batch_size = batch_size
        self.total_train = total_train
        self.path = path
        self.embeddings = PretrainedEmbeddings(1).embeddings

        self.date_split = None
        self.stock_list = None
        self.stocks_train = None
        self.stocks_val = None

    def prepare_data(self):
        train = pd.read_parquet(smart_open(self.path + 'data/train.parquet')).astype(np.float32)
        val = pd.read_parquet(smart_open(self.path + 'data/test.parquet')).astype(np.float32)
        self.stock_list = val.columns.droplevel(1).unique()
        train.index = train.index.astype(str)
        val_tickers = np.random.choice(self.stock_list, self.num_val, replace=False)
        self.stocks_train = train
        self.stocks_val = val[val_tickers]

    def train_dataloader(self):
        train_set = DS(self.stocks_train, self.back_length, self.embeddings)
        sampler = RandomSampler(train_set, False, self.total_train * self.batch_size)
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=4, drop_last=True, sampler=sampler)

    def val_dataloader(self):
        val_set = DS(self.stocks_val, self.back_length, self.embeddings)
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
