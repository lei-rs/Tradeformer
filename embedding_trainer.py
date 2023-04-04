from transformers.optimization import get_cosine_with_hard_restarts_schedule_with_warmup
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from smart_open import smart_open
from torch.optim import AdamW
from keys import *
import torch.nn.functional as nnf
import lightning.pytorch as pl
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.columns)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data.iloc[:, idx].to_numpy())


class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, a2v_dim, dropout=0.1):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, a2v_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(a2v_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(a2v_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = nnf.mse_loss(y_hat, x)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        loss = nnf.mse_loss(y_hat, x)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.99))
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, 10, 50, 4)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    np.random.seed(42)
    DATA = pd.read_parquet(smart_open(S3_PATH + 'data/t2k_returns_embd.parquet')).astype(np.float32)[-966:]
    DATA = (DATA - np.mean(DATA, axis=0)) / np.std(DATA, axis=0)
    train_stocks = np.random.choice(DATA.columns, int(0.8 * len(DATA.columns)), replace=False)
    train = DATA[train_stocks]
    val_stocks = DATA.columns[~DATA.columns.isin(train_stocks)]
    val = DATA[val_stocks]
    train_dl = DataLoader(EmbeddingDataset(train), batch_size=8, num_workers=6, shuffle=True)
    val_dl = DataLoader(EmbeddingDataset(val), batch_size=8, num_workers=6, shuffle=False)
    model = Autoencoder(966, 9)
    checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=50, callbacks=[checkpoint], fast_dev_run=False)
    trainer.fit(model, train_dl, val_dl)
