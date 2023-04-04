from transformers.optimization import get_constant_schedule_with_warmup
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import RobertaModel, RobertaConfig
from torch.optim import AdamW
from utils_data import *
import torch.nn.functional as nnf
import lightning.pytorch as pl
import torch.nn as nn
import numpy as np
import torch


class EmbeddingDataset(Dataset):
    def __init__(self, embd, desc):
        self.embd = embd
        self.desc = desc

    def __len__(self):
        return len(self.embd.columns)

    def __getitem__(self, idx):
        ticker = self.embd.columns[idx]
        tokens = self.desc[ticker]['input_ids']
        mask = self.desc[ticker]['attention_mask']
        embd = torch.from_numpy(self.embd[ticker].to_numpy())
        return tokens, mask, embd


class Embedder(pl.LightningModule):
    def __init__(self, a2v_dim, dropout=0.1):
        super(Embedder, self).__init__()
        cfg = RobertaConfig(vocab_size=16000)
        self.lm = RobertaModel(cfg)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768, a2v_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(a2v_dim)

    def forward(self, tokens, mask):
        ret = self.lm(input_ids=tokens, attention_mask=mask)[0][:, 0, :].squeeze(1) #.mean(dim=1)
        ret = self.head(ret).squeeze()
        return self.norm(ret)

    def training_step(self, batch, batch_idx):
        tokens, mask, embd = batch
        y_hat = self(tokens, mask)
        loss = nnf.mse_loss(y_hat, embd)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, mask, embd = batch
        y_hat = self(tokens, mask)
        loss = nnf.mse_loss(y_hat, embd)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = get_constant_schedule_with_warmup(optimizer, 10)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    EMBD = PretrainedEmbeddings(2).embeddings
    DESC = Descriptions().tokenized
    train_stocks = np.random.choice(EMBD.columns, int(0.8 * len(EMBD.columns)), replace=False)
    val_stocks = EMBD.columns[~EMBD.columns.isin(train_stocks)]
    train = EmbeddingDataset(EMBD[train_stocks], DESC[train_stocks])
    val = EmbeddingDataset(EMBD[val_stocks], DESC[val_stocks])
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    val_loader = DataLoader(val, batch_size=32, shuffle=False)
    model = A2V(50)
    checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=50, callbacks=[checkpoint], fast_dev_run=False)
    trainer.fit(model, train, val)
