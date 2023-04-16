from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from transformers.optimization import get_constant_schedule_with_warmup
from lightning.pytorch.loggers import NeptuneLogger
from torch_optimizer import Lamb
from torch.optim import AdamW
from utils_data import *
from models import *
from keys import *
import argparse
import neptune
import boto3
import uuid


class EmbeddingTrainer(pl.LightningModule):
    def __init__(self, optimizer_params, warmup_steps, max_epochs, **kwargs):
        super(EmbeddingTrainer, self).__init__()
        self.save_hyperparameters()
        self.optimizer_params = optimizer_params
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.TFEncoder = Tradeformer(**kwargs)
        #self.lstm = LSTM([300, 5, 36])

    def forward(self, x, embd):
        x = self.TFEncoder(x, embd)
        #x = self.lstm(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, embd = batch
        output = self(x, embd)
        loss = sharpe_loss(output, y)
        self.log('train_score', -loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, embd = batch
        output = self(x, embd)
        loss = sharpe_loss(output, y)
        self.log('perfect_score', -sharpe_loss(output, y, perfect=True).item())
        self.log('val_score', -loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.optimizer_params, amsgrad=True)
        scheduler = get_constant_schedule_with_warmup(optimizer, self.warmup_steps)
        return [optimizer], [scheduler]


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    run = neptune.init_run(
        api_token=NEPTUNE_API_TOKEN,
        project='lei-research/ASEC',)
    uuid = str(uuid.uuid4())
    run['uuid'] = uuid
    neptune_logger = NeptuneLogger(run=run)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--back_length', type=int, default=300)
    parser.add_argument('--a2v_dim', type=int, default=50)
    parser.add_argument('--embd_meth', type=str, default='dict')
    args, _ = parser.parse_known_args()

    OPTIM_PARAMS = {
        'lr': args.learning_rate,
    }

    dm = DataModule(
        back_length=args.back_length,
        num_val=50,
        batch_size=args.batch_size,
        total_train=2000,
    )

    model = EmbeddingTrainer(
        optimizer_params=OPTIM_PARAMS,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        dims=[args.back_length, 5, 36],
        a2v_dim=args.a2v_dim,
        n_encoders=6,
        n_decoders=6,
        n_heads=6,
    )

    checkpoint = ModelCheckpoint(
        filename='test',
        monitor='val_score',
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.n_gpu,
        max_epochs=args.max_epochs,
        logger=neptune_logger,
        callbacks=[checkpoint],
        fast_dev_run=False,
    )

    trainer.fit(model, dm)
    s3 = boto3.client('s3')
    s3.upload_file(checkpoint.best_model_path, 'asset2vec', f'checkpoints/{uuid}.ckpt')
