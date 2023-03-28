from transformers.optimization import get_constant_schedule_with_warmup
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from utils_data import *
from models import *
from keys import *
import neptune.new as neptune
import argparse
import boto3
import uuid


class EmbeddingTrainer(pl.LightningModule):
    def __init__(self, model_params, optimizer_params, warmup_steps, max_epochs):
        super(EmbeddingTrainer, self).__init__()
        self.save_hyperparameters()

        self.optimizer_params = optimizer_params
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs

        self.Tradeformer = Tradeformer(**model_params)

    def forward(self, x, ticker):
        return self.Tradeformer(x, ticker)

    def training_step(self, batch, batch_idx):
        x, y, ticker = batch
        output = self(x, ticker)
        loss = sharpe_loss(output, y)
        self.log('train_score', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_perfect', sharpe_loss(output, y, perfect=True).item(), on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, ticker = batch
        output = self(x, ticker)
        loss = sharpe_loss(output, y)
        self.log('val_score', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_perfect', sharpe_loss(output, y, perfect=True).item(), on_step=False, on_epoch=True, prog_bar=True)
        return {'val_score': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params, amsgrad=True, maximize=True)
        scheduler = get_constant_schedule_with_warmup(optimizer, self.warmup_steps)
        return [optimizer], [scheduler]

    def get_embeddings(self):
        self.Tradeformer.embedder.compile_embeds()
        return self.Tradeformer.embedder.embeds


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
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--back_length', type=int, default=300)
    parser.add_argument('--forward_length', type=int, default=1)
    parser.add_argument('--a2v_dim', type=int, default=6)
    args, _ = parser.parse_known_args()

    dm = A2VDataModule(
        back_length=args.back_length,
        forward_length=args.forward_length,
        val_frac=0.3,
        batch_size=args.batch_size,
        total_train=500,
    )

    MODEL_PARAMS = {
        'dims': (args.back_length, 18, 18 + args.a2v_dim, args.forward_length),
        'a2v_dim': args.a2v_dim,
        'n_encoders': 6,
        'n_heads': 6,
        'n_mlp': 1,
        'embd_meth': 'nlp',
        'metadata': A2VDescriptions(),
        'total': 1858,
        'nlp_dims': (48, 6, 1),
    }

    OPTIM_PARAMS = {
        'lr': args.learning_rate,
    }

    model = EmbeddingTrainer(
        MODEL_PARAMS,
        OPTIM_PARAMS,
        args.warmup_steps,
        args.max_epochs,
    )

    checkpoint = ModelCheckpoint(
        filename='test',
        monitor='val_score',
        mode='max',
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
