from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
from torch.nn.functional import mse_loss
from utils_data import *
from models import *
from keys import *
import neptune.new as neptune
import argparse
import boto3
import uuid


class EmbeddingTrainer(pl.LightningModule):
    def __init__(self, input_dims, a2v_dim, n_encoders, n_heads, optimizer_params, warmup_steps, max_epochs,
                 embedder, embedding_type, expand_dim=4, activation='gelu', dropout=0.1):
        super(EmbeddingTrainer, self).__init__()
        self.save_hyperparameters()

        hidden_dim = input_dims[1]
        if embedding_type in ('concat'):
            hidden_dim += a2v_dim

        self.optimizer_params = optimizer_params
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.embedder = embedder

        if embedder is None:
            self.embed_dict = torch.randn(1961, a2v_dim).cuda()

        self.embedding_type = embedding_type
        self.TSEncoder = TSEncoderBlock(input_dims, hidden_dim, n_encoders, n_heads, expand_dim, activation, dropout)

    def forward(self, x, input_ids, attention_mask, stock_num):
        if self.embedder is None:
            embedding = self.embed_dict[stock_num]

        else:
            embedding = self.embedder(input_ids, attention_mask)

        if self.embedding_type == 'concat':
            embedding = embedding.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat((x, embedding), dim=-1)
            return self.TSEncoder(x)

        else:
            return self.TSEncoder(x)

    def training_step(self, batch, batch_idx):
        x, mask, mask2, input_ids, attention_mask, stock_num = batch
        output = self(x.masked_fill(mask, 0), input_ids, attention_mask, stock_num)
        loss = mse_loss(output.masked_fill(mask2, 0), x.masked_fill(mask2, 0))
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, mask, mask2, input_ids, attention_mask, stock_num = batch
        output = self(x.masked_fill(mask, 0), input_ids, attention_mask, stock_num)
        loss = mse_loss(output.masked_fill(mask2, 0), x.masked_fill(mask2, 0))
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params, amsgrad=True)
        scheduler = get_cosine_schedule_with_warmup(optimizer, self.warmup_steps, self.max_epochs)
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--back_length', type=int, default=365)
    parser.add_argument('--a2v_dim', type=int, default=30)
    args, _ = parser.parse_known_args()

    roberta_config = RobertaConfig(
        vocab_size=16000,
        hidden_size=384,
        num_hidden_layers=6,
        intermediate_size=1536,
        type_vocab_size=1
    )
    TOKENIZER = RobertaTokenizer.from_pretrained('tokenizer/')
    LM = RobertaModel(config=roberta_config, add_pooling_layer=False)

    embd = Embedder(
        a2v_dim=args.a2v_dim,
        lm=LM,
    )

    OPTIM_PARAMS = {
        'lr': args.learning_rate,
    }

    dm = A2VDataModule(
        back_length=args.back_length,
        val_frac=0.3,
        batch_size=args.batch_size,
        tokenizer=TOKENIZER,
        mask_p=0.8,
    )

    model = EmbeddingTrainer(
        input_dims=[args.back_length, 18],
        a2v_dim=args.a2v_dim,
        n_encoders=6,
        n_heads=12,
        optimizer_params=OPTIM_PARAMS,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs,
        embedder=None,
        embedding_type='concat',
    )

    checkpoint = ModelCheckpoint(
        filename='test',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.n_gpu,
        max_epochs=args.max_epochs,
        logger=neptune_logger,
        callbacks=[checkpoint],
        fast_dev_run=False
    )

    trainer.fit(model, dm)
    s3 = boto3.client('s3')
    s3.upload_file(checkpoint.best_model_path, 'asset2vec', f'checkpoints/{uuid}.ckpt')
