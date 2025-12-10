import os

import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class DataModule(pl.LightningModule):

    def __init__(self, model, optimizer_hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), **self.hparams.optimizer_hparams)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc, prog_bar=True)

def train_model(model, train_loader, test_loader, save_name=None, epochs=10, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join("./", save_name),                         
                         accelerator='auto',
                         max_epochs=epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         enable_model_summary=False)
    trainer.logger._default_hp_metric = None

    model = DataModule(model, **kwargs)
    trainer.fit(model, train_loader, test_loader)

    val_result = trainer.test(model, test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result

def test_model(model, test_loader, save_name=None, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join("./", save_name),                         
                         accelerator='auto',
                         max_epochs=1)

    model = DataModule(model, **kwargs)

    val_result = trainer.test(model, test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result