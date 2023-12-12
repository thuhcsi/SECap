from dataloader1 import AudioMotionDataset, collate_fn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AutoTokenizer
from model2 import MotionAudio
from torch.nn import CrossEntropyLoss
import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import torch.optim as optim
import math


pl.seed_everything(666)
model=MotionAudio()
    
batch_size=16
#put your own dataset and json file here
AM_Dataset = AudioMotionDataset("wav_train.scp","fid2captions.json")
AM_Dataloader = DataLoader(AM_Dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,prefetch_factor=2,persistent_workers=True,num_workers=8)

#create the train and val set
len_train=int(len(AM_Dataset)*0.9)
len_val=len(AM_Dataset)-len_train

train_set,val_set=torch.utils.data.random_split(AM_Dataset,[len_train,len_val],generator=torch.Generator().manual_seed(666))
train_loader=DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,prefetch_factor=2,persistent_workers=True,num_workers=8)
val_loader=DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,prefetch_factor=2,persistent_workers=True,num_workers=8)

#put your own checkpoint dir here
checkpoint_callback = ModelCheckpoint(
        dirpath='yourckpt',
        filename='mymodel2-{epoch:02d}-{train_loss:.5f}',
        save_top_k=20,
        every_n_epochs=5,
        monitor='train_loss',
        mode='min'
    )


trainer = pl.Trainer(
    profiler="simple",
    logger=TensorBoardLogger(name='my_model',save_dir='yourloggerdir'),
    accelerator='gpu',
    max_epochs=10000,
    devices=8,
    log_every_n_steps=50,
    precision="16-mixed",
    callbacks=[checkpoint_callback],
    #accumulate_grad_batches=4,
    strategy="ddp_find_unused_parameters_true"
    )


trainer.fit(model, train_loader, val_loader)