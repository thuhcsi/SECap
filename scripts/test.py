import os
import sys
sys.path.append("..")

from dataloader.dataloader import AudioMotionDataset, collate_fn
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
    
batch_size=1

AM_Dataset = AudioMotionDataset("../dataset/text.txt","../dataset/wav.scp","../dataset/fid2captions.json")
AM_Dataloader = DataLoader(AM_Dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,prefetch_factor=2,persistent_workers=True,num_workers=8)

torch.cuda.empty_cache()
state_dict = torch.load("../model.ckpt",map_location=torch.device('cpu'))["state_dict"]
model.load_state_dict(state_dict,strict=False)
model=model
torch.cuda.empty_cache()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=8,
    precision="32"
    )


trainer.test(model,AM_Dataloader)