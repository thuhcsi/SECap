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
import argparse
import soundfile as sf
import torchaudio
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path=os.path.dirname(os.path.abspath(__file__))
    path=os.path.dirname(path)
    path=os.path.join(path,'dataset/wav/tx_emotion_00201000015.wav')
    #you can change the wavdir to your own wav file
    parser.add_argument("--wavdir", type=str, default=path)
    model=MotionAudio()

    wavdir=parser.parse_args().wavdir

    wav, sr = sf.read(wavdir)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(wav).unsqueeze(0).to(torch.float32)).squeeze(0).numpy()        #print(sample_rate)
    wavform=[waveform]

    torch.cuda.empty_cache()
    state_dict = torch.load("../model.ckpt",map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model=model.to(torch.device('cuda'))
    model.inference(wavform)