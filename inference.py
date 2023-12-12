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
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavdir", type=str, default="wavpath")
    model=MotionAudio()
    import soundfile as sf
    import torchaudio

    wavdir=parser.parse_args().wavdir

    wav, sr = sf.read(wavdir)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(torch.tensor(wav).unsqueeze(0).to(torch.float32)).squeeze(0).numpy()        #print(sample_rate)
    wavform=[waveform]

    torch.cuda.empty_cache()
    state_dict = torch.load("model.ckpt",map_location=torch.device('cpu'))["state_dict"]
    model.load_state_dict(state_dict)
    model=model.to(torch.device('cuda'))
    model.inference(wavform)