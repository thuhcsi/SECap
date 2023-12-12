import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import json
import random
import os
class AudioMotionDataset(Dataset):
    def __init__(self, text_file, wav_scp_file,description_file):
        self.transcriptions = {}
        with open(text_file, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                utt_id = parts[0]
                transcription = parts[1]
                self.transcriptions[utt_id] = transcription
        with open(description_file, 'r') as f:
            self.description = json.load(f)

        self.wav_paths = []
        path=os.path.dirname(os.path.abspath(__file__))
        path=os.path.dirname(path)
        with open(wav_scp_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                utt_id = parts[0]
                wav_path = os.path.join(path,parts[1])
                if utt_id in self.description:
                    self.wav_paths.append((utt_id, wav_path))

    def __getitem__(self, index):
        utt_id, wav_path = self.wav_paths[index]
        describs=self.description[utt_id]
        describ=describs
        transcription = self.transcriptions[utt_id]
        return wav_path, transcription,describ

    def __len__(self):
        return len(self.wav_paths)
import soundfile as sf


def collate_fn(batch):
    wav_paths, transcriptions,describ = zip(*batch)
    waveforms = []
    trans=[]
    describs=[]
    paths=[]
    for wav,tran,des in zip(wav_paths,transcriptions,describ):
        path=wav.split('/')[-1]
        paths.append(path)
        waveform, sample_rate = sf.read(wav)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(torch.tensor(waveform).unsqueeze(0).to(torch.float32)).squeeze(0).numpy()        #print(sample_rate)
        waveforms.append(waveform)
        trans.append(tran)
        describs.append(des)    
    return waveforms,trans,describs, paths
import time
if __name__ == '__main__':
    batch_size = 32
    time1=time.time()
    AM_Dataset = AudioMotionDataset("../dataset/text.txt","../dataset/wav.scp","../dataset/fid2captions.json")
    AM_Dataloader = DataLoader(AM_Dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    data=[]
    wavforms=[]
    trans=[]
    describs=[]
    for batch_idx, (waveforms,trans,describs,_) in enumerate(AM_Dataloader):
        print(batch_idx, trans,describs)
    print(time.time()-time1)
        
    
    
