from typing import List, Union

import numpy as np
import torch
import torch.nn.functional
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import sys
import json
import os

class SimiCal():
    def __init__(self, device=torch.device('cuda')):        
        # Load model from HuggingFace Hub
        self.device = device
        path=os.path.dirname(os.path.abspath(__file__))
        path=os.path.dirname(path)
        tokenizer_path=os.path.join(path,'weights/simi_berttokenizer')
        model_path=os.path.join(path,'weights/simi_shibing624_text2vec-base-chinese-paraphrase')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertModel.from_pretrained(model_path).to(device)

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cos_sim(self, a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return (a_norm * b_norm).sum(-1)

    def __call__(self,inp1,inp2):
        encoded_input1 = self.tokenizer(inp1, padding=True, truncation=True, return_tensors='pt').to(self.device)
        encoded_input2 = self.tokenizer(inp2, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output1 = self.model(**encoded_input1)
            model_output2 = self.model(**encoded_input2)
        sentence_embeddings1 = self.mean_pooling(model_output1, encoded_input1['attention_mask'])
        sentence_embeddings2 = self.mean_pooling(model_output2, encoded_input2['attention_mask'])
        return self.cos_sim(sentence_embeddings1, sentence_embeddings2)

def test_SimiCal():
    sentences1 = ['如何更换花呗绑定银行卡',] 
    sentences2 = ['花呗更改绑定银行卡',]
    simical = SimiCal()
    print(simical(sentences1, sentences2))
def calculate_mean_variance(lst):
    n = len(lst)
    if n < 1:
        return None

    mean = sum(lst) / n
    variance = sum((x - mean) ** 2 for x in lst) / n

    return mean, variance
def predictSimiWrapper(fpath):
    simical = SimiCal()
    dump_dict = []
    with open(fpath,'r',encoding='UTF-8') as f:
        pred1 = []
        pred2 = []
        pred3 = []
        pred4=[]
        pred5=[]
        gt = [] 
        for line in f:
            if('origin' in line):
                gt.append(''.join(line.strip().replace('origin: ','').strip().split()))
            elif('result:' in line):
                pred1.append(line.strip().replace('result: ','').strip())
            elif('result2' in line):
                pred2.append(line.strip().replace('result2: ','').strip())
            elif('result3' in line):
                pred3.append(line.strip().replace('result3: ','').strip())
            elif('result4' in line):
                pred4.append(line.strip().replace('result4: ','').strip())
            elif('result5' in line):
                pred5.append(line.strip().replace('result5: ','').strip())
    simi_scores1 = []
    simi_scores2 = []
    simi_scores3 = []
    simi_scores4 = []
    simi_scores5 = []
    simi_scores=[]
    steps = 1000
    print(len(gt))
    print(len(pred1))
    print(len(pred2))
    print(len(pred3))
    print(len(pred4))
    print(len(pred5))
    for i in tqdm(range(0,len(gt),steps)):
        try:
            simi_scores1.append(simical(gt[i:i+steps], pred1[i:i+steps]).squeeze())
        except:
            print("error")
            print(len(gt[i:i+steps]))
            print(len(pred1[i:i+steps]))
        try:
            simi_scores2.append(simical(gt[i:i+steps], pred2[i:i+steps]).squeeze())
        except:
            print("error")
            print(len(gt[i:i+steps]))
            print(len(pred2[i:i+steps]))
        try:
            simi_scores3.append(simical(gt[i:i+steps], pred3[i:i+steps]).squeeze())
        except:
            print("error")
            print(len(gt[i:i+steps]))
            print(len(pred3[i:i+steps]))
        try:
            simi_scores4.append(simical(gt[i:i+steps], pred4[i:i+steps]).squeeze())
        except:
            print("error")
            print(len(gt[i:i+steps]))
            print(len(pred4[i:i+steps]))
        try:
            simi_scores5.append(simical(gt[i:i+steps], pred5[i:i+steps]).squeeze())
        except:
            print("error")
            print(len(gt[i:i+steps]))
            print(len(pred5[i:i+steps]))
    simi_scores1 = torch.cat(simi_scores1,0).detach().cpu()
    simi_scores2 = torch.cat(simi_scores2,0).detach().cpu()
    simi_scores3 = torch.cat(simi_scores3,0).detach().cpu()
    simi_scores4 = torch.cat(simi_scores4,0).detach().cpu()
    simi_scores5 = torch.cat(simi_scores5,0).detach().cpu()

    simi_scores.append(simi_scores1.mean().item()*100)
    simi_scores.append(simi_scores2.mean().item()*100)
    simi_scores.append(simi_scores3.mean().item()*100)
    simi_scores.append(simi_scores4.mean().item()*100)
    simi_scores.append(simi_scores5.mean().item()*100)

    acc5=[]
    acc8=[]
    acc6=[]
    acc5.append(((simi_scores1 > 0.5).sum()/simi_scores1.shape[0]).item()*100)
    acc5.append(((simi_scores2 > 0.5).sum()/simi_scores2.shape[0]).item()*100)
    acc5.append(((simi_scores3 > 0.5).sum()/simi_scores3.shape[0]).item()*100)
    acc5.append(((simi_scores4 > 0.5).sum()/simi_scores4.shape[0]).item()*100)
    acc5.append(((simi_scores5 > 0.5).sum()/simi_scores5.shape[0]).item()*100)

    acc8.append(((simi_scores1 > 0.8).sum()/simi_scores1.shape[0]).item()*100)
    acc8.append(((simi_scores2 > 0.8).sum()/simi_scores2.shape[0]).item()*100)
    acc8.append(((simi_scores3 > 0.8).sum()/simi_scores3.shape[0]).item()*100)
    acc8.append(((simi_scores4 > 0.8).sum()/simi_scores4.shape[0]).item()*100)
    acc8.append(((simi_scores5 > 0.8).sum()/simi_scores5.shape[0]).item()*100)

    acc6.append(((simi_scores1 > 0.6).sum()/simi_scores1.shape[0]).item()*100)
    acc6.append(((simi_scores2 > 0.6).sum()/simi_scores2.shape[0]).item()*100)
    acc6.append(((simi_scores3 > 0.6).sum()/simi_scores3.shape[0]).item()*100)
    acc6.append(((simi_scores4 > 0.6).sum()/simi_scores4.shape[0]).item()*100)
    acc6.append(((simi_scores5 > 0.6).sum()/simi_scores5.shape[0]).item()*100)

    simi_avg, simi_std = np.mean(simi_scores), np.std(simi_scores)
    acc5_avg, acc5_std = np.mean(acc5), np.std(acc5)
    acc8_avg, acc8_std = np.mean(acc8), np.std(acc8)
    acc6_avg, acc6_std = np.mean(acc6), np.std(acc6)

    print("simi_avg:{:.2f}+{:.2f}".format(simi_avg, simi_std))
    print("acc5_avg:{:.2f}+{:.2f}".format(acc5_avg, acc5_std))
    print("acc8_avg:{:.2f}+{:.2f}".format(acc8_avg, acc8_std))
    print("acc6_avg:{:.2f}+{:.2f}".format(acc6_avg, acc6_std))

    
    

if __name__=="__main__":
    predictSimiWrapper("../result/result.txt")

