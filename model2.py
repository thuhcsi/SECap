import torch
import torch.nn as nn
import lightning.pytorch as pl
from module.Qformer import BertConfig, BertLMHeadModel
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    BertTokenizer, 
    BertModel,
    LlamaTokenizer
)
from module.modeling_llama import LlamaForCausalLM
from CLUB_modules.mi_estimators import *
from tool.get_sentence_simi import SimiCal
import torch.nn.functional as F
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
import os

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

class MotionAudio(pl.LightningModule):
    def __init__(
        self,
        hubert_ckpt="weights/models--TencentGameMate--chinese-hubert-large/snapshots/90cb660492214f687e60f5ca509b20edae6e75bd",
        text2vec_ckpt="weights/models--shibing624--text2vec-base-chinese/snapshots/26420fdf61ddfd92fafbaf3bc21a7c06b1812248",
        llama_ckpt="weights/models--minlik--chinese-llama-7b-merged/snapshots/1ca4d87576f1fef4d44a949fb65bbe6b96675872"):
        super(MotionAudio,self).__init__()
        
        #path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        hubert_ckpt = os.path.join(current_directory, hubert_ckpt)
        text2vec_ckpt = os.path.join(current_directory, text2vec_ckpt)
        llama_ckpt = os.path.join(current_directory, llama_ckpt)

        #hubert
        self.hubert_model=HubertModel.from_pretrained(hubert_ckpt)
        self.hubert_feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained(hubert_ckpt)
        #text2vec
        self.text2vec_model=BertModel.from_pretrained(text2vec_ckpt)
        self.text2vec_tokenizer=BertTokenizer.from_pretrained(text2vec_ckpt)


        #llama
        self.llama_model=LlamaForCausalLM.from_pretrained(llama_ckpt, torch_dtype="auto")
        #self.llama_model = self.llama_model.to(torch.float32)
        self.llama_tokenizer=LlamaTokenizer.from_pretrained(llama_ckpt)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        #self.llama_model.model.resize_token_embeddings(len(self.llama_tokenizer))

        for p in self.parameters():
            p.requires_grad = False
        #Qformer
        self.audio_Qformer,self.audio_query_tokens=self.init_Qformer(num_query_token=32,vision_width=768)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        self.audio_project=nn.Linear(1024,768)

        self.audio_llama_project=nn.Linear(768,4096)

        
        
    def init_Qformer(self,num_query_token, vision_width, cross_attention_freq=2):
        path=os.path.dirname(os.path.abspath(__file__))
        config_path=os.path.join(path,"weights/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55")
        encoder_config = BertConfig.from_pretrained(config_path)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        ckpt=os.path.join(path,"weights/models--bert-base-chinese/snapshots/8d2a91f91cc38c96bb8b4556ba70c392f8d5ee55/pytorch_model.bin")
        Qformer.load_state_dict(torch.load(ckpt),strict=False)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    


    
    def forward(self, audio, describtion):
        #hubert
        with torch.no_grad():
            audio_feature=self.hubert_feature_extractor(audio, padding=True,return_tensors="pt",sampling_rate=16000).input_values.to(self.device)
            audio_feature = audio_feature.half()
            audio_feature=self.hubert_model(audio_feature).last_hidden_state
        audio_feature=self.audio_project(audio_feature)

        #text2vec
        with torch.no_grad():
            #describtion
            describtion=[s+"</s>" for s in describtion]
            describtion_input=self.text2vec_tokenizer(describtion, padding=True, truncation=True, return_tensors='pt').to(self.device)
            describtion_feature=self.text2vec_model(**describtion_input)
            describtion_feature=self.mean_pooling(describtion_feature,describtion_input['attention_mask']).unsqueeze(1)


        #Qformer
        audio_query_tokens=self.audio_query_tokens.expand(audio_feature.shape[0], -1, -1)
        frame_atts = torch.ones(audio_feature.size()[:-1], dtype=torch.long).to(audio_feature.device)
        #print(audio_query_tokens.shape,audio_feature.shape,frame_atts.shape)
        audio_query_output=self.audio_Qformer.bert(
            query_embeds=audio_query_tokens, #[32,768]
            encoder_hidden_states=audio_feature,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        audio_hidden=audio_query_output.last_hidden_state

        text_tokens=self.llama_tokenizer(describtion, padding="longest", truncation=True, return_tensors='pt',add_special_tokens=False).to(self.device)

        #print(audio_hidden.shape)
        audio_input=self.audio_llama_project(audio_hidden)
        batchsize=audio_input.shape[0]
        bos=torch.ones([batchsize, 1],dtype=text_tokens.input_ids.dtype).to(self.device) * self.llama_tokenizer.bos_token_id
        bos_embeds=self.llama_model.model.embed_tokens(bos.to(self.device))
        #in training, we use different prompts for each audio
        prompts=[ "请用一句话用中文表述音频中说话人的情感状态：", "请用一句中文概括音频中讲话者的情感：", "请用一句中文简述音频里说话者的情感表现：", "请用一句中文概述所给音频中说话人的情感：", "请用一句话用中文描述音频中说话人的情感：", "请用一句中文描绘音频中说话者的情感：", "请用一句中文描述所给音频中说话人的情感：", "请用一句中文简要表述音频中说话人的情感：", "请用一句中文概括所给音频中说话者的情感：", "请用一句话用中文描述所给音频中说话人的情感：", "请用一句中文简述所给音频里说话者的情感：", "请用一句中文描述音频中讲话者的情感：", "请用一句中文概述音频中说话人的情感：", "请用一句话用中文表达音频中说话者的情感：", "请用一句中文简要描述音频中说话人的情感：", "请用一句中文概括音频中说话人的情感：", "请用一句中文描述所给音频中讲话者的情感：", "请用一句中文简述音频中说话者的情感：", "请用一句中文概述所给音频中讲话者的情感：", "请用一句话用中文描述音频中讲话者的情感：", "请用一句中文描述音频中说话人的情感状态：", "请用一句中文概括所给音频里说话者的情感：", "请用一句中文简述所给音频中说话人的情感表现：", "请用一句中文概述音频里说话者的情感：", "请用一句话用中文描述音频中说话人的情感表现：", "请用一句中文描绘所给音频中说话者的情感：", "请用一句中文描述音频里讲话者的情感：", "请用一句中文简要表述所给音频中说话人的情感：", "请用一句中文概括音频里说话者的情感：", "请用一句话用中文描述所给音频中讲话者的情感：" ]
        import random
        prompt=prompts[random.randint(0,len(prompts)-1)]
        prompts_id=self.llama_tokenizer(prompt,return_tensors='pt').input_ids.to(self.device)
        prompts_id=prompts_id.expand(batchsize,-1)
        prompts_embeds=self.llama_model.model.embed_tokens(prompts_id)

        
        targets=text_tokens.input_ids.masked_fill(
            text_tokens.input_ids==self.llama_tokenizer.pad_token_id,-100
        )
        text_embeds=self.llama_model.model.embed_tokens(text_tokens.input_ids.to(self.device))
        input_embeds=torch.cat([bos_embeds,audio_input,prompts_embeds,text_embeds],dim=1)
        atts_audio=torch.ones(audio_input.size()[:-1], dtype=torch.long).to(audio_input.device)

        #atts_audio=atts_audio.to(self.device)
        attns_text=text_tokens.attention_mask
        attns_bos=atts_audio[:,:1]
        attns_prompt=torch.ones(prompts_embeds.size()[:-1], dtype=torch.long).to(prompts_embeds.device)
        attns=torch.cat([attns_bos,atts_audio,attns_prompt,attns_text],dim=1)
        print(input_embeds.shape,attns.shape,targets.shape)
        outputs=self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=attns,
            labels=targets,
            return_dict=True,
        )
        loss=outputs.loss
        #print(loss)

        return loss
    def training_step(self, batch, batch_idx):
        audio, describtion,_=batch
        loss=self.forward(audio, describtion)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(audio),sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        audio, describtion,_=batch
        loss=self.forward(audio, describtion)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(audio),sync_dist=True)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.000013, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        return optimizer
    def inference(self, audio):
        with torch.no_grad():
            audio_feature=self.hubert_feature_extractor(audio, padding=True,return_tensors="pt",sampling_rate=16000).input_values.to(self.device)
            audio_feature = audio_feature.float()
            audio_feature=self.hubert_model(audio_feature).last_hidden_state
            
            audio_feature=self.audio_project(audio_feature)



        #Qformer
        audio_query_tokens=self.audio_query_tokens.expand(audio_feature.shape[0], -1, -1)
        frame_atts = torch.ones(audio_feature.size()[:-1], dtype=torch.long).to(audio_feature.device)
        audio_query_output=self.audio_Qformer.bert(
            query_embeds=audio_query_tokens, #[32,768]
            encoder_hidden_states=audio_feature,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        audio_hidden=audio_query_output.last_hidden_state

        #print(audio_hidden.shape)
        audio_input=self.audio_llama_project(audio_hidden)

        batchsize=audio_input.shape[0]
        #in inference, we use the same prompt for all audio
        #prompts=[ "请用一句话用中文表述音频中说话人的情感状态：", "请用一句中文概括音频中讲话者的情感：", "请用一句中文简述音频里说话者的情感表现：", "请用一句中文概述所给音频中说话人的情感：", "请用一句话用中文描述音频中说话人的情感：", "请用一句中文描绘音频中说话者的情感：", "请用一句中文描述所给音频中说话人的情感：", "请用一句中文简要表述音频中说话人的情感：", "请用一句中文概括所给音频中说话者的情感：", "请用一句话用中文描述所给音频中说话人的情感：", "请用一句中文简述所给音频里说话者的情感：", "请用一句中文描述音频中讲话者的情感：", "请用一句中文概述音频中说话人的情感：", "请用一句话用中文表达音频中说话者的情感：", "请用一句中文简要描述音频中说话人的情感：", "请用一句中文概括音频中说话人的情感：", "请用一句中文描述所给音频中讲话者的情感：", "请用一句中文简述音频中说话者的情感：", "请用一句中文概述所给音频中讲话者的情感：", "请用一句话用中文描述音频中讲话者的情感：", "请用一句中文描述音频中说话人的情感状态：", "请用一句中文概括所给音频里说话者的情感：", "请用一句中文简述所给音频中说话人的情感表现：", "请用一句中文概述音频里说话者的情感：", "请用一句话用中文描述音频中说话人的情感表现：", "请用一句中文描绘所给音频中说话者的情感：", "请用一句中文描述音频里讲话者的情感：", "请用一句中文简要表述所给音频中说话人的情感：", "请用一句中文概括音频里说话者的情感：", "请用一句话用中文描述所给音频中讲话者的情感：" ]
        prompt="请用一句中文简述音频里说话者的情感表现："
        import random
        #prompt=prompts[random.randint(0,len(prompts)-1)]
        
        prompts_id=self.llama_tokenizer(prompt,return_tensors='pt').input_ids.to(self.device)
        prompts_id=prompts_id.expand(batchsize,-1)
        prompts_embeds=self.llama_model.model.embed_tokens(prompts_id)

        bos=torch.ones([batchsize, 1],dtype=torch.int64).to(self.device) * self.llama_tokenizer.bos_token_id
        bos_embeds=self.llama_model.model.embed_tokens(bos.to(self.device))
        embeds=torch.cat([bos_embeds,audio_input,prompts_embeds],dim=1)
        #print(embeds.dtype)
        embeds=embeds.half()
        outputs1=[]

        # you may change the num of generated sentences here and change the parameters of llama to get better results such as top_k, top_p, num_beams
        # to reduce randomness,we do this for 8 times and get 8 sentences
        # then we calculate the similarity between each sentence and the other 7 sentences, and remove the 3 sentences with the lowest average similarity
        for i in range(8):
            with torch.no_grad():
                outputs=self.llama_model.generate(
                    inputs_embeds=embeds,
                    max_new_tokens=50,
                    min_new_tokens=3,
                    do_sample=True,
                    top_k=10,
                    top_p=0.95,
                    num_beams=5,
                    repetition_penalty=10.0,
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    eos_token_id=self.llama_tokenizer.eos_token_id,
                    #stopping_criteria=stopping_criteria,
                    early_stopping=True,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,

                )
        #print(outputs)
            output_tokens=self.llama_tokenizer.batch_decode(outputs,skip_special_tokens=True)
            #output_tokens=self.post_processing(output_tokens)
            print(output_tokens)
            outputs1.append(output_tokens[0])
        outputs1=self.post_processing(outputs1,self.device)

        return outputs1,prompt
    
    def post_processing(self, sentences,device):
        similarities = np.zeros((len(sentences), len(sentences)))
        simi_cal=SimiCal(device=device)
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                similarities[i, j] = simi_cal(sentences[i], sentences[j])
        avg_similarities = np.mean(similarities, axis=1)
        least_related_indices=avg_similarities.argsort()[:3]
        remaining_sentences = [sentences[i] for i in range(len(sentences)) if i not in least_related_indices]

        
        return remaining_sentences
    def test_step(self, batch, batch_idx):
        audio,_,describtion,fpath=batch
        output_tokens,prompt=self.inference(audio)
        test_file="result.txt"
        with open(test_file,"a",encoding="utf-8") as f:
            f.write("file: "+fpath[0]+"\n")
            #f.write("prompt: "+prompt+"\n")
            f.write("origin: "+describtion[0]+"\n")
            f.write("result: "+output_tokens[0]+"\n")
            f.write("result2: "+output_tokens[1]+"\n")
            f.write("result3: "+output_tokens[2]+"\n")
            f.write("result4: "+output_tokens[3]+"\n")
            f.write("result5: "+output_tokens[4]+"\n")
            f.write("\n")
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model=MotionAudio()
    print(count_parameters(model))

