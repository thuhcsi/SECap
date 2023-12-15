# SECAP: Speech Emotion Captioning with Large Language Model

![model](picture/model.png)

This repository contains the implementation of the paper "SECap: Speech Emotion Captioning with Large Language Model". It includes the model code, training and testing scripts, and a test dataset. The test dataset consists of 600 wav audio files and their corresponding emotion descriptions.

## Datset
We public the 600-test dataset in the `dataset` folder. The dataset consists of 600 wav audio files and their corresponding emotion descriptions. 

The `dataset` folder contains the following files:

`wav`: the folder contains 600 wav audio files.

`text.txt`: the file contains the transcribtion of the 600 audio files.

`fid2captions.json`: the file contains the emotion captions of the 600 audio files.

## Download
You can simply download the repository by using the following command:

```
git clone https://github.com/thuhcsi/SECap.git
```

## Installation

To install the project dependencies, use the following command:
```
conda env create -f environment.yml
```

## Pretrained Model
You need to download two fils to run the code. The first one is the `model.ckpt`, which is the pretrained model of the SECaps. The second one is the `weights`, which is the required weights of dependencies.

If you download the `weights.7z`, you need to unzip it.

Then you need to put the `model.ckpt` file and `weights` folder in the main folder(SECap).

---

You can download the `model.ckpt` from [kuake pandownload](https://pan.quark.cn/s/1c3deee6cd68) or [google drive](https://drive.google.com/file/d/11q_ZXj0wXnLcw3nH-m8LyI_egMMDaMvR/view?usp=drive_link) and put it in the main folder.

You can download the `weights` from [kuake pandownload](https://pan.quark.cn/s/53891d06c3db) or [google drive](https://drive.google.com/file/d/1wDmijmOhvpgCXdlwvJ_XWm4QsPzLJxFk/view?usp=drive_link) and put it in the main folder.

---

Meanwhile. we have provided the pretrained checkpoints in the huggingface model hub. You can also download from [here](https://huggingface.co/yaoxunxu/SECaps) and put it in the main folder.

---


## Inference and Testing

If you want to test the model on your own data, use the `inference.py` script. For example:

```
cd scripts
python inference.py --wavdir /path/to/your/audio.wav
```


If you want to test the model on the provided test dataset of 600 audio files and their emotion descriptions, use the `test.py` script. For example:

```
cd scripts
python test.py 
```


## Training

If you want to train the model, use the `train.py` script. But first, you need to create a training dataset. The training dataset should be a folder containing audio files and their corresponding emotion descriptions.
For example:

```
cd scripts
python train.py 
```


## Calculating Similarity
We use the sentence similarity to evaluate the generated descriptions.

Specifically, we generate the descriptions of the 600 audio files for 8 times and calculate the similarity between each sentence and the other 7 sentences, and remove the 3 sentences with the lowest average similarity. We use these 5 sentences as the final generated descriptions and calculate the similarity between the generated descriptions and the ground truth descriptions.

If you want to calculate the similarity between the generated descriptions and the ground truth descriptions, use the `tool/get_sentence_simi.py` script. For example:

```
cd tool
# modify the path in get_sentence_simi.py
python get_sentence_simi.py
```

You can also use the `tool/get_sentence_simi.py` script to calculate the similarity between the generated descriptions and the ground truth descriptions of your own data. 

## Result
You can find your result in the `result` folder.

We also provide one of our results in the `result` folder,which is result.txt. 

It uses the prompt"请用中文用一句话描述上面给出的音频中说话人的情感：",You can use one of the training prompts or your own prompt.

## Citation

If you use this repository in your research, please kindly cite our paper:

@article{SECap,
  title={SECap: Speech Emotion Captioning with Large Language Model},
  
}
