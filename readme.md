# SECAP: Speech Emotion Captioning with Large Language Model

This repository contains the implementation of the paper "SECap: Speech Emotion Captioning with Large Language Model". It includes the model code, training and testing scripts, and a test dataset. The test dataset consists of 600 wav audio files and their corresponding emotion descriptions.

## Installation

To install the project dependencies, use the following command:

conda create --name secap --file requirements.txt

## Pretrained Model
You can download the pretrained model from [here](https://drive.google.com/file/d/1-0Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z/view?usp=sharing) and put it in the `EmotionCaption` folder.

Also, you need to download the pretrained weights folder from [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt) and put it in the `EmotionCaption` folder.

## Inference and Testing

If you want to test the model on your own data, use the `inference.py` script. For example:
```
python inference.py --wavdir /path/to/your/audio.wav
```


If you want to test the model on the provided test dataset of 600 audio files and their emotion descriptions, use the `test.py` script. For example:

```
python test.py 
```


## Training

If you want to train the model, use the `train.py` script. But first, you need to create a training dataset. The training dataset should be a folder containing audio files and their corresponding emotion descriptions.
For example:

```
python train.py 
```


## Calculating Similarity

If you want to calculate the similarity between the generated descriptions and the ground truth descriptions, use the `tool/get_sentence_simi.py` script. For example:

```
cd tool
# modify the path in get_sentence_simi.py
python get_sentence_simi.py
```
## Citation

If you use this repository in your research, please cite our paper:

@article{SECap,
  title={SECap: Speech Emotion Captioning with Large Language Model},
  
}
