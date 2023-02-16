import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import ShakespeareDataset
import os
import requests
from typing import Dict, Union, Tuple


def download_data(input_file_path: str):
    if not os.path.exists(input_file_path):
        print("Downloading dataset...")
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
        print("Dataset is downloaded to {}".format(input_file_path))


def return_dataset(
        data_path: int, 
        split: float, 
        block_size: int
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    characters = sorted(list(set(text)))
    dataset_len = len(text)
    train_size = int(dataset_len * split)

    train_text = text[:train_size]
    test_text = text[train_size:]
    
    train_set = ShakespeareDataset(train_text, characters, block_size, train=True)
    test_set = ShakespeareDataset(test_text, characters, block_size, train=False)
    return train_set, test_set


if __name__ == "__main__":
    path = "./input.txt"
    split = 0.8
    block_size = 256

    train, test = return_dataset(path, split, block_size)

    print("train len", len(train))
    print("test len", len(test))

    print("train sample", train.text[:100])
    print("test sample", test.text[:100])