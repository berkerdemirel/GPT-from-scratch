import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import Dict, Union, Tuple


class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            text: str, 
            characters: int, 
            block_size: int, 
            train: bool=True
        ):
        super(ShakespeareDataset, self).__init__()
        self.text = text
        self.characters = characters
        str_to_int_dict = {s:i for i,s in enumerate(self.characters)}
        int_to_str_dict = {i:s for i,s in enumerate(self.characters)}
        self.encoder = lambda s: [str_to_int_dict[c] for c in s]
        self.decoder = lambda l: ''.join([int_to_str_dict[i] for i in l])
        self.data = torch.tensor(self.encoder(self.text), dtype=torch.long)
        self.block_size = block_size
        self.train = train

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        idx = index # torch.randint(len(self.data) - self.block_size, size=(1,))
        if self.train:
            idx = torch.randint(len(self.data) - self.block_size, size=(1,))
        X = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        text = self.text[idx:idx+self.block_size]
        sample = {"X": X, "y": y, "text": text}
        return sample

    def __len__(self) -> int:
        if self.train:
            return 5000
        return len(self.data) - self.block_size



if __name__ == "__main__":
    pth = "input.txt"
    with open(pth, 'r', encoding='utf-8') as f:
        text = f.read()
    
    characters = sorted(list(set(text)))
    block_size = 256
    dataset = ShakespeareDataset(text, characters, block_size, train=False)
    print("len", len(dataset))
    print("sample", dataset[0]["X"].shape)
    print("sample", dataset[0]["y"].shape)
    print("sample", len(dataset[0]["text"]))
    print("sample\n", dataset[0]["text"][:150])