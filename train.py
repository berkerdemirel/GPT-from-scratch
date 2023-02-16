import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from typing import Dict, Union, Tuple
from model import GPT
from utils import download_data, return_dataset


def train_one_epoch(
        train_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str
    ) -> Dict[str, Union[torch.tensor, float]]:
 
    start = time.time()
    model.train()
    losses = torch.zeros(len(train_loader))
    for i, sample in enumerate(train_loader):
        X = sample["X"].to(device)
        y = sample["y"].to(device)
        text = sample["text"]
        logits = model(X)
        loss = criterion(logits, y.view(-1,))
        losses[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    time_elapsed = time.time() - start
    train_info = {"loss": torch.mean(losses), "time": time_elapsed}
    return train_info


def test_one_epoch(
        test_loader: torch.utils.data.DataLoader, 
        model: torch.nn.Module, 
        criterion: torch.nn.Module, 
        device: str
    ) -> Dict[str, Union[torch.tensor, float]]:

    start = time.time()
    model.eval()
    losses = torch.zeros(len(test_loader))
    with torch.inference_mode():
        for i, sample in enumerate(test_loader):
            X = sample["X"].to(device)
            y = sample["y"].to(device)
            text = sample["text"]
            logits = model(X)
            loss = criterion(logits, y.view(-1,))
            losses[i] = loss.item()
    time_elapsed = time.time() - start
    test_info = {"loss": torch.mean(losses), "time": time_elapsed}
    return test_info


def generate_text(
        model: torch.nn.Module, 
        device: str, 
        num_tokens: int
    ):

    idx = torch.zeros((1,1), dtype=torch.long).to(device)
    print(train_set.decoder(model.generate(idx, num_tokens)[0].tolist()))


if __name__ == "__main__":
    # Config Params
    data_path = "./input.txt"
    load_path = None
    epochs = 50
    block_size = 256
    split = 0.9
    batch_size = 64
    initial_lr = 3e-4
    min_lr = 1e-4
    evaluate_every = 10
    n_embed = 384
    num_heads = 6
    n_layers = 6
    device_id = 0
    checkpoint_dir = "./results/"

    # download data
    download_data(data_path)

    # create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True) # create directory

    # read dataset
    train_set, test_set = return_dataset(data_path, split, block_size)

    # create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # set device
    device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu')

    # create model
    num_chars = len(train_set.characters)
    model = GPT(num_chars, block_size, n_embed, num_heads, n_layers)
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    
    # lr scheduler
    lambda_func = lambda epoch: max(0.99 ** epoch, min_lr / initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    # Training loop    
    best_val_loss = 1e5
    for e in range(epochs):
        train_info = train_one_epoch(train_dataloader, model, criterion, optimizer, scheduler, device)
        print("At epoch: {}, train loss: {:.2f}, in {:.2f} seconds".format(e+1, train_info["loss"], train_info["time"]))
        if (e+1) % evaluate_every == 0:
            test_info = test_one_epoch(test_dataloader, model, criterion, device)
            print("\nAt epoch: {}, test loss: {:.2f}, in {:.2f} seconds\n".format(e+1, test_info["loss"], test_info["time"]))
            # save checkpoint
            if best_val_loss > test_info["loss"]:
                torch.save(model.state_dict(), checkpoint_dir + "model_epoch_{}_loss_{:.2f}.pt".format(e, test_info["loss"]))
                best_val_loss = test_info["loss"]

    # Generate some text
    generate_text(model, device, 500)
