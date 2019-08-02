from data import LibirSet
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch import optim
from model import SiAudNet
from train_test import train, test

sigmoid = nn.Sigmoid()


def is_match(model: SiAudNet, gpu: bool, sample_a: torch.Tensor,
             sample_b: torch.Tensor) -> float:
    with torch.no_grad():
        sample_a = sample_a.unsqueeze(1)
        sample_b = sample_b.unsqueeze(1)
        if gpu:
            sample_a = sample_a.cuda()
            sample_b = sample_b.cuda()
        res = model((sample_a, sample_b))
        return sigmoid(res)


def tester() -> None:
    train_on_gpu = torch.cuda.is_available()
    test_data = torch.load("test_clean.pt")
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=64,
                                              pin_memory=True)

    model = SiAudNet()
    if train_on_gpu:
        model = model.cuda()
    model.load_dict("model_siaudnet.pt")
    test(model, train_on_gpu, test_loader)
    print()


def trainer() -> None:
    train_on_gpu = torch.cuda.is_available()
    batch_size = 32
    n_epochs = 50

    # train_data = LibirSet("./LibriSpeech/train-clean-100", from_csv=True)
    train_data = torch.load("dev_train.pt")
    valid_data = torch.load("dev_valid.pt")

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch_size,
                                               pin_memory=True)

    model = SiAudNet()

    if train_on_gpu:
        model = model.cuda()

    file_name = "model_siaudnet.pt"
    optimizer = optim.Adadelta(model.parameters())
    train(
        model,
        train_on_gpu,
        n_epochs,
        train_loader,
        valid_loader,
        optimizer,
        file_name,
        True,
    )


if __name__ == "__main__":
    #trainer()
    tester()
