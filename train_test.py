import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm.auto import tqdm

from model import SiAudNet


def train(
        model: SiAudNet,
        train_on_gpu: bool,
        n_epochs: int,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        file_name: str,
        use_scheduler: bool = False,
) -> None:
    print("Training...")

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         "min",
                                                         verbose=True)

    valid_loss_min = np.Inf  # track change in validation loss
    for epoch in range(n_epochs):

        train_loss = 0.0
        valid_loss = 0.0

        ## train
        model.train()
        for data, target in tqdm(train_loader):
            target = target.float()  # BCELogitLoss requires float loss

            if train_on_gpu:
                target = target.cuda()
                data = (data[0].cuda(), data[1].cuda())

            optimizer.zero_grad()

            output = model(data)
            loss = SiAudNet.criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data[0].size(0)

            del loss

        ## validate
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                target = target.float()
                if train_on_gpu:
                    target = target.cuda()
                    data = (data[0].cuda(), data[1].cuda())

                output = model(data)
                loss = SiAudNet.criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data[0].size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        if use_scheduler:
            scheduler.step(valid_loss)

        # print training/validation statistics
        print(
            f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}"
        )

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(
                f"Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving.."
            )
            torch.save(model.state_dict(), file_name)
            valid_loss_min = valid_loss

    model.load_state_dict(torch.load(file_name))


def test(model: SiAudNet, test_on_gpu: bool,
         test_loader: torch.utils.data.DataLoader) -> None:
    print("Testing...")
    # track test loss
    test_loss = 0.0
    classes = ["not match", "match"]
    class_correct = [0, 0]
    class_total = [0, 0]

    if test_on_gpu:
        model = model.cuda()

    sigmoid = nn.Sigmoid()

    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            target = target.float()  # BCELogitLoss requires float loss
            if test_on_gpu:
                target = target.cuda()
                data = (data[0].cuda(), data[1].cuda())

            output = model(data)

            loss = SiAudNet.criterion(output, target)

            test_loss += loss.item() * data[0].size(0)

            pred = sigmoid(output)

            for curr_target, curr_pred in zip(target, pred):
                if curr_target > 0.5:
                    class_correct[1] += 1 if curr_pred > 0.5 else 0
                    class_total[1] += 1
                else:
                    class_correct[0] += 1 if curr_pred <= 0.5 else 0
                    class_total[0] += 1

    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.6f}\n")

    for i, nn_class in enumerate(classes):
        if class_total[i] > 0:
            print(
                f"Test Accuracy of {nn_class+':':11}{class_correct[i] / class_total[i]:.3%} ({np.sum(class_correct[i])}/{np.sum(class_total[i])})"
            )
        else:
            print(f"Test Accuracy of {nn_class+':':11} N/A")

    print(
        f"\nOverall Test Accuracy: {np.sum(class_correct) / np.sum(class_total):.3%} ({np.sum(class_correct)}/{np.sum(class_total)})"
    )
