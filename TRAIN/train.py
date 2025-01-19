import torch.nn as nn
import torch
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader, random_split
from model import StackedConvNet
import random
import tracemalloc
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a wake word detection model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for training.")
    parser.add_argument("--dataset_folder", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--wakeword_folder_name", type=str, default='cpu', required=False, help="The name of the wakeword folder")
    return parser.parse_args()


def add_noise(mfcc, noise_level=0.01):
    noise = torch.randn_like(mfcc) * noise_level
    return mfcc + noise


def time_shift(mfcc, shift_max=10):
    shift = random.randint(-shift_max, shift_max)
    return torch.roll(mfcc, shifts=shift, dims=-1)


def pitch_shift(mfcc, factor=0.1):
    scale = torch.exp(torch.linspace(-factor, factor, mfcc.size(-1)))
    return mfcc * scale.unsqueeze(0)


class FEATURE_INIT(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.transforms = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=80,
            melkwargs={'n_mels': 80, 'win_length': 160, 'hop_length': 80}
        )

        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=80)
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=80)

        self.All = nn.Sequential(
            self.time_masking,
            self.freq_masking
        )

        self.augment = [self.time_masking, self.freq_masking, self.All, add_noise, time_shift, pitch_shift]

    def __call__(self, x, val=False):
        x = self.transforms(x)

        if val:
            return x

        if random.randint(0, 1) == 1:
            return x

        else:
            aug = random.choice(self.augment)
            return aug(x)


class WakeWordDataset(Dataset):
    def __init__(self, DATA_FOLDER, wakeword_name):
        self.wakeword_name = wakeword_name
        self.DATA_FOLDERS = [os.path.join(DATA_FOLDER, i) for i in os.listdir(DATA_FOLDER)]
        self.DATASET = self.get_data(self.DATA_FOLDERS)
        random.shuffle(self.DATASET)

    def get_data(self, folders):
        out = []
        for folder in folders:
            files = os.listdir(folder)
            out.extend([(os.path.join(folder, i), 1 if self.wakeword_name in folder else 0) for i in files])

        return out

    def __len__(self):
        return len(self.DATASET)

    def __getitem__(self, idx):
        return self.DATASET[idx][0], self.DATASET[idx][1]


def collate_fn(batch, val=False):
    waveform = []
    labels = []

    for x, y in batch:
        x, _ = torchaudio.load(x)
        waveform.append(x.mean(0))
        labels.append(int(y))

    waveform = torch.stack(waveform, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    waveform = FEATURE_EXTRACTOR(waveform, val)
    return waveform, labels


def measure_memory(func):
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Memory used: {current / 1024:.2f} KB")
        print(f"Peak memory usage: {peak / 1024:.2f} KB")
        tracemalloc.stop()
        return result
    return wrapper


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@measure_memory
def validation(model, dataset, device):
    batch = list(range(len(dataset)))
    batch = [dataset[i] for i in batch]
    x, y = collate_fn(batch, val=True)

    x = x.to(device)

    with torch.no_grad():
        pred = model(x).squeeze(1)
        pred = pred.tolist()
        pred = torch.tensor([1 if i >= 0.8 else 0 for i in pred])
        pred = pred.type_as(y)

        corr = (pred == y).sum()
        acc = corr / len(y)

        return acc


def train(model, trainData, dataset, loss_fn, optimizer, epochs, device):
    model.train()
    model.to(device)
    prev_loss = float("inf")
    prev_acc = 0

    for epoch in range(epochs):
        losses = []

        for step, (x, y) in enumerate(trainData):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred.squeeze(1), y)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses.append(loss.item())

            print(f"\r EPOCHS: {epoch}/{epochs} - step: {step}/{len(trainData)} - loss: {loss.item()}", end="")

        print("\nAVG loss: ", sum(losses) / len(losses))
        val_acc = validation(model, dataset, device)
        print(f"\n VALIDATION ACC: {val_acc}")

        if sum(losses) / len(losses) < prev_loss and val_acc > prev_acc:
            torch.save(model.state_dict(), f"model.pt")
            prev_loss = sum(losses) / len(losses)
            prev_acc = val_acc

            print("checkpoint saved")


if __name__ == "__main__":
    args = parse_arguments()

    FEATURE_EXTRACTOR = FEATURE_INIT(16_000)
    dataset = WakeWordDataset(args.dataset_folder)

    train_dataset, val_dataset = random_split(dataset, [len(dataset) - 16, 16])

    train_dataLoader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    model = StackedConvNet(
        in_channels=80,
        intermediate_channels=128,
        out_channels=8,
        pool_size=45,
        embed_dim=15,
        num_layers=4
    )

    print(model)
    print(f"shape: {model(torch.randn(1, 80, 401)).shape}")

    loss_fn = nn.L1Loss()
    from transformers.optimization import Adafactor, AdafactorSchedule

    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)

    print(f"num of parameters: {count_parameters(model)}")

    train(
        model,
        train_dataLoader,
        val_dataset,
        loss_fn,
        optimizer,
        args.epochs,
        args.device
    )
