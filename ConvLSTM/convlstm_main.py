import torch
from convlstm_model import ConvLSTM
from convlstm_dataloader import DatasetInput, batchify

import json
import os
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

parser = argparse.ArgumentParser()
parser.add_argument("--tracks_path", default="./data/train_tracks.json",
                    help="path of the tracks json file", type=str)
parser.add_argument("--batch_size", default=32, help="Batch size.", type=int)

parser.add_argument("--input_dim", default=3, help="ConvLSTM: Number of channels in input", type=int)
parser.add_argument("--hidden_dim", default=3, help="ConvLSTM: Number of hidden channels", type=int)
parser.add_argument("--kernel_size", default=9, help="ConvLSTM: Size of kernel in convolutions", type=int)
parser.add_argument("--num_layers", default=1, help="ConvLSTM: Number of LSTM layers stacked on each other", type=int)

parser.add_argument("--gpu_idx", default=0, help="gpu_idx", type=int)

args, unknown = parser.parse_known_args()

# if CUDA available
device = torch.device("cpu")
if torch.cuda.is_available() and (args.gpu_idx is not None):
    gpu_idx = args.gpu_idx
    assert gpu_idx < torch.cuda.device_count()
    print(f"Training on GPU {gpu_idx}")
    device = torch.device(f"cuda:{gpu_idx}")


train_dataset = DatasetInput(args.tracks_path)
train_sampler = RandomSampler(train_dataset)

train_dl = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                      collate_fn=lambda b: batchify(b), drop_last=False)

model = ConvLSTM(args.input_dim, args.hidden_dim, (args.kernel_size, args.kernel_size), args.num_layers,
                 batch_first=True, bias=True, return_all_layers=False)
model.to(device)

# optimizer = torch.optim.Adam(
#     params=model.parameters(),
#     lr=0.001
# )


total = len(train_dl)


# forward
dl = iter(train_dl)  # dataloader
finished = False
for batch in train_dl:
    image_tensors = batch['image_tensors']
    image_tensors = image_tensors.to(device)

    maskings = batch['maskings']
    maskings = maskings.to(device)

    h = model(image_tensors, maskings)
    h = h.reshape(h.shape[0], -1)  # reshape to be 1d vector for each sample, use this h for following tasks
    print(h.shape)
