
import json

import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2

import os


class DatasetInput(Dataset):
    def __init__(self, tracks_path):

        self.tracks_path = tracks_path

        file = open(tracks_path)
        self.tracks = json.load(file)
        file.close()

        self.track_names = [name for name in self.tracks]

        self.num_tracks = len(self.track_names)
        file = open(self.tracks_path)
        self.tracks = json.load(file)

    def __len__(self):
        return self.num_tracks

    def __getitem__(self, idx):
        frame_paths = self.tracks[self.track_names[idx]]["frames"]

        return self.imageSetGenerator(frame_paths)  # (2110, 224, 224, 3)

    def imageSetGenerator(self, frame_paths):
        image_list = []
        for filename in frame_paths:
            filename = os.path.join("./data/", filename)
            image_list.append(cv2.resize(cv2.imread(filename), (480, 270)))
        return torch.tensor(image_list)


def batchify(examples):

    max_nframes = 0
    for example in examples:
        max_nframes = max(max_nframes, example.shape[0])

    maskings = []
    image_tensors = []

    for example in examples:
        if example.shape[0] < max_nframes:
            num_pad = max_nframes - example.shape[0]
            example_padded = torch.cat([example, torch.zeros([num_pad] + list(example.shape[1:]))])
            image_tensors.append(example_padded)

            maskings.append([1]*example.shape[0] + [0]*num_pad)
        else:
            image_tensors.append(example)
            maskings.append([1] * example.shape[0])

    batch = {
        'image_tensors': torch.stack(image_tensors, dim=0).permute(0,1,4,2,3),
        'maskings': torch.tensor(maskings)
    }


    return batch