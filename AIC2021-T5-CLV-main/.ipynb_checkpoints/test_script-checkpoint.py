import datasets
from torch.utils.data import DataLoader
import config
import importlib
from torchvision.transforms import Compose
from einops.layers.torch import Rearrange, Reduce
from transforms import ToTensor, Resize, Normalize
import torch
from models.video_encoder import R2Plus1D34

transform = Compose([
        ToTensor(),
        Rearrange("t h w c -> c t h w"),
        Resize((224, 224)),
        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])

cfg = config.get_default_config()
train_data=datasets.CityFlowNLVideoDataset(cfg.DATA, json_path = cfg.DATA.TRAIN_JSON_PATH, transform=transform)

trainloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

for batch_idx, batch in enumerate(trainloader):
    print(batch_idx)
    # print(batch)
    break
    
video_frames = batch[0]

print(video_frames.shape)

model = R2Plus1D34()

model.eval()

with torch.no_grad():
    op = model(video_frames)
    
print("Text -> {0}".format(batch[1]))
print("Model Out Shape -> {0}".format(op.shape))