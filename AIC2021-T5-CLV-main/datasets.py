'''
Video Dataset code reused from -> https://github.com/moabitcoin/ig65m-pytorch/blob/master/ig65m/datasets.py
'''
import json
import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, get_worker_info
import torch.nn.functional as F
import torchvision
from utils import get_logger
import math


def default_loader(path):
    return Image.open(path).convert('RGB')


class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg,json_path,transform = None,Random= True):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = Random
        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        self.bk_dic = {}
        self._logger = get_logger()
        
        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.flip_tag = [False]*len(self.list_of_uuids)
        flip_aug = False
        if flip_aug:
            for i in range(len(self.list_of_uuids)):
                text = self.list_of_tracks[i]["nl"]
                for j in range(len(text)):
                    nl = text[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
        print(len(self.all_indexs))
        print("data load")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
   
        tmp_index = self.all_indexs[index]
        flag = self.flip_tag[index]
        track = self.list_of_tracks[tmp_index]
        if self.random:
            nl_idx = int(random.uniform(0, 3))
            frame_idx = int(random.uniform(0, len(track["frames"])))
        else:
            nl_idx = 2
            frame_idx = 0
        text = track["nl"][nl_idx]
        if flag:
            text = text.replace("left","888888").replace("right","left").replace("888888","right")
        
        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
        
        frame = default_loader(frame_path)
        box = track["boxes"][frame_idx]
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
        

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            if self.list_of_uuids[tmp_index] in self.bk_dic:
                bk = self.bk_dic[self.list_of_uuids[tmp_index]]
            else:
                bk = default_loader(self.data_cfg.MOTION_PATH+"/%s.jpg"%self.list_of_uuids[tmp_index])
                self.bk_dic[self.list_of_uuids[tmp_index]] = bk
                bk = self.transform(bk)
                
            if flag:
                crop = torch.flip(crop,[1])
                bk = torch.flip(bk,[1])
            return crop,text,bk,tmp_index
        if flag:
            crop = torch.flip(crop,[1])
        return crop,text,tmp_index



class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg,transform = None):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        self.crop_area = data_cfg.CROP_AREA
        self.transform = transform
        with open(self.data_cfg.TEST_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        for track_id_index,track in enumerate(self.list_of_tracks):
            for frame_idx, frame in enumerate(track["frames"]):
                frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
                box = track["boxes"][frame_idx]
                crop = {"frame": frame_path, "frames_id":frame_idx,"track_id": self.list_of_uuids[track_id_index], "box": box}
                self.list_of_crops.append(crop)
        self._logger = get_logger()

    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        track = self.list_of_crops[index]
        frame_path = track["frame"]

        frame = default_loader(frame_path)
        box = track["box"]
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
        

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
        if self.data_cfg.USE_MOTION:
            bk = default_loader(self.data_cfg.MOTION_PATH+"/%s.jpg"%track["track_id"])
            bk = self.transform(bk)
            return crop,bk,track["track_id"],track["frames_id"]
        return crop,track["track_id"],track["frames_id"]


class CityFlowNLVideoDataset(Dataset):
    def __init__(self, data_cfg,json_path,transform = None,Random= True):
        """
        Dataset to use video data for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = Random
        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        self.bk_dic = {}
        self._logger = get_logger()
        
        self.all_indexs = list(range(len(self.list_of_uuids)))
        print(len(self.all_indexs))
        print("data load")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):   
        tmp_index = self.all_indexs[index]
        
        track = self.list_of_tracks[tmp_index]
        if self.random:
            nl_idx = int(random.uniform(0, 3))
            frame_idx = int(random.uniform(0, len(track["frames"])))
        else:
            nl_idx = 2
            frame_idx = 0
        text = track["nl"][nl_idx]
        
        video_path = os.path.join(self.data_cfg.CITYFLOW_PATH, self.list_of_uuids[tmp_index])
        
        info = get_worker_info()
        video = cv2.VideoCapture(video_path)

        if info is None:
            rng = FrameRange(video, self.first, self.last)
        else:
            per = int(math.ceil((self.last - self.first) / float(info.num_workers)))
            wid = info.id

            first = self.first + wid * per
            last = min(first + per, self.last)

            rng = FrameRange(video, first, last)

        if self.transform is not None:
            fn = self.transform
        else:
            fn = lambda v: v  # noqa: E731

        return TransformedRange(BatchedRange(rng, self.clip), fn), text, tmp_index


class FrameRange:
    def __init__(self, video, first, last):
        assert first <= last

        for i in range(first):
            ret, _ = video.read()

            if not ret:
                raise RuntimeError("seeking to frame at index {} failed".format(i))

        self.video = video
        self.it = first
        self.last = last

    def __next__(self):
        if self.it >= self.last or not self.video.isOpened():
            raise StopIteration

        ok, frame = self.video.read()

        if not ok:
            raise RuntimeError("decoding frame at index {} failed".format(self.it))

        self.it += 1

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class BatchedRange:
    def __init__(self, rng, n):
        self.rng = rng
        self.n = n

    def __next__(self):
        ret = []

        for i in range(self.n):
            ret.append(next(self.rng))

        return ret


class TransformedRange:
    def __init__(self, rng, fn):
        self.rng = rng
        self.fn = fn

    def __next__(self):
        return self.fn(next(self.rng))