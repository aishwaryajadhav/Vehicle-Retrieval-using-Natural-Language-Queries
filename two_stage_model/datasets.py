import json
import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from utils import get_logger
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

class TextTestLoader(Dataset):
    def __init__(self, text_list):
        self.text = text_list

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        return self.text[index]


class ImageTestLoader(Dataset):
    def __init__(self, image_list, base_path, crop_area = None, transforms = None, boxes = None):
        self.image_paths = image_list
        self.base_path = base_path
        self.transforms = transforms
        self.boxes = boxes
        self.crop_area = crop_area

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        frame_path = os.path.join(self.base_path, self.image_paths[index])
        frame = default_loader(frame_path)
        
        if(self.boxes is not None):
            box = self.boxes[index]
            if self.crop_area == 1.6666667:
                box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
            else:
                box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
            
            frame = frame.crop(box)

        if self.transforms is not None:
            frame = self.transforms(frame)

        return frame

            




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
        
        self.uuid_to_index = {}
        self.index_to_uuid = {}
        self.list_of_tracks = []

        for i, (k,v) in enumerate(tracks.items()):
            self.uuid_to_index[k] = i
            self.index_to_uuid[i] = k
            self.list_of_tracks.append(v)

        self.targets_ohe, self.target_ind = self.process_track_targets(len(self.list_of_tracks))
        
        self.transform = transform
        self.bk_dic = {}
        self._logger = get_logger()
        
        self.all_indexs = list(self.index_to_uuid.keys())
        self.flip_tag = [False]*len(self.list_of_tracks)
        flip_aug = False
        # print(len(self.all_indexs))
        # if flip_aug:
        #     for i in range(len(self.list_of_tracks)):
        #         text = self.list_of_tracks[i]["nl"]
        #         for j in range(len(text)):
        #             nl = text[j]
        #             if "turn" in nl:
        #                 if "left" in nl:
        #                     self.all_indexs.append(i)
        #                     self.flip_tag.append(True)
        #                     break
        #                 elif "right" in nl:
        #                     self.all_indexs.append(i)
        #                     self.flip_tag.append(True)
        #                     break
        # print(len(self.all_indexs))
        # print("data load")

    def process_track_targets(self, n):
        target_lst = []
        target_ind = []
        max_len = -1
        for track in self.list_of_tracks:
            target_oh = torch.zeros(n)
            target_id = []
            targets = track["targets"]
            
            for ut in targets:
                ind = self.uuid_to_index[ut]
                target_oh[ind] = 1
                target_id.append(ind)
            
            if(len(target_id) > max_len):
                max_len = len(target_id)
                
            target_lst.append(target_oh)
            target_ind.append(target_id)
            
        for i, tt in enumerate(target_ind):
            while(len(tt) < max_len):
                tt.append(-1)
            target_ind[i]=tt

        return target_lst, target_ind


    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
   
        tmp_index = self.all_indexs[index]
        # flag = self.flip_tag[index]
        flag=False
        track = self.list_of_tracks[index]
        target = self.targets_ohe[index]
        target_ids = torch.Tensor(self.target_ind[index])
        if self.random:
            nl_idx = int(random.uniform(0, len(track["subjects"])-1))
            # print(len(track["subjects"]))
            # print(nl_idx)
            frame_idx = int(random.uniform(0, len(track["frames"])-1))
        else:
            nl_idx = 0
            frame_idx = 0
        text = track["subjects"][nl_idx]
      
        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, track["frames"][frame_idx])
        
        frame = default_loader(frame_path)
        box = track["boxes"][frame_idx]
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
        
        #Uncomment
        # pdb.set_trace()

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)

        return crop,text,target,target_ids,tmp_index

#Need to modify for new usecase
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
        
        return crop,track["track_id"],track["frames_id"]

####################################Stage 2 Dataset########################################################


class CityFlowNLDataset_Stage2(Dataset):
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

        self.uuid_to_index = {}
        self.index_to_uuid = {}
        self.list_of_tracks = []
        # self.targets = []

        for i, (k,v) in enumerate(tracks.items()):
            self.uuid_to_index[k] = i
            self.index_to_uuid[i] = k
            self.list_of_tracks.append(v)
            # t = v["targets"].index(k)
            # self.targets.append(t)

        self.transform = transform
        # self.bk_dic = {}
        self._logger = get_logger()
        
        self.all_indexs = list(self.index_to_uuid.keys())
        self.flip_tag = [False]*len(self.list_of_tracks)
        flip_aug = False
     
        # print(len(self.all_indexs))
        # print("data load")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
   
        tmp_index = self.all_indexs[index]
        track = self.list_of_tracks[tmp_index]
        targets = track["targets"].copy()

        if self.random:
            nl_idx = int(random.uniform(0, len(track["aug_nl"])-1))
            random.shuffle(targets)
        else:
            nl_idx = 2
        
        text = track["aug_nl"][nl_idx]
        
        #using shuffled (or not) targets and extract index of correct track
        tind = targets.index(self.index_to_uuid[tmp_index])   
        
        batch_lim = 16
        if(len(targets) > batch_lim):
            if(tind >= batch_lim):
                tind_new = random.randint(0, batch_lim-1)
                targets[tind_new] = targets[tind]
                tind = tind_new
            targets = targets[:batch_lim]
            
       
        # if self.index_to_uuid[tmp_index] in self.bk_dic:
        #     bk_list = self.bk_dic[self.index_to_uuid[tmp_index]]
        # else:
        bk_list = []
        for tuid in targets:
            bk = default_loader(self.data_cfg.MOTION_PATH+"/%s.jpg"%tuid)
            bk = self.transform(bk)
            bk_list.append(bk)

        bk_list = torch.stack(bk_list, axis = 0)   

        return bk_list,text,tind,tmp_index




class CityFlowNLDataset_Stage2_TripletLoss(Dataset):
    def __init__(self, data_cfg,json_path,transform = None,Random= True):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = Random
        with open(json_path) as f:
            self.tracks = json.load(f)

        self.uuid_to_index = {}
        self.index_to_uuid = {}
        

        for i, (k,v) in enumerate(self.tracks.items()):
            self.uuid_to_index[k] = i
            self.index_to_uuid[i] = k
            

        # self.data_pairs = []

        # for k,v in self.tracks.items():
        #     for tuid in v['targets']:
        #         if(tuid != k):
        #             self.data_pairs.append((k,tuid))
            
        self.transform = transform
        # self.bk_dic = {}
        self._logger = get_logger()
        
    
    def __len__(self):
        return len(self.index_to_uuid)


    def __getitem__(self, index):
   
        quid = self.index_to_uuid[index]
        # qin, neg_in = self.data_pairs[index]
        query_track = self.tracks[quid]

        if self.random:
            nl_idx = np.random.randint(0, len(query_track["aug_nl"]))
            neg_uid = quid
            
            while(neg_uid == quid and len(query_track["targets"]) > 1):
                neg_uid = np.random.choice(query_track["targets"])
        else:
            nl_idx = 0
            if(query_track['targets'][0] == quid and len(query_track['targets']) > 1):
                neg_uid = query_track['targets'][1]
            else:
                neg_uid = query_track['targets'][0]
          
        
        text = query_track["aug_nl"][nl_idx]
        
        bk_pos = default_loader(self.data_cfg.MOTION_PATH+"/%s.jpg"%quid)
        bk_neg = default_loader(self.data_cfg.MOTION_PATH+"/%s.jpg"%neg_uid)
        
        if(self.transform is not None):
            bk_pos = self.transform(bk_pos)
            bk_neg = self.transform(bk_neg)
            
        return text, bk_pos, bk_neg


    
    

class CityFlowNLDataset_Stage1_TripletLoss(Dataset):
    def __init__(self, data_cfg,json_path,transform = None,Random= True):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        self.crop_area = data_cfg.CROP_AREA
        self.random = Random
        with open(json_path) as f:
            self.tracks = json.load(f)

        self.uuid_to_index = {}
        self.index_to_uuid = {}
        
        self.all_uids = list(self.tracks.keys())
        self.neg_samples = {}
        
        for i, (k,v) in enumerate(self.tracks.items()):
            self.uuid_to_index[k] = i
            self.index_to_uuid[i] = k
            
            neg = k
            while(neg in v['targets']):
                neg = np.random.choice(self.all_uids)
            
            self.neg_samples[k] = neg
            
        
        # self.data_pairs = []

        # for k,v in self.tracks.items():
        #     for tuid in v['targets']:
        #         if(tuid != k):
        #             self.data_pairs.append((k,tuid))
            
        self.transform = transform
        # self.bk_dic = {}
        self._logger = get_logger()
        
    
    def __len__(self):
        return len(self.index_to_uuid)

    def process_frame(self,uid, frame_no):
        frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, self.tracks[uid]["frames"][frame_no])
        
        frame = default_loader(frame_path)
        box = self.tracks[uid]["boxes"][frame_no]
        
        if self.crop_area == 1.6666667:
            box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
        else:
            box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
        
        #Uncomment
        # pdb.set_trace()

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
            
        return crop
            
            
    def __getitem__(self, index):
   
        quid = self.index_to_uuid[index]
        # qin, neg_in = self.data_pairs[index]
        query_track = self.tracks[quid]
     
        if self.random:
            nl_idx = np.random.randint(0, len(query_track["aug_nl"]))
            pos_uid = np.random.choice(query_track["targets"])
            neg_uid = pos_uid
            while(neg_uid == pos_uid):
                neg_uid = np.random.choice(self.all_uids)
                
            pos_frame = np.random.randint(0, len(self.tracks[pos_uid]['frames']))
            neg_frame = np.random.randint(0, len(self.tracks[neg_uid]['frames']))
        else:
            nl_idx = 0
            pos_uid =  quid
            neg_uid = self.neg_samples[quid]
            
            pos_frame = 0
            neg_frame = 0
          
        
        text = query_track["aug_nl"][nl_idx]
        
        bk_pos = self.process_frame(pos_uid, pos_frame)
        bk_neg = self.process_frame(neg_uid, neg_frame)
        
            
        return text, bk_pos, bk_neg

