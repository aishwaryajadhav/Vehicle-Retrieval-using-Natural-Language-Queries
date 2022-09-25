import json
import os
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision
from datasets import TextTestLoader, ImageTestLoader
from torch.utils.data import DataLoader
from transformers import BertTokenizer,RobertaTokenizer, RobertaModel
import numpy as np
import pandas as pd
import pickle

class EncodingGenerator:
    def __init__(self, data_cfg,json_path):
        self.data_cfg = data_cfg.clone()
        with open(json_path) as f:
            tracks = json.load(f)
        
        self.track_ids =  tracks.keys()

        self.masked_queries = []
        self.raw_frames = []
        self.boxes = []
        self.query_subjects = []
        self.frame_ids = []
        self.subject_ids = []
        self.mq_ids = []
        self.motion_images = []

        for k,v in tracks.items():
            tid = [k] * len(v['frames'])
            self.frame_ids.extend(tid)
            self.raw_frames.extend(v['frames'])
            self.boxes.extend(v['boxes'])

            tid = [k] * len(v['subjects'])
            self.subject_ids.extend(tid)
            self.query_subjects.extend(v['subjects'])

            tid = [k] * len(v['aug_nl'])
            self.mq_ids.extend(tid)
            self.masked_queries.extend(v['aug_nl'])

            self.motion_images.append(k+'.jpg')
    

    def generate_image_encodings(self, model, img_list, id_list, base_path, save_path, boxes = None):

        if(boxes is None):
            resize_to = self.data_cfg.STAGE2MODEL.DATA_SIZE
        else:
            resize_to = self.data_cfg.STAGE1MODEL.DATA_SIZE

        transform_test = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((resize_to,resize_to)),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])

        img_data = ImageTestLoader(img_list,base_path, crop_area=self.data_cfg.DATA.CROP_AREA, transforms=transform_test, boxes=boxes)
        imgloader = DataLoader(dataset=img_data, batch_size = self.data_cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=self.data_cfg.TEST.NUM_WORKERS)

        model.eval()
        encodings = []

        with torch.no_grad():
            for image in imgloader:
                vis_encoding = model.encode_images(image.cuda())

                vis_encoding = vis_encoding.cpu().detach().numpy()
                print(vis_encoding.shape) #should be batch x (512 or 1024)
                encodings.append(vis_encoding)
       
        encodings = list(np.concatenate(encodings, axis = 0))
        df = pd.DataFrame({'ids': id_list, 'encoding': encodings})
        df = df.groupby('ids')['encoding'].apply(np.mean)
        df = df.to_dict()
       
        with open(save_path,'wb') as fs:
            pickle.dump(df, fs)



    def generate_text_encoding(self, model, text_list, id_list, save_path):
        text_data = TextTestLoader(text_list)
        textloader = DataLoader(dataset=text_data, batch_size=self.data_cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=self.data_cfg.TEST.NUM_WORKERS)
        model.eval()

        encodings = []

        with torch.no_grad():
            if self.data_cfg.MODEL.BERT_TYPE == "BERT":
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            elif self.data_cfg.MODEL.BERT_TYPE == "ROBERTA":
                tokenizer = RobertaTokenizer.from_pretrained(self.data_cfg.MODEL.BERT_NAME)

            for text in textloader:
                tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')

                lang_encoding = model.encode_text(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda())

                lang_encoding = lang_encoding.cpu().detach().numpy()
                # print(lang_encoding.shape) #should be batch x (512 or 1024)
                encodings.append(lang_encoding)
              

        encodings = list(np.concatenate(encodings, axis = 0))
        # print(len(encodings))
        df = pd.DataFrame({'ids': id_list, 'encoding': encodings})
        df = df.groupby('ids')['encoding'].apply(np.mean)
        df = df.to_dict()

        # print(len(df.keys()))
       
        with open(save_path,'wb') as fs:
            pickle.dump(df, fs)
        











